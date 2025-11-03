import argparse
from contextlib import contextmanager
import json
import multiprocessing
import os
import time
from pathlib import Path
import logging

from comet_ml import Experiment
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint import load, save

import tqdm
import datasets
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

# fixes for reset_parameters not existing
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding


def reset_rope(self: LlamaRotaryEmbedding):
    self.inv_freq, self.attention_scaling = self.rope_init_fn(
        self.config, self.inv_freq.device
    )
    self.original_inv_freq = self.inv_freq


LlamaRMSNorm.reset_parameters = lambda self: torch.nn.init.ones_(self.weight)
LlamaRotaryEmbedding.reset_parameters = reset_rope


LOGGER = logging.getLogger(__name__)


@record
def main():
    parser = _get_parser()
    args = parser.parse_args()

    rank = int(os.getenv("RANK", "0"))
    local_rank = rank % torch.cuda.device_count()
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(rank=rank, world_size=world_size, device_id=device)

    logging.basicConfig(
        format=f"[rank={rank}] [%(asctime)s] %(levelname)s:%(message)s",
        level=logging.INFO,
    )

    LOGGER.debug(os.environ)
    LOGGER.debug(args)
    LOGGER.debug(f"local_rank={local_rank} rank={rank} world size={world_size}")

    # Initialize Comet ML experiment (only on rank 0)
    experiment = None
    if args.experiment_name is not None and rank == 0:
        experiment = Experiment(
            project_name=args.comet_project_name,
            workspace=args.comet_workspace,
            auto_metric_logging=True,
            auto_param_logging=True,
            log_code=True,
            log_graph=True,
            auto_histogram_weight_logging=False,
            auto_histogram_gradient_logging=False,
            auto_histogram_activation_logging=False,
        )
        experiment.set_name(f"{args.experiment_name}-{world_size}gpu")
        experiment.add_tags(["fsdp", "llm-training", "pytorch", "distributed"])

        # Log hyperparameters
        experiment.log_parameters(
            {
                "model_name": args.model_name,
                "dataset_name": args.dataset_name,
                "dataset_subset": args.dataset_subset,
                "batch_size": args.batch_size,
                "seq_length": args.seq_length,
                "learning_rate": args.lr,
                "num_epochs": args.num_epochs,
                "seed": args.seed,
                "log_freq": args.log_freq,
                "ckpt_freq": args.ckpt_freq,
                "optimizer": "AdamW",
                "lr_scheduler": "CosineAnnealingLR",
                "dtype": "bfloat16",
                "save_dir": args.save_dir,
                "world_size": world_size,
                "rank": rank,
                "local_rank": local_rank,
                "cpu_offload": args.cpu_offload,
            }
        )

        LOGGER.info(f"Comet ML experiment initialized: {experiment.url}")

    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    # NOTE: meta device will not allocate any memory
    model: torch.nn.Module
    with rank0_first(), torch.device("meta"):
        config = AutoConfig.from_pretrained(args.model_name, use_cache=False)
        model = AutoModelForCausalLM.from_config(
            config, dtype=dtype, attn_implementation="flash_attention_2"
        )
    total_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"Training {total_params} model parameters")

    # Log model information to Comet
    if experiment:
        experiment.log_parameters(
            {
                "total_parameters": total_params,
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            }
        )
        experiment.log_text(str(model), metadata={"type": "model_architecture"})

        # Log GPU information
        gpu_props = torch.cuda.get_device_properties(device)
        experiment.log_parameters(
            {
                "gpu_name": gpu_props.name,
                "gpu_memory_gb": gpu_props.total_memory / 1e9,
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
            }
        )
        experiment.log_other("device", str(device))
        experiment.log_other("dtype", str(dtype))

    # fsdp_config = dict(
    #     reshard_after_forward=True,
    #     offload_policy=CPUOffloadPolicy() if args.cpu_offload else None,
    #     mp_policy=MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.bfloat16),
    # )
    # for decoder in model.model.layers:
    #     fully_shard(decoder, **fsdp_config)
    # fully_shard(model, **fsdp_config)

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from functools import partial

    # Auto-wrap policy: wrap each transformer decoder layer
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )

    # Configure activation checkpointing for memory efficiency
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
        apply_activation_checkpointing,
    )

    # Apply activation checkpointing to decoder layers
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
    )

    # Apply activation checkpointing to all wrapped decoder layers
    check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )

    model.to_empty(device="cpu" if args.cpu_offload else device)
    model.apply(
        lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
    )
    initial_mem_stats = get_mem_stats(device)
    LOGGER.info(f"Initialized model uses {initial_mem_stats['curr_alloc_gb']}gb")

    # Log initial memory stats to Comet
    if experiment:
        experiment.log_metrics(
            {
                "initial_memory_gb": initial_mem_stats["curr_alloc_gb"],
                "total_gpu_memory_gb": initial_mem_stats["total_gb"],
            },
            step=0,
        )

    # NOTE: since this can download data, make sure to do the main process first
    # NOTE: This assumes that the data is on a **shared** network drive, accessible to all processes
    with rank0_first():
        train_data = _load_and_preprocess_data(args, config)
    LOGGER.debug(f"{len(train_data)} training samples")

    # Log dataset information to Comet
    if experiment:
        experiment.log_parameters(
            {
                "num_training_samples": len(train_data),
            }
        )

    dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=2,
        # NOTE: this sampler will split dataset evenly across workers
        sampler=DistributedSampler(train_data, shuffle=True, drop_last=True),
    )
    LOGGER.debug(f"{len(dataloader)} batches per epoch")

    # Log dataloader information to Comet
    if experiment:
        experiment.log_parameters(
            {
                "batches_per_epoch": len(dataloader),
                "total_training_steps": len(dataloader) * args.num_epochs,
            }
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=1000, eta_min=args.lr * 1e-2
    )

    is_experiment = False
    exp_dir: Path = Path(args.save_dir)
    if args.experiment_name is not None:
        is_experiment = True
        exp_dir = exp_dir / args.experiment_name

    # NOTE: full_state_dict=False means we will be saving sharded checkpoints.
    ckpt_opts = StateDictOptions(full_state_dict=False, cpu_offload=True)

    # attempt resume
    state = {
        "epoch": 0,
        "global_step": 0,
        "epoch_step": 0,
        "running_loss": 0,
    }
    resumed = False
    if is_experiment and (exp_dir / "state.json").exists():
        sharded_model_state, sharded_optimizer_state = get_state_dict(
            model, optimizer, options=ckpt_opts
        )
        load(
            dict(model=sharded_model_state, optimizer=sharded_optimizer_state),
            checkpoint_id=exp_dir / "checkpoint",
        )
        set_state_dict(
            model,
            optimizer,
            model_state_dict=sharded_model_state,
            optim_state_dict=sharded_optimizer_state,
            options=ckpt_opts,
        )
        lr_scheduler.load_state_dict(
            torch.load(
                exp_dir / "lr_scheduler.pt", map_location=device, weights_only=True
            )
        )
        with open(exp_dir / "state.json") as fp:
            state = json.load(fp)
        resumed = True
    if is_experiment:
        LOGGER.info(f"Resumed={resumed} | {state}")
    dist.barrier()

    if is_experiment and (
        (exp_dir.is_mount() and rank == 0)
        or (not exp_dir.is_mount() and local_rank == 0)
    ):
        LOGGER.info(f"Creating experiment root directory")
        exp_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    if is_experiment:
        (exp_dir / f"rank-{rank}").mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Worker saving to {exp_dir / f'rank-{rank}'}")

    timers = {k: LocalTimer(device) for k in ["data", "forward", "backward", "update"]}

    # Gradient accumulation to maintain effective batch size
    gradient_accumulation_steps = args.gradient_accumulation_steps

    for state["epoch"] in range(state["epoch"], args.num_epochs):
        LOGGER.info(f"Begin epoch {state['epoch']} at step {state['epoch_step']}")

        progress_bar = tqdm.tqdm(range(len(dataloader)), disable=rank > 0)
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])

        dataloader.sampler.set_epoch(state["epoch"])
        batches = iter(dataloader)

        for i_step in range(len(dataloader)):
            with timers["data"], torch.no_grad():
                batch = next(batches)
                batch = {k: v.to(device=device) for k, v in batch.items()}

            if i_step < state["epoch_step"]:
                # NOTE: for resuming
                continue

            with timers["forward"]:
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                del batch

            with timers["backward"]:
                loss.backward()

            # Only update weights every N steps
            with timers["update"]:
                if (i_step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=not args.cpu_offload)

            state["global_step"] += 1
            state["epoch_step"] += 1
            state["running_loss"] += outputs.loss.item()  # Track unscaled loss
            progress_bar.update(1)

            if state["global_step"] % args.log_freq == 0:
                tok_per_step = world_size * args.batch_size * args.seq_length
                ms_per_step = sum(t.avg_elapsed_ms() for t in timers.values())
                mem_stats = get_mem_stats(device)
                avg_loss = state["running_loss"] / args.log_freq
                current_lr = lr_scheduler.get_last_lr()[0]
                tokens_per_s = 1000 * tok_per_step / ms_per_step

                info = {
                    "global_step": state["global_step"],
                    "lr": current_lr,
                    "running_loss": avg_loss,
                    "epoch": state["epoch"],
                    "epoch_progress": state["epoch_step"] / len(dataloader),
                    "num_batches_remaining": len(dataloader) - i_step,
                    **mem_stats,
                    "tokens_per_s": tokens_per_s,
                    "time/total": ms_per_step,
                    **{
                        f"time/{k}": timer.avg_elapsed_ms()
                        for k, timer in timers.items()
                    },
                }

                # LOGGER.info(info)

                # Log metrics to Comet ML (only rank 0)
                if experiment:
                    experiment.log_metrics(
                        {
                            "loss": avg_loss,
                            "learning_rate": current_lr,
                            "tokens_per_second": tokens_per_s,
                            "epoch": state["epoch"],
                            "epoch_progress": state["epoch_step"] / len(dataloader),
                        },
                        step=state["global_step"],
                    )

                    # Log GPU memory metrics
                    experiment.log_metrics(
                        {
                            "memory/current_allocated_gb": mem_stats["curr_alloc_gb"],
                            "memory/peak_allocated_gb": mem_stats["peak_alloc_gb"],
                            "memory/current_reserved_gb": mem_stats["curr_resv_gb"],
                            "memory/peak_reserved_gb": mem_stats["peak_resv_gb"],
                        },
                        step=state["global_step"],
                    )

                    # Log timing metrics
                    experiment.log_metrics(
                        {
                            "time/total_ms": ms_per_step,
                            "time/data_ms": timers["data"].avg_elapsed_ms(),
                            "time/forward_ms": timers["forward"].avg_elapsed_ms(),
                            "time/backward_ms": timers["backward"].avg_elapsed_ms(),
                            "time/update_ms": timers["update"].avg_elapsed_ms(),
                        },
                        step=state["global_step"],
                    )

                torch.cuda.reset_peak_memory_stats(device)
                state["running_loss"] = 0
                for t in timers.values():
                    t.reset()

            if is_experiment and state["global_step"] % args.ckpt_freq == 0:
                dist.barrier()
                # NOTE: we have to call this on ALL ranks
                sharded_model_state, sharded_optimizer_state = get_state_dict(
                    model, optimizer, options=ckpt_opts
                )
                save(
                    dict(model=sharded_model_state, optimizer=sharded_optimizer_state),
                    checkpoint_id=exp_dir / "checkpoint",
                )
                if rank == 0:
                    torch.save(lr_scheduler.state_dict(), exp_dir / "lr_scheduler.pt")
                    with open(exp_dir / "state.json", "w") as fp:
                        json.dump(state, fp)

                    # Log checkpoint to Comet ML
                    if experiment:
                        experiment.log_model(
                            name=f"checkpoint-step-{state['global_step']}",
                            file_or_folder=str(exp_dir),
                            metadata={
                                "global_step": state["global_step"],
                                "epoch": state["epoch"],
                                "loss": avg_loss if "avg_loss" in locals() else None,
                            },
                        )
                dist.barrier()

        # Log epoch completion
        if experiment:
            experiment.log_metric(
                "epoch_completed", state["epoch"], step=state["global_step"]
            )

        state["epoch_step"] = 0

    # Training completed - end the experiment
    if experiment:
        LOGGER.info("Training completed. Ending Comet ML experiment.")
        experiment.end()


def _load_and_preprocess_data(args, config):
    """
    Function created using code found in
    https://github.com/huggingface/transformers/blob/v4.45.1/examples/pytorch/language-modeling/run_clm_no_trainer.py
    """
    from itertools import chain

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    data = datasets.load_dataset(args.dataset_name, args.dataset_subset)

    column_names = data["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = data.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    seq_length = args.seq_length or tokenizer.model_max_length
    if seq_length > config.max_position_embeddings:
        seq_length = min(1024, config.max_position_embeddings)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        if total_length > seq_length:
            total_length = (total_length // seq_length) * seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {seq_length}",
    )

    return lm_datasets["train"]


def get_mem_stats(device=None):
    mem = torch.cuda.memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "total_gb": 1e-9 * props.total_memory,
        "curr_alloc_gb": 1e-9 * mem["allocated_bytes.all.current"],
        "peak_alloc_gb": 1e-9 * mem["allocated_bytes.all.peak"],
        "curr_resv_gb": 1e-9 * mem["reserved_bytes.all.current"],
        "peak_resv_gb": 1e-9 * mem["reserved_bytes.all.peak"],
    }


@contextmanager
def rank0_first():
    rank = dist.get_rank()
    if rank == 0:
        yield
    dist.barrier()
    if rank > 0:
        yield
    dist.barrier()


class LocalTimer:
    def __init__(self, device: torch.device):
        if device.type == "cpu":
            self.synchronize = lambda: torch.cpu.synchronize(device=device)
        elif device.type == "cuda":
            self.synchronize = lambda: torch.cuda.synchronize(device=device)
        self.measurements = []
        self.start_time = None

    def __enter__(self):
        self.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if traceback is None:
            self.synchronize()
            end_time = time.time()
            self.measurements.append(end_time - self.start_time)
        self.start_time = None

    def avg_elapsed_ms(self):
        return 1000 * (sum(self.measurements) / len(self.measurements))

    def reset(self):
        self.measurements = []
        self.start_time = None


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment-name", default="fsdp")
    parser.add_argument("-d", "--dataset-name", default="tatsu-lab/alpaca")
    parser.add_argument("-m", "--model-name", default="meta-llama/llama-3.2-1B")
    parser.add_argument("--dataset-subset", default=None)
    parser.add_argument("--save-dir", default="../outputs")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num-epochs", default=100, type=int)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=4, type=int)
    parser.add_argument("--log-freq", default=1, type=int)
    parser.add_argument("--ckpt-freq", default=500, type=int)
    parser.add_argument("-s", "--seq-length", default=1024, type=int)
    parser.add_argument("--cpu-offload", default=False, action="store_true")

    # Comet ML arguments
    parser.add_argument(
        "--comet-project-name",
        default="distributed-training-demo",
        help="Comet ML project name",
    )
    parser.add_argument(
        "--comet-workspace", default="intro-ai", help="Comet ML workspace name"
    )

    return parser


if __name__ == "__main__":
    main()
