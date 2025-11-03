<div align="center">

# Distributed Training Demo

[![Python](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.57%2B-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/transformers/)
[![Datasets](https://img.shields.io/badge/HF%20Datasets-4.3%2B-ff6f00?logo=huggingface&logoColor=white)](https://huggingface.co/docs/datasets)
[![Comet ML](https://img.shields.io/badge/Comet%20ML-enabled-2C3E50?logo=comet&logoColor=FBB040)](https://www.comet.com/)
[![CUDA](https://img.shields.io/badge/CUDA-Ampere%20(8.0)-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Single GPU](https://img.shields.io/badge/Single%20GPU-blue)](#single-gpu-)
[![DDP](https://img.shields.io/badge/DDP-green)](#ddp-)
[![FSDP](https://img.shields.io/badge/FSDP-purple)](#fsdp-)

Train LLMs with PyTorch across single GPU, DDP, and FSDP—with rich, zero-config Comet ML logging for metrics, memory, and timing.

</div>

## Table of contents

- Overview
- Features
- Requirements
- Installation
- Quickstart
- Usage
  - Single GPU
  - DDP (DistributedDataParallel)
  - FSDP (Fully Sharded Data Parallel)
  - Comet ML configuration
- What gets logged
- Repository layout
- Tips and troubleshooting
- Acknowledgments

## Overview

This repository demonstrates modern training patterns for decoder-only LLMs using PyTorch, Hugging Face Transformers/Datasets, and optional Comet ML experiment tracking. It includes progressively more scalable scripts: single GPU, DDP, and FSDP with mixed precision and optional CPU offload. All scripts log useful runtime stats so you can compare throughput, memory, and timing across setups.

## Features

- Single GPU, DDP, and FSDP training modes
- BF16/FP16/FP32 precision options (single GPU supports all; DDP/FSDP use BF16 by default)
- FlashAttention 2 path for fp16/bf16 where available
- Resume/checkpointing for all modes (sharded checkpoints for FSDP)
- Automatic logging to Comet ML (loss, LR, tokens/s, memory, timing breakdown, parameters)
- Reproducible tokenization and grouping pipeline (Hugging Face Datasets)

## Requirements

- OS: Linux (tested on Ubuntu 24.04)
- Python: 3.13+
- GPU: NVIDIA Ampere or newer recommended for FlashAttention 2 (compute capability 8.0)
- CUDA toolchain compatible with your PyTorch build

Python dependencies are pinned in `pyproject.toml`:

- torch>=2.9.0
- transformers>=4.57.1
- datasets>=4.3.0
- comet-ml>=3.54.0
- tqdm>=4.67.1
- flash-attn (built against your local torch/CUDA)

Note on FlashAttention: this project configures build vars to target compute capability 8.0 by default. If you don’t have a compatible GPU, you can still run with SDPA attention paths (e.g., use FP32 or adjust dtype), or remove `flash-attn` from your environment.

## Installation

```bash
pip install -e .
```

Optional: set your Comet ML API key once to enable logging:

```bash
export COMET_API_KEY="your-api-key-here"
```

You can find your API key at: https://www.comet.com/account-settings/api-keys

## Quickstart

The repository contains three training entry points:

- `01-single-gpu/train-llm.py` — simple single-GPU trainer
- `02-ddp/train-llm.py` — multi-GPU on a node via DDP
- `03-fsdp/train-llm.py` — FSDP for large models with sharded state and optional CPU offload

Default model: `meta-llama/llama-3.2-1B` • Default dataset: `tatsu-lab/alpaca`

## Usage

### Single GPU 🚀

From the project root:

```bash
cd 01-single-gpu
python train-llm.py \
  --experiment-name my-first-exp \
  --model-name meta-llama/llama-3.2-1B \
  --dataset-name tatsu-lab/alpaca \
  --batch-size 8 \
  --num-epochs 3 \
  --lr 3e-5 \
  --device cuda \
  --dtype bf16
```

Run without Comet ML by omitting `--experiment-name`:

```bash
python train-llm.py --model-name meta-llama/llama-3.2-1B --batch-size 8 --device cuda --dtype bf16
```

Device and dtype options:

```bash
# CPU with FP32
python train-llm.py --device cpu --dtype fp32

# GPU with BF16 (default)
python train-llm.py --device cuda --dtype bf16

# GPU with FP16
python train-llm.py --device cuda --dtype fp16
```

### DDP 🚂

Use `torchrun` to launch across multiple GPUs on a single node:

```bash
cd 02-ddp
torchrun --standalone --nproc_per_node=4 train-llm.py \
  --experiment-name ddp-4gpu \
  --model-name meta-llama/llama-3.2-1B \
  --dataset-name tatsu-lab/alpaca \
  --batch-size 8 \
  --num-epochs 3 \
  --lr 3e-5
```

Notes:
- DDP script uses BF16 and FlashAttention 2 by default.
- Comet logging happens on rank 0; metrics are aggregated there.
- Checkpoints are saved by rank 0.

Multi-node (example):

```bash
MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 train-llm.py ...
```

Make sure your dataset/model caches and output paths are on shared storage or pre-synced.

### FSDP 🔊

Launch with `torchrun` the same way, optionally enabling CPU offload:

```bash
cd 03-fsdp
torchrun --standalone --nproc_per_node=4 train-llm.py \
  --experiment-name fsdp-4gpu \
  --model-name meta-llama/llama-3.2-1B \
  --dataset-name tatsu-lab/alpaca \
  --batch-size 8 \
  --num-epochs 3 \
  --lr 3e-5 \
  --cpu-offload
```

Notes:
- Model is constructed on the meta device, fully sharded, then materialized with `to_empty`.
- Checkpoints are saved in sharded format using `torch.distributed.checkpoint`.
- Use `--cpu-offload` when GPU memory is tight; this may reduce throughput.

### Comet ML configuration

The scripts accept:

- `--experiment-name` (omit to disable logging)
- `--comet-project-name` (default: `distributed-training-demo`)
- `--comet-workspace` (default: `intro-ai`)

Set your API key once:

```bash
export COMET_API_KEY="your-api-key-here"
```

Open the printed experiment URL to see live metrics and assets.

## What gets logged

Hyperparameters
- Model and dataset names, sequence length, batch size, LR, epochs, optimizer, scheduler, dtype
- GPU name/memory, PyTorch/CUDA versions, world size (DDP/FSDP)

Training metrics (every `--log-freq` steps)
- Loss, learning rate, tokens/second (global tokens/s for distributed)
- Memory: current/peak allocated and reserved (per rank), total GPU memory
- Timing breakdown: data, forward, backward, optimizer/update, and total step time

Model information
- Total vs trainable parameters
- Text serialization of the model architecture

Checkpoints
- Single GPU/DDP: state dicts for model (and optimizer in single GPU), LR scheduler, and `state.json`
- FSDP: sharded model/optimizer checkpoints via `torch.distributed.checkpoint`, LR scheduler, and `state.json`

## Repository layout

```
.
├── 01-single-gpu/
│   └── train-llm.py
├── 02-ddp/
│   └── train-llm.py
├── 03-fsdp/
│   └── train-llm.py
├── outputs/
│   └── single-gpu/        # example output location
├── main.py                # (optional) project entry point stub
├── pyproject.toml         # dependencies (torch, transformers, datasets, comet-ml, flash-attn)
└── README.md
```

Default save root is `--save-dir ../outputs` (relative to each script directory). If you set `--experiment-name`, artifacts are created under `../outputs/<experiment-name>/`.

## Tips and troubleshooting 🧰

- FlashAttention build issues
  - Target arch is configured for compute capability 8.0 (A100). On other GPUs, either rebuild with the correct arch or remove `flash-attn` from your environment to fall back to SDPA.
  - Ensure your CUDA toolkit and PyTorch wheels are compatible.
- OOM (out of memory)
  - Lower `--batch-size` and/or `--seq-length`.
  - Use BF16/FP16 when possible; enable `--cpu-offload` in FSDP.
- Slow throughput
  - Increase `--batch-size` if memory allows; ensure pinned/shared caches for datasets/models on multi-node.
  - Check that `torch.compile` didn’t deopt; try running a few warmup steps.
- Resuming
  - If an `--experiment-name` directory exists with state files, training will attempt to resume.
- Multi-node
  - Use shared storage for HF caches and outputs, or pre-sync them. Set `MASTER_ADDR/MASTER_PORT` and the appropriate `torchrun` flags.

## Acknowledgments

This project is **adapted from** the excellent [Lambda Labs Distributed Training Guide](https://github.com/LambdaLabsML/distributed-training-guide) and modified for educational purposes. Most of the code structure and patterns come from their original work. 🙏

Built with:
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/) and [Datasets](https://huggingface.co/docs/datasets)
- [Comet ML](https://www.comet.com/)

---

If you find this useful, a star is always appreciated. Happy training! 🌟
