# TorchNeuron — Native PyTorch for AWS Trainium & Inferentia

## Overview

TorchNeuron is a native PyTorch backend for AWS Trainium and Inferentia hardware. It provides eager mode execution, `torch.compile` support, and standard PyTorch distributed APIs — requiring minimal code changes from existing PyTorch/CUDA code.

> **Status**: TorchNeuron native backend is in closed Beta (announced at re:Invent '25). The current production path uses `torch-neuronx` with XLA-based compilation. This guide covers both.

## Current Path (torch-neuronx with XLA)

This is what works today on SageMaker AI and EC2.

### Device Placement

```python
# Models run on Neuron via XLA
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()
model = model.to(device)
```

### Training with HF Trainer

```bash
# Environment setup
export NEURON_CC_FLAGS="--model-type=transformer"

# Launch with torchrun (required even for single worker)
NEURON_RT_STOCHASTIC_ROUNDING_EN=1 torchrun --nproc_per_node=2 run_glue.py \
  --model_name_or_path bert-large-uncased \
  --task_name mrpc \
  --do_train --do_eval \
  --bf16 \
  --use_cpu True \
  --max_seq_length 128 \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 5
```

### Precompilation (Optional but Recommended)

```bash
# Extract and compile XLA graphs in parallel (one-time)
neuron_parallel_compile ./run.sh

# Subsequent runs load from cache — much faster
./run.sh
```

## TorchNeuron Native Backend (Coming)

When TorchNeuron reaches GA, the workflow simplifies significantly.

### Device Placement

```python
# Just change device — no XLA imports needed
model = model.to('neuron')
```

### torch.compile

```python
@torch.compile(backend="neuron")
def train_step(model, batch):
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    return loss
```

### Eager Mode

Operations execute immediately — no graph extraction or precompilation needed. Supports debugging, tensor inspection, and print statements.

### Mixed Precision

```python
with torch.autocast(device_type="neuron"):
    outputs = model(**batch)
```

### Distributed Training

Standard PyTorch distributed APIs work natively:
- FSDP (v1, v2, SimpleFSDP)
- DDP (Distributed Data Parallel)
- DTensor (Distributed Tensor)
- Tensor Parallelism

### Migration from CUDA

| CUDA | TorchNeuron |
|---|---|
| `.to('cuda')` | `.to('neuron')` |
| `torch.compile(backend="inductor")` | `torch.compile(backend="neuron")` |
| `torch.autocast(device_type="cuda")` | `torch.autocast(device_type="neuron")` |
| `NGPU=8` | `NGPU=32` (NeuronCores, not chips) |

### NKI Custom Kernels

For performance-critical operations:

```python
from torch_neuronx import nki

@nki.jit
def custom_kernel(in_ptr, out_ptr):
    import nki.language as nl
    data = nl.load(in_ptr[0:128])
    result = nl.sin(data)
    nl.store(out_ptr[0:128], value=result)
```

## Key Differences: XLA Path vs TorchNeuron Native

| Feature | XLA Path (current) | TorchNeuron Native (coming) |
|---|---|---|
| Execution | Graph-based (trace + compile) | Eager + torch.compile |
| Precompilation | Required for performance | Not needed |
| Debugging | Limited (graph mode) | Full eager debugging |
| Distributed | XLA-specific APIs | Standard PyTorch distributed |
| Code changes | Moderate (XLA imports, workarounds) | Minimal (device swap only) |
| HF integration | Via optimum-neuron wrappers | Direct Transformers APIs |

## Supported Hardware

| Instance | Chip | NeuronCores | Use Case |
|---|---|---|---|
| trn1.2xlarge | Trainium 1 | 2 | Small model training/inference |
| trn1.32xlarge | Trainium 1 | 32 | Large model training |
| trn1n.32xlarge | Trainium 1 | 32 | Multi-node training (high bandwidth) |
| trn2.48xlarge | Trainium 2 | 64 | Very large models (70B+) |
| inf2.xlarge | Inferentia 2 | 2 | Small model inference |
| inf2.8xlarge | Inferentia 2 | 8 | Medium model inference |
| inf2.24xlarge | Inferentia 2 | 24 | Large model inference |
| inf2.48xlarge | Inferentia 2 | 48 | Very large model inference |

## Known Issues (Current XLA Path)

- Use `torchrun` even for single worker (transformers >= 4.44)
- Use `--use_cpu True` with `--bf16` to avoid "bf16/gpu not supported" error
- Use `transformers <= 4.53.3` or pass `--optim adamw_torch` to avoid fused AdamW error
- Pad to max length for variable-length inputs to avoid recompilation
- Set `eval_do_concat_batches=False` to avoid per-step compilation during eval
