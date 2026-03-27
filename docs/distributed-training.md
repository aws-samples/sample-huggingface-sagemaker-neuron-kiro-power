# Distributed Training on Trainium

## Overview

Train large Hugging Face models across multiple NeuronCores on AWS Trainium using standard PyTorch distributed APIs.

## Parallelism Strategies

| Strategy | What It Shards | When to Use |
|---|---|---|
| Data Parallel (DDP) | Data across workers | Model fits on one NeuronCore |
| FSDP | Model params + gradients + optimizer | Model too large for one NeuronCore |
| Tensor Parallelism (TP) | Individual layers across cores | Very large layers (e.g., attention) |
| Pipeline Parallelism (PP) | Model layers across stages | Very deep models |

## Data Parallel Training

Simplest approach — replicate the model on each NeuronCore, split data across them.

```bash
# 2 workers on trn1.2xlarge (2 NeuronCores)
torchrun --nproc_per_node=2 train.py

# 32 workers on trn1.32xlarge (32 NeuronCores)
torchrun --nproc_per_node=32 train.py
```

### With HF Trainer (Current XLA Path)

```bash
export NEURON_CC_FLAGS="--model-type=transformer"

NEURON_RT_STOCHASTIC_ROUNDING_EN=1 torchrun --nproc_per_node=2 run_glue.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --do_train \
  --bf16 \
  --use_cpu True \
  --per_device_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3
```

### With TorchNeuron Native (Coming)

```python
# Standard PyTorch DDP — no special imports
import torch.distributed as dist

dist.init_process_group(backend="nccl")  # TorchNeuron handles this
model = model.to('neuron')
model = torch.nn.parallel.DistributedDataParallel(model)
```

## FSDP (Fully Sharded Data Parallel)

For models that don't fit on a single NeuronCore.

### TorchNeuron Native (Coming)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = model.to('neuron')
model = FSDP(model)
```

TorchNeuron supports FSDPv1, FSDPv2, and SimpleFSDP. For `torch.compile`, SimpleFSDP is recommended.

## Tensor Parallelism

Split individual layers across NeuronCores. Use NxD Core or NxD Training libraries.

### With NxD Training

```bash
# Llama 3.1 8B with TP + ZeRO-1
torchrun --nproc_per_node=32 train_llama.py \
  --tensor_parallel_size 8 \
  --zero_1 True
```

## NeuronCore Counts by Instance

| Instance | NeuronCores | Typical Parallelism |
|---|---|---|
| trn1.2xlarge | 2 | DP (2 workers) |
| trn1.32xlarge | 32 | DP (32), TP (8/16/32), FSDP |
| trn1n.32xlarge | 32 | Multi-node DP/TP (high bandwidth) |
| trn2.48xlarge | 64 | TP (32/64), FSDP, PP |

## Multi-Node Training

For models requiring more than one instance:

```bash
# On trn1n.32xlarge (optimized for multi-node)
torchrun \
  --nnodes=2 \
  --nproc_per_node=32 \
  --master_addr=$MASTER_ADDR \
  --master_port=41000 \
  train.py
```

Use SageMaker HyperPod or ParallelCluster for managed multi-node orchestration.

## Tips

- Always use `torchrun` even for single worker (required with transformers >= 4.44)
- Use `neuron_parallel_compile` to precompile graphs before long training runs
- Set `NEURON_RT_STOCHASTIC_ROUNDING_EN=1` for better convergence with BF16
- Pad inputs to fixed lengths to avoid recompilation from variable shapes
- Use `--save_total_limit` to control checkpoint storage
