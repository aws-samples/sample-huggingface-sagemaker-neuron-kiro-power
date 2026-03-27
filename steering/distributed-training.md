# Distributed Training on Trainium

## When to Activate

Activate when the user is training large models across multiple NeuronCores or multiple Trainium instances.

## Parallelism Quick Reference

| Strategy | Use When | NeuronCores |
|---|---|---|
| Data Parallel (DDP) | Model fits on 1 core | 2+ |
| FSDP | Model too large for 1 core | 8+ |
| Tensor Parallelism | Very large layers | 8/16/32 |
| Pipeline Parallelism | Very deep models | 16+ |

## NeuronCore Counts

| Instance | NeuronCores | Typical Use |
|---|---|---|
| trn1.2xlarge | 2 | Small models, DP |
| trn1.32xlarge | 32 | Large models, TP/FSDP |
| trn1n.32xlarge | 32 | Multi-node (high bandwidth) |
| trn2.48xlarge | 64 | Very large models (70B+) |

## Common Patterns

### Data Parallel
```bash
torchrun --nproc_per_node=2 train.py  # trn1.2xlarge
torchrun --nproc_per_node=32 train.py  # trn1.32xlarge
```

### Multi-Node
```bash
torchrun --nnodes=2 --nproc_per_node=32 --master_addr=$MASTER_ADDR train.py
```

Use SageMaker HyperPod for managed multi-node orchestration.

## Tips

- `nproc_per_node` = number of NeuronCores to use
- Use `neuron_parallel_compile` before long training runs
- Set `NEURON_RT_STOCHASTIC_ROUNDING_EN=1` for BF16 training
- Pad inputs to fixed lengths to avoid recompilation
