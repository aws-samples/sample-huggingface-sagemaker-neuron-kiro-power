# TorchNeuron — Native PyTorch for AWS Trainium & Inferentia

## When to Activate

Activate this steering when the user is:
- Working with Hugging Face models on AWS Trainium or Inferentia
- Deploying models on SageMaker with Neuron containers
- Training or fine-tuning on Trainium instances
- Asking about TorchNeuron, torch-neuronx, or optimum-neuron
- Migrating CUDA/GPU code to Neuron

## Key Concepts

TorchNeuron is a native PyTorch backend for AWS Trainium/Inferentia. Two paths exist:

**Current (production):** `torch-neuronx` with XLA-based compilation
- Requires `neuron_parallel_compile` for graph extraction
- Uses `optimum-neuron` wrappers for HF Transformers integration
- Models run via XLA device: `xm.xla_device()`

**Coming (Beta):** TorchNeuron native backend
- Eager mode: `model.to('neuron')`
- JIT: `torch.compile(backend="neuron")`
- Standard PyTorch distributed (FSDP, DDP, DTensor, TP)
- No optimum-neuron wrappers needed

## Decision Tree

```
User wants to deploy HF model on Neuron?
├── LLM (7B+ params) → Use vLLM container on SageMaker
│   └── deploy_model tool with serving_type="vllm"
├── Smaller model inference → Use inference container
│   └── deploy_model tool with serving_type="inference"
└── Fine-tuning → Use training container on Trainium
    └── create_training_job tool
```

```
Which instance?
├── < 3B params → ml.trn1.2xlarge (train) / ml.inf2.xlarge (infer)
├── 3B - 13B → ml.trn1.32xlarge (train) / ml.inf2.8xlarge (infer)
├── 13B - 70B → ml.trn1.32xlarge (train) / ml.inf2.48xlarge (infer)
└── 70B+ → ml.trn2.48xlarge (both)
```

## Critical Reminders

- Always use `torchrun` even for single worker (HF transformers >= 4.44)
- Always set `--bf16` and `--use_cpu True` for Neuron training
- Set `NEURON_CC_FLAGS="--model-type=transformer"` for transformer models
- Set `NEURON_RT_STOCHASTIC_ROUNDING_EN=1` for training
- Pad inputs to fixed lengths to avoid recompilation
- Always remind users to `delete_endpoint()` after testing to avoid charges
- First deployment of large models takes 15-30 min (compilation)

## Available MCP Tools

- `deploy_model` — Deploy HF model to SageMaker endpoint with Neuron
- `create_training_job` — Launch fine-tuning on Trainium via SageMaker
- `list_endpoints` — List active SageMaker endpoints
- `describe_endpoint` — Get endpoint details and status
- `delete_endpoint` — Delete a SageMaker endpoint
- `recommend_instance` — Suggest instance type for a model

Use the HF Hub MCP server to search for models before deploying.
