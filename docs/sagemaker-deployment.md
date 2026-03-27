# SageMaker AI Deployment — Hugging Face Models on Neuron

## Overview

Deploy Hugging Face models on Amazon SageMaker AI using Neuron-optimized Deep Learning Containers (DLCs). Supports inference on Inferentia 2 and training on Trainium instances.

## Deployment Paths

| Path | Best For | Complexity |
|---|---|---|
| SageMaker AI SDK | Custom deployments, full control | Medium |
| SageMaker JumpStart | Quick deployment of popular models | Low |
| vLLM on Neuron | High-performance LLM serving | Medium |

## Neuron DLC Container Images

### Training
```
763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training-neuronx:2.8.0-transformers4.55.4-neuronx-py310-sdk2.26.0-ubuntu22.04
```

### Inference (General)
```
763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-inference-neuronx:2.8.0-transformers4.55.4-neuronx-py310-sdk2.26.0-ubuntu22.04
```

### LLM Serving (vLLM)
```
763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-vllm-inference-neuronx:0.11.0-optimum0.4.4-neuronx-py310-sdk2.26.1-ubuntu22.04
```

## Deploy with SageMaker AI SDK

### Inference Endpoint

```python
from sagemaker.huggingface import HuggingFaceModel

role = "arn:aws:iam::<account>:role/<sagemaker-role>"

model = HuggingFaceModel(
    model_data=None,  # Load from HF Hub
    role=role,
    transformers_version="4.55",
    pytorch_version="2.8",
    py_version="py310",
    env={
        "HF_MODEL_ID": "meta-llama/Llama-3.1-8B-Instruct",
        "NEURON_RT_NUM_CORES": "2",
    },
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference-neuronx:2.8.0-transformers4.55.4-neuronx-py310-sdk2.26.0-ubuntu22.04",
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.inf2.xlarge",
)

result = predictor.predict({"inputs": "What is deep learning?"})
```

### LLM Serving with vLLM

```python
from sagemaker.huggingface import get_huggingface_llm_image_uri

image_uri = get_huggingface_llm_image_uri("huggingface-neuronx")

model = HuggingFaceModel(
    role=role,
    image_uri=image_uri,
    env={
        "HF_MODEL_ID": "meta-llama/Llama-3.3-70B-Instruct",
        "TENSOR_PARALLEL_SIZE": "24",
        "MAX_MODEL_LEN": "4096",
    },
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.inf2.48xlarge",
)
```

## Train with SageMaker AI SDK

```python
from sagemaker.huggingface import HuggingFace

estimator = HuggingFace(
    entry_point="train.py",
    source_dir="./scripts",
    instance_type="ml.trn1.32xlarge",
    instance_count=1,
    role=role,
    transformers_version="4.55",
    pytorch_version="2.8",
    py_version="py310",
    hyperparameters={
        "model_name_or_path": "Qwen/Qwen3-1.7B",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
        "learning_rate": 2e-5,
        "bf16": True,
    },
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training-neuronx:2.8.0-transformers4.55.4-neuronx-py310-sdk2.26.0-ubuntu22.04",
)

estimator.fit({"train": "s3://bucket/train", "test": "s3://bucket/test"})
```

## Instance Selection Guide

| Model Size | Inference Instance | Training Instance | Notes |
|---|---|---|---|
| < 3B params | ml.inf2.xlarge | ml.trn1.2xlarge | Single NeuronCore sufficient |
| 3B - 13B | ml.inf2.8xlarge | ml.trn1.32xlarge | TP across 2-8 cores |
| 13B - 70B | ml.inf2.48xlarge | ml.trn1.32xlarge | Full TP, consider vLLM |
| 70B+ | ml.trn2.48xlarge | ml.trn2.48xlarge | Trn2 required, multi-node for training |

## SageMaker Integration Points

- **JumpStart**: One-click deployment of popular HF models on Neuron
- **HyperPod**: Managed infrastructure for large-scale distributed training on Trainium
- **Training Jobs**: Managed training with automatic checkpointing
- **Endpoints**: Managed inference with auto-scaling

## Cleanup

Always delete endpoints when done to avoid charges:

```python
predictor.delete_endpoint()
```
