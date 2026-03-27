# SageMaker AI Deployment — Hugging Face on Neuron

## When to Activate

Activate when the user is deploying or managing SageMaker AI endpoints or training jobs with Hugging Face models on Neuron hardware.

## Neuron DLC Container Images

### Training (Trainium)
```
763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training-neuronx:2.8.0-transformers4.55.4-neuronx-py310-sdk2.26.0-ubuntu22.04
```
Supported instances: ml.trn1.2xlarge, ml.trn1.32xlarge, ml.trn1n.32xlarge, ml.trn2.48xlarge

### Inference (General)
```
763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-inference-neuronx:2.8.0-transformers4.55.4-neuronx-py310-sdk2.26.0-ubuntu22.04
```
Supported instances: ml.inf2.xlarge, ml.inf2.8xlarge, ml.inf2.24xlarge, ml.inf2.48xlarge

### LLM Serving (vLLM)
```
763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-vllm-inference-neuronx:0.11.0-optimum0.4.4-neuronx-py310-sdk2.26.1-ubuntu22.04
```
Supported instances: ml.inf2.24xlarge, ml.inf2.48xlarge, ml.trn1.32xlarge, ml.trn2.48xlarge

## Deployment Patterns

### Quick Deploy (SageMaker AI SDK)
```python
from sagemaker.huggingface import HuggingFaceModel
model = HuggingFaceModel(role=role, image_uri=neuron_image, env={"HF_MODEL_ID": model_id})
predictor = model.deploy(instance_type="ml.inf2.xlarge", initial_instance_count=1)
```

### LLM Deploy (vLLM)
```python
env = {"HF_MODEL_ID": model_id, "TENSOR_PARALLEL_SIZE": "24", "MAX_MODEL_LEN": "4096"}
model = HuggingFaceModel(role=role, image_uri=vllm_image, env=env)
predictor = model.deploy(instance_type="ml.inf2.48xlarge", initial_instance_count=1,
                         container_startup_health_check_timeout=1800)
```

### Training Job
```python
from sagemaker.huggingface import HuggingFace
estimator = HuggingFace(entry_point="train.py", instance_type="ml.trn1.2xlarge",
                        image_uri=training_image, hyperparameters={...})
estimator.fit({"train": s3_train_path})
```

## Critical Notes

- Use `us-east-1` or `us-west-2` for best Neuron DLC availability
- Replace `{region}` in container URIs with your target region
- Set `container_startup_health_check_timeout=1800` for large LLMs
- SageMaker AI SDK v2 is required (`pip install "sagemaker<3.0.0"`)
- Always clean up endpoints after testing
