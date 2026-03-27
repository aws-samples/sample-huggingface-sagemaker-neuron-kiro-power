# Example: Deploy Llama 3.3 70B on SageMaker AI with Neuron

Deploy Meta's Llama 3.3 70B Instruct model on a SageMaker AI endpoint using vLLM on Inferentia 2.

## Prerequisites

- AWS account with SageMaker AI access
- IAM role with SageMaker AI, S3, and ECR permissions
- Service quota for `ml.inf2.48xlarge` (request via AWS console if needed)
- Hugging Face token with access to Llama 3.3 (accept license on HF Hub)

## Deploy

```python
import sagemaker
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

role = sagemaker.get_execution_role()
session = sagemaker.Session()

# Get the Neuron vLLM container
image_uri = get_huggingface_llm_image_uri("huggingface-neuronx", version="0.11.0")

model = HuggingFaceModel(
    role=role,
    image_uri=image_uri,
    env={
        "HF_MODEL_ID": "meta-llama/Llama-3.3-70B-Instruct",
        "HF_TOKEN": "<your-hf-token>",
        "TENSOR_PARALLEL_SIZE": "24",
        "MAX_MODEL_LEN": "4096",
        "MAX_BATCH_SIZE": "8",
    },
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.inf2.48xlarge",
    endpoint_name="llama-33-70b-neuron",
    container_startup_health_check_timeout=1800,  # Model compilation takes time
)
```

## Test

```python
response = predictor.predict({
    "inputs": "Explain quantum computing in simple terms.",
    "parameters": {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
    },
})

print(response[0]["generated_text"])
```

## Cleanup

```python
predictor.delete_endpoint()
```

## Cost Estimate

- `ml.inf2.48xlarge`: Check [SageMaker AI pricing](https://aws.amazon.com/sagemaker/pricing/) for current rates
- First deployment takes ~15-30 min (model download + Neuron compilation)
- Subsequent deployments are faster if using cached compilations

## Notes

- 24 NeuronCores provide enough memory for the 70B model with tensor parallelism
- vLLM handles continuous batching and KV cache management automatically
- For lower latency on shorter contexts, reduce `MAX_MODEL_LEN`
- For higher throughput, increase `MAX_BATCH_SIZE`
