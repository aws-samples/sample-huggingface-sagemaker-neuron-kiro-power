# Model Optimization on Neuron

## Overview

Optimize Hugging Face models for best performance and cost on AWS Trainium and Inferentia hardware.

## Compilation

### Current Path (XLA)

Models must be compiled to Neuron-optimized graphs before execution.

```bash
# Precompile graphs in parallel (one-time per config)
neuron_parallel_compile ./train.sh

# Compiled graphs are cached in Neuron persistent cache
# Subsequent runs skip compilation
./train.sh
```

### TorchNeuron Native (Coming)

```python
# JIT compilation via torch.compile
model = torch.compile(model, backend="neuron")

# Or eager mode — no compilation needed (slower but debuggable)
model = model.to('neuron')
```

## Mixed Precision (BF16)

BF16 is the recommended precision for Neuron hardware.

### Current Path

```bash
# Via HF Trainer
torchrun train.py --bf16 --use_cpu True

# Compiler flag for transformer models
export NEURON_CC_FLAGS="--model-type=transformer"
```

### TorchNeuron Native

```python
with torch.autocast(device_type="neuron"):
    outputs = model(**batch)
```

## Neuron Persistent Cache

Compiled graphs are cached to avoid recompilation:

```bash
# Cache location (default: ~/.cache/neuron)
export NEURON_COMPILE_CACHE_URL="s3://my-bucket/neuron-cache"

# Share cache across team members via S3
```

## LLM Serving with vLLM

For production LLM inference, use vLLM on Neuron:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=24,
    max_model_len=4096,
    device="neuron",
)

outputs = llm.generate(["What is AI?"], SamplingParams(temperature=0.7))
```

## NKI (Neuron Kernel Interface)

For custom performance-critical operations:

```python
from torch_neuronx import nki
import nki.language as nl

@nki.jit
def fused_rmsnorm(input_ptr, weight_ptr, output_ptr):
    """Custom fused RMSNorm kernel for Trainium."""
    data = nl.load(input_ptr[0:128])
    w = nl.load(weight_ptr[0:128])
    
    # RMS normalization
    sq = data * data
    mean_sq = nl.mean(sq, axis=-1)
    rms = nl.rsqrt(mean_sq + 1e-6)
    normalized = data * rms
    
    result = normalized * w
    nl.store(output_ptr[0:128], value=result)
```

NKI kernels work in both eager mode and `torch.compile`.

## Performance Tips

| Tip | Impact | Applies To |
|---|---|---|
| Use BF16 mixed precision | 2x throughput | Training + Inference |
| Precompile with `neuron_parallel_compile` | Faster first run | Training (XLA path) |
| Pad inputs to fixed lengths | Avoid recompilation | Training (XLA path) |
| Use vLLM for LLM serving | Optimized batching + KV cache | Inference |
| Set `NEURON_RT_STOCHASTIC_ROUNDING_EN=1` | Better convergence | Training |
| Use Neuron persistent cache on S3 | Share compilations | Team workflows |
| Match precompile and run epochs | Avoid extra compilations | Training (XLA path) |

## Model Compatibility

Most Hugging Face Transformers models work on Neuron. Validated models include:

| Category | Models |
|---|---|
| LLMs | Llama 3.x, Qwen 3.x, Mistral, GPT-NeoX |
| Vision-Language | Qwen2-VL, Qwen3-VL, Pixtral |
| MoE | Qwen3 235B MoE |
| Image Gen | Flux |
| Encoders | BERT, RoBERTa, DistilBERT |
| Seq2Seq | T5, BART |

For the full list, check the [Neuron model samples](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/models/index.html).
