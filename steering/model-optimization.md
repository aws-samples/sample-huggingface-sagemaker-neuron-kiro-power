# Model Optimization on Neuron

## When to Activate

Activate when the user asks about performance, compilation, mixed precision, vLLM serving, or NKI custom kernels on Neuron.

## Quick Optimization Checklist

1. ✅ Use BF16 mixed precision (`--bf16`)
2. ✅ Precompile with `neuron_parallel_compile` (XLA path)
3. ✅ Set `NEURON_CC_FLAGS="--model-type=transformer"`
4. ✅ Set `NEURON_RT_STOCHASTIC_ROUNDING_EN=1` for training
5. ✅ Pad inputs to fixed lengths
6. ✅ Use vLLM for LLM inference serving
7. ✅ Share Neuron cache via S3 for team workflows

## Compilation

### Current (XLA)
```bash
neuron_parallel_compile ./train.sh  # One-time precompilation
./train.sh                          # Fast — loads from cache
```

### TorchNeuron Native (Coming)
```python
model = torch.compile(model, backend="neuron")  # JIT
# or
model = model.to('neuron')  # Eager — no compilation
```

## LLM Serving

Use vLLM on Neuron for production LLM inference:
- Continuous batching
- KV cache management
- Tensor parallelism
- Container: `huggingface-vllm-inference-neuronx`

## Neuron Cache

```bash
# Share compilations across team via S3
export NEURON_COMPILE_CACHE_URL="s3://my-bucket/neuron-cache"
```

## Validated Models

LLMs: Llama 3.x, Qwen 3.x, Mistral | Vision-Language: Qwen2-VL, Qwen3-VL | MoE: Qwen3 235B | Image Gen: Flux | Encoders: BERT, RoBERTa, DistilBERT
