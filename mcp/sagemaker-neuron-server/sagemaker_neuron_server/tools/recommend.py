"""Recommend instance types for Hugging Face models on Neuron."""

import json
import os
from mcp.server.fastmcp import FastMCP
from ..utils.aws import INSTANCE_RECOMMENDATIONS, NEURON_DLCS, get_neuron_compile_params


def _fetch_model_params(model_id: str) -> float | None:
    """Fetch parameter count from HF Hub. Returns billions or None."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN", None))
        info = api.model_info(model_id, files_metadata=True)

        # Method 1: safetensors metadata (if available)
        if info.safetensors and "total" in getattr(info.safetensors, "parameters", {}):
            return info.safetensors["parameters"]["total"] / 1e9

        # Method 2: estimate from safetensor file sizes (BF16 = 2 bytes/param)
        if info.siblings:
            total_bytes = sum(
                s.size for s in info.siblings
                if s.rfilename.endswith(".safetensors") and s.size
            )
            if total_bytes > 0:
                return total_bytes / 2 / 1e9

        # Method 3: estimate from .bin file sizes
        if info.siblings:
            total_bytes = sum(
                s.size for s in info.siblings
                if s.rfilename.endswith(".bin") and s.size
            )
            if total_bytes > 0:
                return total_bytes / 2 / 1e9

    except Exception:  # intentional — HF Hub may be unreachable, fall back gracefully
        pass

    return None


def _estimate_category(params_b: float) -> str:
    if params_b < 3:
        return "small"
    elif params_b < 13:
        return "medium"
    elif params_b <= 75:
        return "large"
    return "xlarge"


def register_recommend_tools(mcp: FastMCP):

    @mcp.tool()
    def recommend_instance(
        model_id: str,
        use_case: str = "inference",
        params_billions: float = 0,
    ) -> str:
        """Recommend a Neuron instance type for a Hugging Face model.

        Args:
            model_id: HF model ID (e.g., 'meta-llama/Llama-3.3-70B-Instruct')
            use_case: 'inference' or 'training'
            params_billions: Override model size in billions (auto-fetched from HF Hub if not provided)
        """
        params_b = None
        source = None

        # 1. Use explicit override if provided
        if params_billions > 0:
            params_b = params_billions
            source = "user_provided"

        # 2. Try fetching from HF Hub
        if params_b is None:
            params_b = _fetch_model_params(model_id)
            if params_b is not None:
                source = "huggingface_hub"

        # 3. If all else fails, return error
        if params_b is None:
            return json.dumps({
                "error": f"Could not determine size of '{model_id}'. Model not found on HF Hub or missing file metadata. Please provide params_billions.",
            })

        category = _estimate_category(params_b)
        rec = INSTANCE_RECOMMENDATIONS[category]
        instance = rec[use_case]
        serving_type = "vllm" if use_case == "inference" and params_b >= 7 else use_case
        container = NEURON_DLCS.get(serving_type, NEURON_DLCS.get(use_case, {}))
        compile_params = get_neuron_compile_params(params_b, instance)

        return json.dumps({
            "model_id": model_id,
            "params_billions": round(params_b, 2),
            "params_source": source,
            "category": category,
            "use_case": use_case,
            "recommended_instance": instance,
            "serving_type": serving_type,
            "neuron_compile_params": compile_params,
            "available_instances": container.get("instances", []),
            "notes": _get_notes(category, use_case),
            "disclaimer": "Instance recommendations are based on model size and Neuron hardware specifications. Verify model architecture compatibility with Neuron SDK at awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/models/index.html before deploying to production.",
        })


def _get_notes(category: str, use_case: str) -> str:
    notes = {
        ("small", "training"): "Single NeuronCore sufficient. Use --nproc_per_node=1.",
        ("small", "inference"): "Single NeuronCore. Low latency, cost-effective.",
        ("medium", "training"): "Use data parallelism with 2-8 NeuronCores.",
        ("medium", "inference"): "2-4 NeuronCores with tensor parallelism.",
        ("large", "training"): "Use FSDP or TP across 32 NeuronCores on trn1.32xlarge.",
        ("large", "inference"): "Tensor parallelism across 24+ NeuronCores. Consider vLLM for serving.",
        ("xlarge", "training"): "Multi-node training on Trn2. Use NxD Training with TP+PP.",
        ("xlarge", "inference"): "Trn2 with full tensor parallelism. vLLM recommended.",
    }
    return notes.get((category, use_case), "")
