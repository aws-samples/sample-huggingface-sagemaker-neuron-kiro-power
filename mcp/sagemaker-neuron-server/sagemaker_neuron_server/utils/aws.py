"""Shared AWS utilities for SageMaker and Neuron operations."""

import os
import boto3
from botocore.exceptions import ClientError

# Default Neuron DLC container images — fallback only when dynamic ECR lookup fails. Override via env vars.
_DEFAULT_TRAINING_DLC = "763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-training-neuronx:2.8.0-transformers4.55.4-neuronx-py310-sdk2.26.0-ubuntu22.04"
_DEFAULT_INFERENCE_DLC = "763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-inference-neuronx:2.8-neuronx-py310-sdk2.26.0-ubuntu22.04-v1"
_DEFAULT_VLLM_DLC = "763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-vllm-inference-neuronx:0.11.0-optimum0.4.4-neuronx-py310-sdk2.26.1-ubuntu22.04"

NEURON_DLCS = {
    "training": {
        "image": os.environ.get("NEURON_TRAINING_DLC_URI", _DEFAULT_TRAINING_DLC),
        "instances": ["ml.trn1.2xlarge", "ml.trn1.32xlarge", "ml.trn1n.32xlarge", "ml.trn2.48xlarge"],
    },
    "inference": {
        "image": os.environ.get("NEURON_INFERENCE_DLC_URI", _DEFAULT_INFERENCE_DLC),
        "instances": ["ml.inf2.xlarge", "ml.inf2.8xlarge", "ml.inf2.24xlarge", "ml.inf2.48xlarge", "ml.trn1.2xlarge", "ml.trn1.32xlarge"],
    },
    "vllm": {
        "image": os.environ.get("NEURON_VLLM_DLC_URI", _DEFAULT_VLLM_DLC),
        "instances": ["ml.inf2.24xlarge", "ml.inf2.48xlarge", "ml.trn1.32xlarge", "ml.trn2.48xlarge"],
    },
}

# Model size to instance recommendations
INSTANCE_RECOMMENDATIONS = {
    "small": {  # < 3B params
        "training": "ml.trn1.2xlarge",
        "inference": "ml.inf2.xlarge",
    },
    "medium": {  # 3B - 13B params
        "training": "ml.trn1.32xlarge",
        "inference": "ml.inf2.8xlarge",
    },
    "large": {  # 13B - 70B params
        "training": "ml.trn1.32xlarge",
        "inference": "ml.inf2.48xlarge",
    },
    "xlarge": {  # 70B+ params
        "training": "ml.trn2.48xlarge",
        "inference": "ml.inf2.48xlarge",
    },
}

# Neuron cores per instance type
NEURON_CORES = {
    "ml.inf2.xlarge": 2,
    "ml.inf2.8xlarge": 2,
    "ml.inf2.24xlarge": 12,
    "ml.inf2.48xlarge": 24,
    "ml.trn1.2xlarge": 2,
    "ml.trn1.32xlarge": 32,
    "ml.trn1n.32xlarge": 32,
    "ml.trn2.48xlarge": 64,
}

# HBM per Neuron core (GB)
NEURON_HBM_PER_CORE = {
    "inf2": 16,
    "trn1": 16,
    "trn1n": 16,
    "trn2": 32,
}


def get_neuron_compile_params(params_b: float, instance_type: str) -> dict:
    """Auto-derive Neuron compile params from model size and instance type.

    Returns dict with num_cores, sequence_length, batch_size, auto_cast_type.
    Based on: model BF16 weight memory, KV cache budget, and instance HBM.
    """
    chip = instance_type.split(".")[1].replace("ml.", "")  # e.g. "inf2", "trn1"
    for prefix in ("inf2", "trn2", "trn1n", "trn1"):
        if prefix in chip:
            chip = prefix
            break
    hbm_per_core = NEURON_HBM_PER_CORE.get(chip, 16)
    total_cores = NEURON_CORES.get(instance_type, 2)

    # BF16 model weight memory: ~2 bytes per param
    weight_gb = params_b * 2

    # Minimum cores needed to fit weights (with 60% HBM budget for weights, rest for KV cache + activations)
    min_cores = max(1, int(weight_gb / (hbm_per_core * 0.6)) + 1)
    # TP degree must be power of 2 and <= total cores
    tp_degree = 1
    while tp_degree < min_cores and tp_degree < total_cores:
        tp_degree *= 2
    tp_degree = min(tp_degree, total_cores)

    # Remaining HBM per core after weights for KV cache
    weight_per_core = weight_gb / tp_degree
    free_hbm = hbm_per_core - weight_per_core

    # Sequence length: scale down for larger models
    if free_hbm > 8:
        seq_len = 4096
    elif free_hbm > 4:
        seq_len = 2048
    elif free_hbm > 2:
        seq_len = 1024
    else:
        seq_len = 512

    # Batch size: conservative for large models
    if params_b < 3:
        batch_size = 4
    elif params_b < 13:
        batch_size = 2
    else:
        batch_size = 1

    return {
        "num_cores": tp_degree,
        "sequence_length": seq_len,
        "batch_size": batch_size,
        "auto_cast_type": "bf16",
        "weight_memory_gb": round(weight_gb, 1),
        "hbm_per_core_gb": hbm_per_core,
        "total_cores_available": total_cores,
    }


def _get_default_region() -> str:
    return os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def get_sagemaker_client(region: str = "") -> boto3.client:
    """Get a SageMaker client. Region from param, env var, or us-east-1 fallback."""
    return boto3.client("sagemaker", region_name=region or _get_default_region())


def get_s3_client(region: str = "") -> boto3.client:
    """Get an S3 client. Region from param, env var, or us-east-1 fallback."""
    return boto3.client("s3", region_name=region or _get_default_region())


_TRUSTED_ECR_REGISTRIES = {os.environ.get("AWS_DLC_REGISTRY", "763104351884")}


def _get_latest_ecr_image(repo_name: str, region: str, registry: str = None) -> str:
    """Query ECR for the latest tagged image in a repository."""
    registry = registry or os.environ.get("AWS_DLC_REGISTRY", "763104351884")
    if registry not in _TRUSTED_ECR_REGISTRIES:
        raise ValueError(f"Untrusted ECR registry: {registry}. Configure AWS_DLC_REGISTRY env var.")
    ecr = boto3.client("ecr", region_name=region)
    resp = ecr.describe_images(
        registryId=registry,
        repositoryName=repo_name,
        filter={"tagStatus": "TAGGED"},
        maxResults=50,
    )
    images = [img for img in resp["imageDetails"] if "imageTags" in img]
    if not images:
        return ""
    latest = max(images, key=lambda x: x["imagePushedAt"])
    tag = next((t for t in latest["imageTags"] if t != "latest"), latest["imageTags"][0])
    return f"{registry}.dkr.ecr.{region}.amazonaws.com/{repo_name}:{tag}"


# Map use cases to ECR repository names
_USE_CASE_REPOS = {
    "training": "huggingface-pytorch-training-neuronx",
    "inference": "huggingface-pytorch-inference-neuronx",
    "vllm": "huggingface-vllm-inference-neuronx",
}


def get_container_image(use_case: str, region: str = "") -> str:
    """Get the Neuron DLC image URI for a given use case and region.

    Dynamically queries ECR for the latest patched image. Falls back to
    hardcoded defaults only if the ECR lookup fails.
    """
    region = region or _get_default_region()
    dlc = NEURON_DLCS.get(use_case)
    if not dlc:
        raise ValueError(f"Unknown use case: {use_case}. Choose from: {list(NEURON_DLCS.keys())}")

    repo = _USE_CASE_REPOS.get(use_case)
    if repo:
        try:
            uri = _get_latest_ecr_image(repo, region)
            if uri:
                return uri
        except Exception:
            pass  # intentional fallback — if ECR lookup fails, use hardcoded default image

    return dlc["image"].format(region=region)


def get_sagemaker_role() -> str:
    """Get the SageMaker execution role from env or SageMaker session."""
    role = os.environ.get("SAGEMAKER_ROLE_ARN", "")
    if role:
        return role
    try:
        import sagemaker
        return sagemaker.get_execution_role()
    except Exception:
        return ""
