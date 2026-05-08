"""Launch SageMaker AI training jobs for Hugging Face models on Trainium."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from ..utils.aws import get_sagemaker_role, get_container_image

# Training scripts are in separate files (not embedded) to avoid inline script patterns.
# They are copied to a temp dir and uploaded to SageMaker AI for remote execution on Trainium.
_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

# Neuron compiler environment flags (recommended by HF docs)
_NEURON_ENV = {
    "NEURON_CC_FLAGS": "--model-type transformer --retry_failed_compilation",
    "NEURON_FUSE_SOFTMAX": "1",
    "NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS": "3",
    "MALLOC_ARENA_MAX": "64",
}

# Instance local storage limits
_INSTANCE_VOLUMES = {
    "ml.trn1.2xlarge": 474,
    "ml.trn1.32xlarge": 7600,
    "ml.trn1n.32xlarge": 7600,
    "ml.trn2.48xlarge": 7600,
}

# Neuron cores per instance (for tensor parallelism)
_INSTANCE_CORES = {
    "ml.trn1.2xlarge": 2,
    "ml.trn1.32xlarge": 32,
    "ml.trn1n.32xlarge": 32,
    "ml.trn2.48xlarge": 64,
}


def register_training_tools(mcp: FastMCP):

    @mcp.tool()
    def create_training_job(
        model_id: str,
        job_name: str,
        instance_type: str = "",
        region: str = "",
        role_arn: str = "",
        s3_output_path: str = "",
        source_dir: str = "",
        entry_point: str = "",
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        hyperparameters: str = "{}",
        instance_count: int = 1,
        volume_size_gb: int = 0,
        max_runtime_seconds: int = 86400,
        tensor_parallel_size: int = 0,
        kms_key_id: str = "",
    ) -> str:
        """Launch a fine-tuning job for a Hugging Face model on Trainium via SageMaker AI.

        Uses optimum-neuron (NeuronSFTTrainer) — the HF-recommended approach for Trainium.

        Args:
            model_id: HF model ID (e.g., 'Qwen/Qwen3-1.7B')
            job_name: Name for the SageMaker AI training job
            instance_type: Trainium instance type (from env INSTANCE_TYPE or provide explicitly)
            region: AWS region (from env AWS_DEFAULT_REGION or provide explicitly)
            role_arn: IAM role ARN (auto-detected from env SAGEMAKER_ROLE_ARN or SageMaker session)
            s3_output_path: S3 path for model artifacts (from env S3_OUTPUT_PATH or provide explicitly)
            source_dir: Local directory containing training script (uses default if empty)
            entry_point: Training script filename (default: train.py)
            dataset_name: HF dataset name for default script (default: wikitext)
            dataset_config: HF dataset config for default script (default: wikitext-2-raw-v1)
            hyperparameters: JSON string of additional hyperparameters
            instance_count: Number of training instances
            volume_size_gb: EBS volume size in GB (auto-sized per instance if 0)
            max_runtime_seconds: Max training time in seconds (default 24h)
            tensor_parallel_size: Neuron cores for tensor parallelism (auto-detected from instance if 0)
            kms_key_id: Optional KMS key ARN for encryption at rest (uses AWS-managed keys if empty)

        Required IAM Permissions:
            sagemaker:CreateTrainingJob, DescribeTrainingJob
            s3:GetObject, PutObject, ListBucket on training data and output paths
            ecr:GetAuthorizationToken, BatchCheckLayerAvailability, GetDownloadUrlForLayer, BatchGetImage
            logs:CreateLogGroup, CreateLogStream, PutLogEvents

        Note: S3 buckets must have Block Public Access enabled, encryption at rest,
        and bucket policies enforcing TLS/HTTPS transport.
        """
        region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        instance_type = instance_type or os.environ.get("INSTANCE_TYPE", "")
        role_arn = role_arn or get_sagemaker_role()
        s3_output_path = s3_output_path or os.environ.get("S3_OUTPUT_PATH", "")

        if not instance_type:
            return json.dumps({"error": "instance_type is required. Set INSTANCE_TYPE env var or pass explicitly."})
        if not role_arn:
            return json.dumps({"error": "role_arn is required. Set SAGEMAKER_ROLE_ARN env var or pass explicitly."})
        if not s3_output_path:
            return json.dumps({"error": "s3_output_path is required. Set S3_OUTPUT_PATH env var or pass explicitly."})
        if not job_name:
            return json.dumps({"error": "job_name is required."})

        try:
            import sagemaker
            from sagemaker.huggingface import HuggingFace

            sess = sagemaker.Session(boto_session=__import__('boto3').Session(region_name=region))
            extra_hp = json.loads(hyperparameters) if hyperparameters else {}
            if not isinstance(extra_hp, dict):
                return json.dumps({"error": "hyperparameters must be a JSON object."})

            if volume_size_gb <= 0:
                volume_size_gb = _INSTANCE_VOLUMES.get(instance_type, 256)

            if tensor_parallel_size <= 0:
                tensor_parallel_size = 1

            # Copy default training scripts to temp dir if no source_dir provided
            if not source_dir:
                tmp_dir = tempfile.mkdtemp(prefix="hf_train_")
                shutil.copy2(_SCRIPTS_DIR / "train.py", os.path.join(tmp_dir, "train.py"))
                shutil.copy2(_SCRIPTS_DIR / "launch.py", os.path.join(tmp_dir, "launch.py"))
                shutil.copy2(_SCRIPTS_DIR / "requirements.txt", os.path.join(tmp_dir, "requirements.txt"))
                source_dir = tmp_dir
                entry_point = "launch.py"
            elif not entry_point:
                entry_point = "train.py"

            hp = {
                "model_id": model_id,
                "dataset_name": dataset_name,
                "dataset_config": dataset_config,
                "tensor_parallel_size": str(tensor_parallel_size),
                "bf16": "",
                "do_train": "",
                "gradient_accumulation_steps": "8",
                "logging_steps": "2",
                "lr_scheduler_type": "cosine",
                "overwrite_output_dir": "",
                **extra_hp,
            }

            image_uri = get_container_image("training", region)

            # Add Neuron compilation cache to S3 (skip recompilation on subsequent runs)
            env = {
                **_NEURON_ENV,
                "NEURON_COMPILE_CACHE_URL": f"{s3_output_path}/neuron-cache",
            }

            # No distribution parameter — launch.py handles torchrun internally
            # with the correct nproc_per_node = tensor_parallel_size.
            estimator = HuggingFace(
                entry_point=entry_point,
                source_dir=source_dir,
                role=role_arn,
                instance_type=instance_type,
                instance_count=instance_count,
                volume_size=volume_size_gb,
                image_uri=image_uri,
                py_version="py310",
                output_path=s3_output_path,
                max_run=max_runtime_seconds,
                hyperparameters=hp,
                sagemaker_session=sess,
                base_job_name=job_name,
                environment=env,
            )
            if kms_key_id:
                estimator_kwargs["volume_kms_key"] = kms_key_id
                estimator_kwargs["output_kms_key"] = kms_key_id

            estimator.fit(wait=False)
            actual_job_name = estimator.latest_training_job.name

        except Exception as e:
            return json.dumps({"error": str(e), "job_name": job_name})

        return json.dumps({
            "status": "starting",
            "job_name": actual_job_name,
            "model_id": model_id,
            "instance_type": instance_type,
            "instance_count": instance_count,
            "tensor_parallel_size": tensor_parallel_size,
            "entry_point": entry_point,
            "region": region,
            "message": f"Training job '{actual_job_name}' launched via HuggingFace Estimator with optimum-neuron. Monitor in SageMaker AI console.",
        })
