"""Deploy Hugging Face models to SageMaker AI endpoints with Neuron containers."""

import json
import time
from botocore.exceptions import ClientError
from mcp.server.fastmcp import FastMCP
from ..utils.aws import get_sagemaker_client, get_container_image, get_sagemaker_role


def register_deploy_tools(mcp: FastMCP):

    @mcp.tool()
    def deploy_model(
        model_id: str,
        endpoint_name: str,
        instance_type: str = "",
        region: str = "",
        role_arn: str = "",
        serving_type: str = "inference",
        num_neuron_cores: int = 2,
        env_vars: str = "{}",
        image_uri: str = "",
        model_data: str = "",
        kms_key_id: str = "",
    ) -> str:
        """Deploy a Hugging Face model to a SageMaker AI endpoint with Neuron.

        Args:
            model_id: HF model ID (e.g., 'meta-llama/Llama-3.3-70B-Instruct'). Ignored when model_data is set.
            endpoint_name: Name for the SageMaker endpoint
            instance_type: SageMaker instance type (from env INSTANCE_TYPE or provide explicitly)
            region: AWS region (from env AWS_DEFAULT_REGION or provide explicitly)
            role_arn: IAM role ARN (auto-detected from env SAGEMAKER_ROLE_ARN or SageMaker session)
            serving_type: 'inference' for general models, 'vllm' for LLM serving
            num_neuron_cores: Number of Neuron cores for model sharding
            env_vars: JSON string of additional environment variables
            image_uri: Override container image URI (uses latest ECR image if empty)
            model_data: S3 URI of model.tar.gz for fine-tuned model deployment
            kms_key_id: Optional KMS key ARN for encryption at rest (uses AWS-managed keys if empty)

        Required IAM Permissions:
            sagemaker:CreateModel, CreateEndpoint, CreateEndpointConfig, DescribeEndpoint, DeleteEndpoint
            s3:GetObject on model artifact locations
            ecr:GetAuthorizationToken, BatchCheckLayerAvailability, GetDownloadUrlForLayer, BatchGetImage
            logs:CreateLogGroup, CreateLogStream, PutLogEvents
        """
        import os
        instance_type = instance_type or os.environ.get("INSTANCE_TYPE", "")
        role_arn = role_arn or get_sagemaker_role()

        # Force inference serving type for fine-tuned models from S3
        # (vLLM containers don't support model_data tar.gz pattern)
        if model_data and serving_type == "vllm":
            serving_type = "inference"

        if not instance_type:
            return json.dumps({"error": "instance_type is required. Set INSTANCE_TYPE env var or pass explicitly."})
        if not role_arn:
            return json.dumps({"error": "role_arn is required. Set SAGEMAKER_ROLE_ARN env var or pass explicitly."})
        if not endpoint_name:
            return json.dumps({"error": "endpoint_name is required."})

        try:
            sm = get_sagemaker_client(region)

            # Check if endpoint already exists
            try:
                resp = sm.describe_endpoint(EndpointName=endpoint_name)
                status = resp["EndpointStatus"]
                return json.dumps({
                    "status": status.lower(),
                    "endpoint_name": endpoint_name,
                    "message": f"Endpoint '{endpoint_name}' already exists (status: {status}). Use describe_endpoint for details or delete_endpoint to recreate.",
                })
            except sm.exceptions.ClientError:
                pass  # Endpoint doesn't exist, proceed with creation

            image_uri = image_uri or get_container_image(serving_type, region)
            extra_env = json.loads(env_vars) if env_vars else {}
            if not isinstance(extra_env, dict):
                return json.dumps({"error": "env_vars must be a JSON object."})

            env = {
                "NEURON_RT_NUM_CORES": str(num_neuron_cores),
                "HF_NUM_CORES": str(num_neuron_cores),
                "HF_AUTO_CAST_TYPE": "bf16",
                "HF_OPTIMUM_BATCH_SIZE": "1",
                "HF_OPTIMUM_SEQUENCE_LENGTH": "4096",
                "NEURON_COMPILE_CACHE_URL": "/tmp/neuron-cache",  # container-only path, /tmp is the only writable dir in SageMaker inference containers
                "NEURONX_DUMP_TO": "/tmp",  # container-only path, required for Neuron compiler output
                **extra_env,
            }
            if not model_data:
                env["HF_MODEL_ID"] = model_id

            if serving_type == "vllm":
                env.setdefault("MAX_MODEL_LEN", "4096")
                env["TENSOR_PARALLEL_SIZE"] = str(num_neuron_cores)

            timestamp = int(time.time())
            model_name = f"{endpoint_name}-model-{timestamp}"
            config_name = f"{endpoint_name}-config-{timestamp}"

            container = {
                "Image": image_uri,
                "Environment": env,
            }
            if model_data:
                container["ModelDataUrl"] = model_data

            sm.create_model(
                ModelName=model_name,
                PrimaryContainer=container,
                ExecutionRoleArn=role_arn,
            )

            endpoint_config = {
                "EndpointConfigName": config_name,
                "ProductionVariants": [{
                    "VariantName": "primary",
                    "ModelName": model_name,
                    "InstanceType": instance_type,
                    "InitialInstanceCount": 1,
                    "ContainerStartupHealthCheckTimeoutInSeconds": 900,
                    "ModelDataDownloadTimeoutInSeconds": 900,
                }],
            }
            if kms_key_id:
                endpoint_config["KmsKeyId"] = kms_key_id
            sm.create_endpoint_config(**endpoint_config)

            sm.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name,
            )

        except ClientError as e:
            return json.dumps({"error": str(e), "endpoint_name": endpoint_name})
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON in env_vars parameter."})

        # Poll until InService or failed
        import time as _time
        deadline = _time.time() + 15 * 60
        while _time.time() < deadline:
            _time.sleep(120)  # nosemgrep: arbitrary-sleep
            try:
                ep = sm.describe_endpoint(EndpointName=endpoint_name)
                status = ep["EndpointStatus"]
                if status == "InService":
                    return json.dumps({
                        "status": "InService",
                        "endpoint_name": endpoint_name,
                        "instance_type": instance_type,
                        "message": f"✅ Endpoint '{endpoint_name}' is READY for inference.",
                    })
                if status == "Failed":
                    return json.dumps({
                        "status": "Failed",
                        "endpoint_name": endpoint_name,
                        "reason": ep.get("FailureReason", "Unknown"),
                    })
            except ClientError:
                pass

        return json.dumps({
            "status": "creating",
            "endpoint_name": endpoint_name,
            "instance_type": instance_type,
            "message": f"Endpoint '{endpoint_name}' still creating after 15 minutes. Use describe_endpoint to check.",
        })
