"""Manage SageMaker AI endpoints and training jobs."""

import json
import os
from botocore.exceptions import ClientError
from mcp.server.fastmcp import FastMCP
from ..utils.aws import get_sagemaker_client


def _region(region: str = "") -> str:
    return region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


_active_endpoint = {"name": ""}


def register_endpoint_tools(mcp: FastMCP):

    @mcp.tool()
    def set_active_endpoint(endpoint_name: str) -> str:
        """Set the active endpoint for subsequent invoke calls.

        Args:
            endpoint_name: Name of the SageMaker endpoint to use as default
        """
        _active_endpoint["name"] = endpoint_name
        return json.dumps({
            "active_endpoint": endpoint_name,
            "message": f"Active endpoint set to '{endpoint_name}'. All subsequent user questions should be routed to this endpoint via invoke_endpoint unless the user explicitly asks for general knowledge.",
        })

    @mcp.tool()
    def get_active_endpoint() -> str:
        """Get the currently active endpoint name."""
        return json.dumps({"active_endpoint": _active_endpoint["name"] or "(none set)"})

    @mcp.tool()
    def list_endpoints(region: str = "", status_filter: str = "InService") -> str:
        """List SageMaker AI endpoints, optionally filtered by status.

        Args:
            region: AWS region
            status_filter: Filter by status ('InService', 'Creating', 'Failed', 'All')
        """
        sm = get_sagemaker_client(_region(region))
        kwargs = {}
        if status_filter != "All":
            kwargs["StatusEquals"] = status_filter

        response = sm.list_endpoints(**kwargs, MaxResults=50)
        endpoints = [{
            "name": ep["EndpointName"],
            "status": ep["EndpointStatus"],
            "created": ep["CreationTime"].isoformat(),
        } for ep in response.get("Endpoints", [])]

        return json.dumps({"endpoints": endpoints, "count": len(endpoints)})

    @mcp.tool()
    def describe_endpoint(endpoint_name: str, region: str = "") -> str:
        """Get details of a SageMaker AI endpoint.

        Args:
            endpoint_name: Name of the SageMaker endpoint
            region: AWS region
        """
        sm = get_sagemaker_client(_region(region))
        try:
            ep = sm.describe_endpoint(EndpointName=endpoint_name)
        except ClientError as e:
            return json.dumps({"error": str(e), "endpoint_name": endpoint_name})

        return json.dumps({
            "name": ep["EndpointName"],
            "status": ep["EndpointStatus"],
            "created": ep["CreationTime"].isoformat(),
            "last_modified": ep["LastModifiedTime"].isoformat(),
            "arn": ep["EndpointArn"],
        })

    @mcp.tool()
    def delete_endpoint(endpoint_name: str, region: str = "") -> str:
        """Delete a SageMaker AI endpoint.

        Args:
            endpoint_name: Name of the SageMaker endpoint to delete
            region: AWS region
        """
        sm = get_sagemaker_client(_region(region))
        try:
            sm.delete_endpoint(EndpointName=endpoint_name)
        except ClientError as e:
            return json.dumps({"error": str(e), "endpoint_name": endpoint_name})

        return json.dumps({
            "status": "deleting",
            "endpoint_name": endpoint_name,
            "message": f"Endpoint '{endpoint_name}' is being deleted.",
        })

    @mcp.tool()
    def wait_for_endpoint(endpoint_name: str, region: str = "", timeout_minutes: int = 15) -> str:
        """Poll a SageMaker AI endpoint until it reaches InService or fails.

        Args:
            endpoint_name: Name of the SageMaker endpoint
            region: AWS region
            timeout_minutes: Max minutes to wait (default 15)
        """
        import time as _time
        sm = get_sagemaker_client(_region(region))
        deadline = _time.time() + timeout_minutes * 60
        while _time.time() < deadline:
            try:
                ep = sm.describe_endpoint(EndpointName=endpoint_name)
                status = ep["EndpointStatus"]
                if status == "InService":
                    return json.dumps({"endpoint_name": endpoint_name, "status": "InService", "message": f"✅ Endpoint '{endpoint_name}' is READY."})
                if status == "Failed":
                    return json.dumps({"endpoint_name": endpoint_name, "status": "Failed", "reason": ep.get("FailureReason", "Unknown")})
            except ClientError as e:
                return json.dumps({"error": str(e), "endpoint_name": endpoint_name})
            _time.sleep(120)  # nosemgrep: arbitrary-sleep
        return json.dumps({"endpoint_name": endpoint_name, "status": "Timeout", "message": f"Endpoint not ready after {timeout_minutes} minutes."})

    @mcp.tool()
    def invoke_endpoint(endpoint_name: str = "", prompt: str = "", region: str = "", max_new_tokens: int = 512) -> str:
        """Send a prompt to a SageMaker AI endpoint and return the generated text.

        Args:
            endpoint_name: Name of the SageMaker endpoint (uses active endpoint if not specified)
            prompt: Text prompt to send to the model
            region: AWS region
            max_new_tokens: Maximum number of tokens to generate
        """
        endpoint_name = endpoint_name or _active_endpoint["name"]
        if not endpoint_name:
            return json.dumps({"error": "No endpoint specified. Use set_active_endpoint or pass endpoint_name."})
        if not prompt:
            return json.dumps({"error": "prompt is required."})
        import boto3
        import time as _time
        from botocore.config import Config
        from botocore.exceptions import ReadTimeoutError
        runtime = boto3.client("sagemaker-runtime", region_name=_region(region), config=Config(read_timeout=300))
        formatted = f"<|im_start|>user\n/no_think\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        payload = json.dumps({"inputs": formatted, "parameters": {"max_new_tokens": max_new_tokens}})

        # Auto-retry: Neuron compilation on first call can take 5-10 min
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                resp = runtime.invoke_endpoint(
                    EndpointName=endpoint_name, ContentType="application/json", Body=payload,
                )
                body = json.loads(resp["Body"].read().decode())
                text = body[0].get("generated_text", str(body)) if isinstance(body, list) else body.get("generated_text", str(body))
                if text.startswith(formatted):
                    text = text[len(formatted):]
                import re
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
                    text = text.split(tok)[0]
                return json.dumps({"endpoint_name": endpoint_name, "prompt": prompt, "generated_text": text.strip()})
            except (ReadTimeoutError, ClientError) as e:
                if attempt < max_retries:
                    _time.sleep(30)  # nosemgrep: arbitrary-sleep
                    continue
                return json.dumps({
                    "endpoint_name": endpoint_name,
                    "error": f"Model is still warming up (Neuron compilation). Tried {max_retries} times. Please try again in 1-2 minutes.",
                })

    @mcp.tool()
    def describe_training_job(job_name: str, region: str = "") -> str:
        """Get details and status of a SageMaker AI training job.

        Args:
            job_name: Name of the SageMaker AI training job
            region: AWS region
        """
        sm = get_sagemaker_client(_region(region))
        try:
            job = sm.describe_training_job(TrainingJobName=job_name)
        except ClientError as e:
            return json.dumps({"error": str(e), "job_name": job_name})

        result = {
            "job_name": job["TrainingJobName"],
            "status": job["TrainingJobStatus"],
            "instance_type": job["ResourceConfig"]["InstanceType"],
            "instance_count": job["ResourceConfig"]["InstanceCount"],
            "created": job["CreationTime"].isoformat(),
            "last_modified": job["LastModifiedTime"].isoformat(),
            "arn": job["TrainingJobArn"],
        }

        if job.get("TrainingStartTime"):
            result["started"] = job["TrainingStartTime"].isoformat()
        if job.get("TrainingEndTime"):
            result["ended"] = job["TrainingEndTime"].isoformat()
        if job.get("FailureReason"):
            result["failure_reason"] = job["FailureReason"]
        if job.get("ModelArtifacts"):
            result["model_artifacts"] = job["ModelArtifacts"].get("S3ModelArtifacts", "")

        return json.dumps(result)
