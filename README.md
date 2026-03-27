# Kiro Power: Fine-tuning and Deploying Hugging Face Models on AWS Neuron

Kiro Power for fine-tuning and deploying Hugging Face models on AWS with **Amazon SageMaker AI** and **AWS Neuron** (Trainium / Inferentia).

## Architecture

```
                          ┌──────────────────────────────────────────────────┐
                          │                   Kiro IDE                       │
                          │                                                  │
                          │  ┌────────────────────────────────────────────┐  │
                          │  │        Kiro Power (power.json)             │  │
                          │  │                                            │  │
                          │  │  Steering Docs:                            │  │
                          │  │   • TorchNeuron Patterns                   │  │
                          │  │   • SageMaker AI Deployment                   │  │
                          │  │   • Distributed Training                   │  │
                          │  │   • Model Optimization                     │  │
                          │  │                                            │  │
  ┌──────────────────┐    │  │  MCP Servers:                              │  │
  │ Amazon Bedrock   │    │  │   • Hugging Face MCP Server (remote) ──────┼──┼──► Hugging Face Hub
  │                  │    │  │     Model discovery, search, model cards   │  │    (models, datasets)
  │ Claude Sonnet 4.x│◄──-┼──┼──►                                         │  │
  │ (orchestrates,   │    │  │   • SageMaker AI Neuron MCP Server (custom) ──┼──┼──► Amazon ECR
  │  grounds,        │    │  │     11 tools: recommend, deploy, train,    │  │    (DLC images)
  │  validates)      │    │  │     invoke, list, describe, delete,        │  │
  └──────────────────┘    │  │     wait_for_endpoint, describe_job ───────┼──┼───┐
                          │  └────────────────────────────────────────────┘  │   │
                          │                                                  │   │
                          └──────────────────────────────────────────────────┘   │
                                                                                 │
                                                                                 ▼
  ┌─────────┐                                                           Amazon SageMaker AI
  │ Gradio  │──────────────────────────────────────────────────────►    (endpoints, jobs)
  │ Chat UI │                                                                    │
  └─────────┘                                                              ┌─────┴─────┐
                                                                           ▼           ▼
                                                                      AWS Trainium  AWS Inferentia
                                                                     (fine-tuning)    (inference)
```

## What It Does

This Power gives Kiro two complementary MCP servers:

- **HF Hub MCP** — Model discovery, search, and metadata from Hugging Face Hub
- **SageMaker AI Neuron MCP** — Deploy, train, invoke, and manage models on AWS Neuron hardware

Together they enable an end-to-end workflow from the IDE: discover a model → recommend an instance → fine-tune on Trainium → deploy on Inferentia → run inference — all through natural language in Kiro chat. Kiro uses Claude to ground and validate model responses for accuracy.

## MCP Tools (11)

### Deployment & Inference
| Tool | Description |
|---|---|
| `deploy_model` | Deploy a Hugging Face model to a SageMaker AI endpoint with Neuron containers. Supports both Hub models and fine-tuned models from S3. Checks for existing endpoints and polls until InService. |
| `invoke_endpoint` | Send a prompt to a deployed endpoint and return generated text. Applies Qwen3 chat template, strips thinking tokens, and auto-retries up to 3 times during Neuron cold-start compilation. |

### Training
| Tool | Description |
|---|---|
| `create_training_job` | Launch a fine-tuning job on Trainium via SageMaker AI. Uses LoRA with optimum-neuron's NeuronSFTTrainer. Includes two-phase Neuron compilation and auto-consolidates adapters. Output adapter can be merged with the base model for deployment (see Workflow). |
| `describe_training_job` | Get details and status of a SageMaker AI training job including instance type, start/end times, and S3 model artifact location. |

### Endpoint Management
| Tool | Description |
|---|---|
| `list_endpoints` | List active SageMaker AI endpoints, optionally filtered by status (InService, Creating, Failed, All). |
| `describe_endpoint` | Get details of a SageMaker AI endpoint including status, creation time, and ARN. |
| `delete_endpoint` | Delete a SageMaker AI endpoint. Requires manual approval in Kiro. |
| `wait_for_endpoint` | Poll an endpoint every 2 minutes until it reaches InService or fails. Times out after 15 minutes. |
| `set_active_endpoint` | Set the active endpoint for subsequent invoke calls. Questions are automatically routed to this endpoint. |
| `get_active_endpoint` | Get the currently active endpoint name. |

### Recommendation
| Tool | Description |
|---|---|
| `recommend_instance` | Recommend the optimal Neuron instance type for any Hugging Face model. Auto-derives Neuron compile parameters (TP degree, sequence length, batch size) based on model size and instance HBM. Fetches model metadata from HF Hub or accepts user-provided parameter count. |

Also integrates with the [Hugging Face MCP Server](https://huggingface.co/mcp) for model/dataset search on HF Hub.

## Workflow

1. **Train** — Fine-tune a model using `create_training_job` (automatically merges adapter into base model)
2. **Deploy** — Deploy the model using `deploy_model`
3. **Set Endpoint** — Set the active endpoint using `set_active_endpoint`
4. **Invoke** — Ask questions directly (routed to active endpoint via `invoke_endpoint`)

## Demo Flow

1. **Search** — "Find the best text generation model under 1B parameters" (HF Hub MCP)
2. **Recommend** — "Recommend an AWS Neuron instance for Qwen3-0.6B" (auto-derives compile params)
3. **Training proof** — "Show me training job qwen3-0-6b-finetune-..." (completed on trn1.2xlarge)
4. **Deploy** — "Deploy my fine-tuned model to inf2.8xlarge" (checks if exists, polls until ready)
5. **Inference** — "What is machine learning?" (routed to active endpoint via Neuron)

## Sample Queries

These queries have been tested and produce reliable results. Replace `my-endpoint` and `my-bucket` with your actual endpoint name and S3 bucket before querying.

- "Search for the best text generation model under 1B parameters and recommend an AWS Neuron instance"
- "Show me the status of my training job"
- "Deploy my fine-tuned model from s3://my-bucket/model.tar.gz to inf2.8xlarge endpoint named my-endpoint"
- "Set active endpoint to my-endpoint"
- "What is transfer learning?"
- "What is machine learning?"
- "Explain what a neural network is"
- "What is deep learning?"
- "What is the difference between AI and machine learning?"
- "Describe endpoint my-endpoint"
- "Delete endpoint my-endpoint"

## Setup

### 1. Open in Kiro IDE

Open this project folder in Kiro. In the chat panel, select **Claude Sonnet 4.5** from the model dropdown at the bottom.

### 2. Enable MCP Servers

Go to the MCP dropdown in Kiro and click **Enable MCP**. Then open Workspace MCP Config (JSON) and add:

```json
{
  "mcpServers": {
    "huggingface-hub": {
      "url": "https://huggingface.co/mcp",
      "disabled": false
    },
    "sagemaker-neuron": {
      "command": "bash",
      "args": ["mcp/sagemaker-neuron-server/run_server.sh"],
      "env": {
        "AWS_DEFAULT_REGION": "us-west-2",
        "AWS_PROFILE": "your-profile",
        "SAGEMAKER_ROLE_ARN": "arn:aws:iam::YOUR_ACCOUNT:role/YOUR_ROLE"
      },
      "disabled": false,
      "autoApprove": ["recommend_instance", "list_endpoints", "describe_endpoint", "describe_training_job", "invoke_endpoint"]
    }
  }
}
```

The `run_server.sh` script auto-creates a Python virtual environment and installs dependencies on first run.

### 3. Verify

Check the MCP dropdown in Kiro — both servers should show as connected.

## Grounding with Claude

Kiro uses Claude as an intelligent layer on top of deployed models. When the deployed model returns a response, Claude validates the output for accuracy and adds context — catching inaccuracies and providing corrections. This grounding ensures users get reliable information even from smaller models.

## Gradio Demo

```bash
cd demo
pip install gradio boto3
AWS_PROFILE=your-profile python app.py
```

Opens a chat UI at `http://127.0.0.1:7860` connected to your deployed endpoint.

| Variable | Description | Default |
|----------|-------------|---------|
| `ENDPOINT_NAME` | SageMaker AI endpoint name | `qwen3-finetuned-kiro` |
| `AWS_DEFAULT_REGION` | AWS region | `us-west-2` |
| `MAX_NEW_TOKENS` | Max tokens to generate | `512` |
| `AWS_PROFILE` | AWS CLI profile | `default` |

## Steering

| File | Content |
|---|---|
| `steering/torchneuron.md` | TorchNeuron patterns, device placement, torch.compile, CUDA migration |
| `steering/sagemaker-deployment.md` | Container URIs, deployment patterns, SDK usage |
| `steering/distributed-training.md` | DDP, FSDP, TP parallelism on Trainium |
| `steering/model-optimization.md` | Compilation, BF16, Neuron cache, vLLM, NKI |

## Structure

```
├── power.json                          # Power manifest
├── demo/
│   └── app.py                          # Gradio chat demo
├── steering/                           # Agent steering docs
├── mcp/sagemaker-neuron-server/
│   ├── run_server.sh                   # Auto-venv launcher
│   ├── pyproject.toml                  # Package config
│   └── sagemaker_neuron_server/
│       ├── __init__.py                 # Registers all 11 tools with FastMCP
│       ├── __main__.py                 # Entry point for python -m
│       ├── tools/
│       │   ├── deploy.py               # deploy_model
│       │   ├── endpoint.py             # list, describe, delete, wait, invoke
│       │   ├── training.py             # create_training_job
│       │   └── recommend.py            # recommend_instance
│       ├── utils/
│       │   └── aws.py                  # ECR lookup, Neuron params, instance recs
│       └── scripts/
│           ├── train.py                # LoRA fine-tuning script (runs on Trainium)
│           └── launch.py               # Two-phase launcher (runs on Trainium)
└── docs/                               # Domain knowledge + examples
```

## Security

### Shared Responsibility Model

This solution follows the [AWS Shared Responsibility Model](https://aws.amazon.com/compliance/shared-responsibility-model/):

**AWS Responsibilities:**
- Securing the underlying infrastructure for Amazon SageMaker AI, AWS Trainium, and AWS Inferentia
- Patching and maintaining Neuron DLC container images
- Physical security of data centers and network infrastructure

**Customer Responsibilities:**
- Configuring IAM roles and policies with least-privilege access
- Securing data in transit and at rest (encryption keys, S3 bucket policies)
- Network configuration (VPC, security groups, endpoint policies)
- Monitoring and logging (CloudTrail, CloudWatch)
- Application code security and dependency management
- Endpoint lifecycle management and cost control

### Built-in Security Controls

- No hardcoded container image tags — dynamically queries ECR for latest patched images
- No hardcoded credentials — all via environment variables or IAM roles
- No embedded scripts — training scripts are separate files
- Destructive tools (deploy, delete, train) require manual approval in Kiro

### IAM Permissions

The IAM role specified in `SAGEMAKER_ROLE_ARN` requires the following minimum permissions:

**For Deployment (deploy_model):**
- `sagemaker:CreateModel`, `CreateEndpoint`, `CreateEndpointConfig`, `DescribeEndpoint`, `DeleteEndpoint`
- `s3:GetObject` on model artifact locations
- `ecr:GetAuthorizationToken`, `BatchCheckLayerAvailability`, `GetDownloadUrlForLayer`, `BatchGetImage`
- `logs:CreateLogGroup`, `CreateLogStream`, `PutLogEvents`

**For Training (create_training_job):**
- `sagemaker:CreateTrainingJob`, `DescribeTrainingJob`
- `s3:GetObject`, `PutObject`, `ListBucket` on training data and output paths
- `ecr:GetAuthorizationToken`, `BatchCheckLayerAvailability`, `GetDownloadUrlForLayer`, `BatchGetImage`

**For Inference (invoke_endpoint):**
- `sagemaker-runtime:InvokeEndpoint`

### S3 Security Requirements

All S3 buckets used by this solution must be configured with:
- **Block Public Access**: Enable all four settings
- **Encryption at rest**: Configure server-side encryption (SSE-S3 or SSE-KMS)
- **TLS/HTTPS enforcement**: Add bucket policy requiring `aws:SecureTransport`
- **Access logging**: Enable S3 access logging for audit trails

### Encryption

- **At rest**: Amazon SageMaker AI encrypts data at rest by default using AWS-managed keys. For additional control, specify a customer-managed KMS key.
- **In transit**: All API calls use TLS 1.2+. Enable inter-container traffic encryption for distributed training.

### Service-Specific Security

- **Amazon SageMaker AI**: Use VPC mode for endpoints to isolate network traffic. Enable network isolation for training jobs when internet access is not required.
- **AWS Neuron**: Neuron compilation cache may contain model architecture information — secure S3 bucket access accordingly.
- **Amazon ECR**: Enable image scanning to detect vulnerabilities. This solution dynamically fetches the latest patched images.
- **Amazon S3**: Use bucket policies to restrict access to authorized principals only.

### Third-Party Dependencies

This solution integrates with the [Hugging Face MCP Server](https://huggingface.co/mcp), a remote third-party service for model discovery. Users should review the [Hugging Face Terms of Service](https://huggingface.co/terms-of-service) and assess the security implications of this external dependency for their use case.

For more information, see [Security in Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/security.html).

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file.

## Disclaimer

This solution is for demonstrative purposes only and is not intended for production use without additional security and compliance review. It is each customer's responsibility to ensure proper configuration of AWS services, IAM roles, and endpoint security for their specific use case.
