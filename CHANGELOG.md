# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-26

### Added
- Initial release of Kiro Power for Hugging Face models on SageMaker AI Neuron
- 9 MCP tools: deploy_model, create_training_job, invoke_endpoint, recommend_instance, list_endpoints, describe_endpoint, delete_endpoint, wait_for_endpoint, describe_training_job
- HF Hub MCP Server integration for model discovery and search
- Custom SageMaker AI Neuron MCP Server with FastMCP
- 4 steering docs: TorchNeuron Patterns, SageMaker AI Deployment, Distributed Training, Model Optimization
- Gradio chat UI for deployed endpoints
- SageMaker AI Studio notebook for end-to-end testing
- Auto-retry for Neuron cold-start compilation timeouts
- Dynamic ECR container image lookup (no hardcoded image tags)
- Neuron compile parameter auto-derivation from model size and instance type
- LoRA fine-tuning with optimum-neuron NeuronSFTTrainer
- Two-phase Neuron training (compile then train)
- Portable run_server.sh with auto-venv creation
