# Support

## Getting Help

This is sample code provided for educational and demonstration purposes. Support is provided on a best-effort basis through the following channels:

### GitHub Discussions
- Use [GitHub Discussions](../../discussions) for questions about implementation
- Search existing discussions before creating new ones
- Provide clear descriptions and relevant code snippets

### Documentation
- Review the [README.md](README.md) for setup and usage instructions
- Check the [notebook](Huggingface-kiro-powers.ipynb) for interactive examples in SageMaker AI Studio
- Refer to the steering docs in `steering/` for domain knowledge

### Common Issues
- **MCP Server Not Connecting**: Ensure `run_server.sh` has execute permissions and Python 3 is available
- **AWS Credentials**: Set `AWS_PROFILE` and `SAGEMAKER_ROLE_ARN` in the MCP server env config
- **Neuron Compilation Timeout**: First inference call takes 5-10 minutes for Neuron graph compilation. Subsequent calls are fast.
- **Endpoint Creation**: Requires SageMaker quota for inf2/trn1 instances in your region

## What This Project Is
- **Sample code** demonstrating Kiro Power integration with Hugging Face and AWS Neuron
- **Educational resource** for deploying and fine-tuning HF models on SageMaker AI with Neuron
- **Reference implementation** for MCP-based AI tooling

## What This Project Is Not
- **Production-ready service** without additional security and compliance review
- **Supported AWS service** with SLA guarantees

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for information about contributing to this project.

## Security Issues
For security-related issues, please see [SECURITY.md](SECURITY.md) for reporting instructions.
