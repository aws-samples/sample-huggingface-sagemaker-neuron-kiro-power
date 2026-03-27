# Security Policy

## Reporting Security Issues

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.

For more information about reporting security issues, please see [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications).

## AWS Shared Responsibility Model

This sample code operates under the [AWS Shared Responsibility Model](https://aws.amazon.com/compliance/shared-responsibility-model/). AWS is responsible for securing the underlying cloud infrastructure (Amazon SageMaker AI, AWS Trainium, AWS Inferentia), while you are responsible for securing your data, configuring IAM roles, managing network access, and ensuring your application code follows security best practices.

For detailed information, see:
- [AWS Shared Responsibility Model](https://aws.amazon.com/compliance/shared-responsibility-model/)
- [Security in Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/security.html)

## Security Implementation Guide

### 1. IAM Role Configuration (Priority: High)

Create a least-privilege IAM role for SageMaker AI execution. Scope permissions to specific resources:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateModel",
                "sagemaker:CreateEndpoint",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:DescribeEndpoint",
                "sagemaker:DeleteEndpoint",
                "sagemaker:CreateTrainingJob",
                "sagemaker:DescribeTrainingJob"
            ],
            "Resource": "arn:aws:sagemaker:<region>:<account-id>:*"
        },
        {
            "Effect": "Allow",
            "Action": ["sagemaker-runtime:InvokeEndpoint"],
            "Resource": "arn:aws:sagemaker:<region>:<account-id>:endpoint/*"
        },
        {
            "Effect": "Allow",
            "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
            "Resource": ["arn:aws:s3:::<bucket-name>", "arn:aws:s3:::<bucket-name>/*"]
        },
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
            "Resource": "arn:aws:logs:<region>:<account-id>:*"
        }
    ]
}
```

### 2. Network Isolation (Priority: High)

Deploy endpoints in private subnets with VPC configuration:

```python
model.deploy(
    instance_type="ml.inf2.xlarge",
    vpc_config={
        "SecurityGroupIds": ["sg-xxxxx"],
        "Subnets": ["subnet-xxxxx"]
    }
)
```

### 3. Encryption (Priority: Medium)

Enable KMS encryption for training and deployment:

```python
estimator = HuggingFace(
    volume_kms_key="arn:aws:kms:<region>:<account-id>:key/xxxxx",
    output_kms_key="arn:aws:kms:<region>:<account-id>:key/xxxxx",
    enable_inter_container_traffic_encryption=True
)
```

### 4. Credential Management (Priority: Medium)

Use AWS Secrets Manager for tokens instead of environment variables in production:

```python
import boto3
secrets = boto3.client('secretsmanager')
token = secrets.get_secret_value(SecretId='hf-token')['SecretString']
```

## Security Best Practices

This sample code is provided for educational and demonstration purposes. Before using any code in production:

- Configure IAM roles with least-privilege permissions scoped to specific resources
- Enable VPC isolation for SageMaker AI endpoints and training jobs
- Configure encryption at rest (KMS) and in transit (TLS) for all data
- Enable S3 Block Public Access and bucket policies requiring HTTPS
- Enable CloudTrail logging and CloudWatch monitoring
- Review and rotate credentials regularly
- Keep DLC container images updated (this solution dynamically fetches latest images)

## Disclaimer

This is sample code, for non-production usage. You should work with your security and legal teams to meet your organizational security, regulatory and compliance requirements before deployment.
