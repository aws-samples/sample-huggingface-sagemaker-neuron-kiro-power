"""Gradio chat demo for Hugging Face models on SageMaker Neuron."""

import json
import os

import boto3
import gradio as gr
from botocore.config import Config

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "qwen3-finetuned-kiro")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))

runtime = boto3.client(
    "sagemaker-runtime",
    region_name=REGION,
    config=Config(read_timeout=300),
)


def invoke(prompt: str) -> str:
    """Send a prompt to the SageMaker endpoint."""
    formatted = f"<|im_start|>user\n/no_think\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    payload = json.dumps({
        "inputs": formatted,
        "parameters": {"max_new_tokens": MAX_NEW_TOKENS},
    })
    resp = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=payload,
    )
    body = json.loads(resp["Body"].read().decode())
    if isinstance(body, list):
        text = body[0].get("generated_text", str(body))
    else:
        text = body.get("generated_text", str(body))
    # Strip the prompt prefix from the response if echoed back
    if text.startswith(formatted):
        text = text[len(formatted):]
    # Remove <think>...</think> blocks
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Clean up any trailing special tokens
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        text = text.split(tok)[0]
    return text.strip()


def chat(message: str, history: list) -> str:
    """Chat handler for Gradio."""
    try:
        return invoke(message)
    except Exception as e:
        return f"Error: {e}"


demo = gr.ChatInterface(
    fn=chat,
    title="🤗 Fine-tuning and Deploying Hugging Face Models on AWS Neuron using Kiro Power",
    description=(
        f"Chat with a fine-tuned **Qwen3-0.6B** model deployed on "
        f"**Amazon SageMaker AI** with **AWS Inferentia2** (endpoint: `{ENDPOINT_NAME}`). "
        f"Powered by the HF + AWS Kiro Power MCP integration."
    ),
    examples=[
        "What is transfer learning?",
        "What is machine learning?",
        "Explain what a neural network is",
        "What is deep learning?",
        "What is the difference between AI and machine learning?",
    ],
)

if __name__ == "__main__":
    demo.launch(share=False)
