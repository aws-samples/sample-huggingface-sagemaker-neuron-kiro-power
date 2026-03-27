# Example: Fine-tune Qwen3 1.7B on Trainium

Fine-tune Qwen3 1.7B for instruction following using SageMaker AI training jobs on Trainium.

## Prerequisites

- AWS account with SageMaker AI access
- IAM role with SageMaker AI, S3, and ECR permissions
- Service quota for `ml.trn1.2xlarge`

## Training Script

Create `scripts/train.py`:

```python
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer
from optimum.neuron.models.training import NeuronModelForCausalLM


def format_instruction(example):
    instruction = f"### Instruction\n{example['instruction']}"
    context = f"### Context\n{example['context']}" if example.get("context") else None
    response = f"### Answer\n{example['response']}"
    parts = [instruction, context, response]
    return {"text": "\n\n".join(p for p in parts if p)}


def main():
    model_id = "Qwen/Qwen3-1.7B"

    # Load dataset
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    dataset = dataset.map(format_instruction)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = NeuronModelForCausalLM.from_pretrained(model_id)

    # Training config
    training_args = NeuronSFTConfig(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=10,
        save_total_limit=2,
        max_seq_length=512,
    )

    # Train
    trainer = NeuronSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model("./output/final")


if __name__ == "__main__":
    main()
```

## Launch on SageMaker AI

```python
import sagemaker
from sagemaker.huggingface import HuggingFace

role = sagemaker.get_execution_role()
session = sagemaker.Session()

estimator = HuggingFace(
    entry_point="train.py",
    source_dir="./scripts",
    instance_type="ml.trn1.2xlarge",
    instance_count=1,
    role=role,
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training-neuronx:2.8.0-transformers4.55.4-neuronx-py310-sdk2.26.0-ubuntu22.04",
    hyperparameters={
        "model_name_or_path": "Qwen/Qwen3-1.7B",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "learning_rate": 2e-5,
    },
    volume_size=512,
)

estimator.fit()
```

## Deploy the Fine-tuned Model

```python
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.inf2.xlarge",
)

result = predictor.predict({
    "inputs": "### Instruction\nExplain what a neural network is.\n\n### Answer\n",
    "parameters": {"max_new_tokens": 128},
})

print(result)
predictor.delete_endpoint()
```

## Migration to TorchNeuron (When Available)

When TorchNeuron reaches GA, the training script simplifies to:

```python
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# Standard HF — just change device
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B").to('neuron')

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    bf16=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

No `optimum-neuron` imports needed — just standard Transformers.
