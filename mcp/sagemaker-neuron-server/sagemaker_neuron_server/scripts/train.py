
"""Optimum Neuron training script for fine-tuning on AWS Trainium.
Follows the official HF tutorial exactly:
https://huggingface.co/docs/optimum-neuron/training_tutorials/finetune_qwen3
"""
import json
import os
import torch
from dataclasses import dataclass, field
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, HfArgumentParser
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM

# Dataset — using wikitext for demo; override via --dataset_name / --dataset_config
def load_and_prepare_dataset(tokenizer, dataset_name, dataset_config):
    dataset = load_dataset(dataset_name, dataset_config, split="train[:2000]")
    dataset = dataset.filter(lambda x: x["text"] is not None and len(x["text"].strip()) > 10)
    eos = tokenizer.eos_token

    def preprocess(examples):
        chats = []
        for text in examples["text"]:
            chat = [
                {"role": "user", "content": "Continue the following text:"},
                {"role": "assistant", "content": text + eos},
            ]
            chats.append(chat)
        return {"messages": chats}

    dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    return dataset

def train(model_id, tokenizer, dataset, training_args):
    trn_config = training_args.trn_config
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = NeuronModelForCausalLM.from_pretrained(
        model_id,
        trn_config,
        torch_dtype=dtype,
    )

    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "o_proj", "k_proj", "up_proj", "down_proj", "gate_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = training_args.to_dict()
    sft_config = NeuronSFTConfig(
        max_length=256,
        packing=True,
        **args,
    )

    def formatting_function(examples):
        return tokenizer.apply_chat_template(examples["messages"], tokenize=False, add_generation_prompt=False)

    trainer = NeuronSFTTrainer(
        args=sft_config,
        model=model,
        peft_config=lora_config,
        processing_class=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_function,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

    # Consolidate Neuron-sharded adapter into standard adapter_model.safetensors
    import subprocess, sys
    adapter_dir = os.path.join(training_args.output_dir, "adapter_default")
    if os.path.isdir(adapter_dir):
        print(f"=== Consolidating adapter shards: {adapter_dir} ===", flush=True)
        subprocess.run([sys.executable, "-m", "optimum.commands.optimum_cli",  # nosemgrep: dangerous-subprocess-use-audit
                        "neuron", "consolidate", training_args.output_dir, adapter_dir], check=True)
        # Merge adapter into base model and save at output_dir root for direct deployment
        from transformers import AutoModelForCausalLM
        from peft import PeftModel, PeftConfig
        base = AutoModelForCausalLM.from_pretrained(script_args.model_id)
        config = PeftConfig.from_pretrained(adapter_dir)
        model = PeftModel.from_pretrained(base, adapter_dir, config=config)
        model = model.merge_and_unload()
        # Ensure torch_dtype is set in config (needed by Neuron inference toolkit)
        model.config.torch_dtype = torch.bfloat16
        model.save_pretrained(training_args.output_dir)
        # Force torch_dtype in saved config.json (some models don't serialize it correctly)
        config_path = os.path.join(training_args.output_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["torch_dtype"] = "bfloat16"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        AutoTokenizer.from_pretrained(script_args.model_id).save_pretrained(training_args.output_dir)
        # Add inference.py to fix Neuron HLO write path in read-only containers
        code_dir = os.path.join(training_args.output_dir, "code")
        os.makedirs(code_dir, exist_ok=True)
        with open(os.path.join(code_dir, "inference.py"), "w", encoding="utf-8") as f:
            f.write("import os\nos.chdir('/tmp')\n")
        print(f"=== Merged model saved to {training_args.output_dir} ===", flush=True)

@dataclass
class ScriptArguments:
    model_id: str = field(default="Qwen/Qwen3-0.6B", metadata={"help": "HF model ID"})
    dataset_name: str = field(default="wikitext", metadata={"help": "HF dataset name"})
    dataset_config: str = field(default="wikitext-2-raw-v1", metadata={"help": "HF dataset config"})

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, NeuronTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    # Override output_dir to SageMaker model dir
    training_args.output_dir = os.environ.get("SM_MODEL_DIR", training_args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    dataset = load_and_prepare_dataset(tokenizer, script_args.dataset_name, script_args.dataset_config)
    train(model_id=script_args.model_id, tokenizer=tokenizer, dataset=dataset, training_args=training_args)
