# app/retrain.py
import json
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# Paths
LOGS_PATH = "../data/logs.jsonl"
ADAPTER_SAVE = "../models/lora_adapter"
BASE_MODEL = "microsoft/phi-1_5"  # base model

# Load feedback logs (only positive feedback)
def load_feedback_data():
    samples = []
    if not os.path.exists(LOGS_PATH):
        print(f"⚠ No logs found at {LOGS_PATH}")
        return Dataset.from_list(samples)

    with open(LOGS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("feedback") == "good":
                samples.append({"text": f"User: {d['prompt']}\nAssistant: {d['response']}"})

    print(f"✅ Loaded {len(samples)} positive feedback samples")
    return Dataset.from_list(samples)

# Tokenization
def tokenize_fn(example, tokenizer):
    out = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    out["labels"] = out["input_ids"].copy()
    return out

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_feedback_data()
    if len(dataset) == 0:
        print("⚠ No positive feedback to train on. Exiting.")
        return

    dataset = dataset.map(lambda e: tokenize_fn(e, tokenizer))

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )

    # Apply LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # Training arguments
    args = TrainingArguments(
        output_dir="../models/checkpoints",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=5,
        save_total_limit=1,
        fp16=torch.cuda.is_available()
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training finished!")

    # Ensure folder exists and save LoRA adapter
    os.makedirs(ADAPTER_SAVE, exist_ok=True)
    model.save_pretrained(ADAPTER_SAVE)
    tokenizer.save_pretrained(ADAPTER_SAVE)
    print(f"✅ Saved LoRA adapter to {ADAPTER_SAVE}")
    print("📂 Contents:")
    print(os.listdir(ADAPTER_SAVE))

if __name__ == "__main__":
    main()
