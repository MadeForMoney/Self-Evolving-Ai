# 🚀 Fine-Tuning Microsoft Phi Model with LoRA using PEFT

This repository demonstrates how to fine-tune the **Microsoft Phi model** using **LoRA (Low-Rank Adaptation)** with **PEFT (Parameter-Efficient Fine-Tuning)**. The project includes training, inference, and logging utilities, wrapped into an easy-to-run modular setup.

---

## 📂 Project Structure

├── app/ # Scripts for training, inference, and evaluation
│ ├── train.py
│ ├── inference.py
│ └── utils.py
│
├── data/ # Logs, datasets, and results
│ └── logs/ # Training logs and metrics
│
├── models/ # Base + fine-tuned model weights stored here
│ ├── base_model/ # Pretrained Microsoft Phi model
│ └── lora_adapter/ # LoRA adapter weights after fine-tuning
│
├── requirements.txt # Python dependencies
│
└── README.md # Project documentation



---

## 🧠 Overview

LoRA fine-tuning allows you to **train only a small subset of parameters (low-rank adapters)** while keeping the base model frozen.  
This makes training faster, cheaper, and highly memory efficient — perfect for running on GPUs with limited VRAM.

Here’s how it works conceptually:

1. The **base model (Microsoft Phi)** is loaded in frozen mode (no gradient updates).
2. LoRA layers (trainable matrices) are **inserted** into attention or projection layers.
3. Training updates **only LoRA parameters**, not the full model.
4. The resulting **LoRA adapter weights** are saved separately and can be merged later.

---

## ⚙️ Setup Instructions

### 1️⃣ Create Environment

bash
python -m venv venv
source venv/bin/activate      # (Linux/Mac)
venv\Scripts\activate         # (Windows)
2️⃣ Install Dependencies

pip install -r requirements.txt

🏋️ Training the Model
To fine-tune the Microsoft Phi model using LoRA:

python app/train.py
This will:

Load the base model from models/base_model/

Attach LoRA adapters using PEFT

Train only the adapter layers

Save LoRA weights into models/lora_adapter/

Log training progress into data/logs/

Example training code snippet (from train.py):

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-1_5"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
🧪 Inference (Using the Fine-Tuned Model)
To run inference using the LoRA-fine-tuned adapter:


python app/inference.py
This script automatically merges the LoRA adapter with the base model to produce output using the updated weights.

Example:

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
model = PeftModel.from_pretrained(base_model, "models/lora_adapter/")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

prompt = "Explain quantum entanglement in simple terms."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
📊 Logs & Results
All training logs and evaluation metrics are saved under:

bash

data/logs/
You can visualize training curves or losses using tensorboard or by manually plotting from logs.

🧩 Model Saving and Loading
After training, your folder will look like:

bash

models/
├── base_model/         # Original frozen Microsoft Phi weights
└── lora_adapter/       # Fine-tuned adapter weights
To merge LoRA weights with the base model for standalone deployment:

python

from peft import PeftModel

merged_model = PeftModel.from_pretrained(base_model, "models/lora_adapter/")
merged_model.merge_and_unload()
merged_model.save_pretrained("models/final_phi_finetuned/")
🧰 Notes
LoRA fine-tuning doesn’t overwrite base model weights — only adapters are modified.

You can train multiple LoRA adapters for different tasks and swap them in/out.

PEFT provides a flexible and efficient way to train very large models even on modest GPUs.

🧑‍💻 Author
Developed by Nithilan M
B.Tech Artificial Intelligence & Data Science

🪪 License
This project is released under the MIT License.







