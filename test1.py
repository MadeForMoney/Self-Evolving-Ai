# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from peft import PeftModel
# import torch

# BASE_MODEL = "microsoft/phi-1_5"
# ADAPTER_MODEL = "nps798/phi-1_5-qlora-alpaca-instruction"  # Public QLoRA adapter

# device = 0 if torch.cuda.is_available() else -1
# dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     device_map="auto" if torch.cuda.is_available() else None,
#     dtype=dtype
# )

# # Load QLoRA adapter
# model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
# print("✅ QLoRA adapter loaded successfully!")

# gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

# prompt = "Hello AI, how are you today?"
# result = gen(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
# print(result[0]["generated_text"])

import os
print(os.listdir("../models/lora_adapter"))

