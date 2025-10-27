from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json
import os

app = FastAPI(title="Personal Self-Evolving AI")

# Mount static folder for frontend
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

BASE_MODEL = "microsoft/phi-1_5"
ADAPTER_PATH = "../models/lora_adapter"
LOG_FILE = "data/logs.jsonl"

# Load tokenizer and model
tok = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
try:
    model.load_adapter(ADAPTER_PATH)
    print("✅ Loaded LoRA adapter.")
except Exception:
    print("⚠ Using base model only.")

gen = pipeline("text-generation", model=model, tokenizer=tok, device=0 if torch.cuda.is_available() else -1)

# Pydantic models
class Query(BaseModel):
    prompt: str
    response: str = ""
    feedback: str = ""

# Chat endpoint
@app.post("/chat/")
def chat(q: Query):
    prompt = q.prompt
    result = gen(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
    response = result[0]["generated_text"]
    # Log chat without feedback yet
    # entry = {"prompt": prompt, "response": response, "feedback": ""}
    # os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    # with open(LOG_FILE, "a", encoding="utf-8") as f:
    #     f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"response": response}

# Feedback endpoint
@app.post("/feedback/")
def feedback(q: Query):
    entry = {"prompt": q.prompt, "response": q.response, "feedback": q.feedback}
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"status": "ok"}
