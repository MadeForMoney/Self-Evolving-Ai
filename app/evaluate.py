# app/evaluate.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, json

BASE_MODEL = "microsoft/phi-1_5"
ADAPTER_PATH = "../models/lora_adapter"
EVAL_QUERIES = [
    "What is LoRA?",
    "Explain transformers in one line.",
    "How does fine-tuning differ from pretraining?"
]

def main():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    try:
        model.load_adapter(ADAPTER_PATH)
    except Exception:
        print("⚠ No adapter found, evaluating base model only.")
    gen = pipeline("text-generation", model=model, tokenizer=tok, device=0 if torch.cuda.is_available() else -1)
    scores = []
    for q in EVAL_QUERIES:
        out = gen(q, max_new_tokens=60)[0]["generated_text"]
        print(f"\n🧩 Prompt: {q}\n→ {out}\n")
        scores.append(len(out))  # trivial metric
    print(f"Average output length: {sum(scores)/len(scores):.1f}")

if __name__ == "__main__":
    main()
