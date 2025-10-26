# app/chat_client.py
import requests, json, os

SERVER = "http://127.0.0.1:8000/generate"
LOG_PATH = "../data/logs.jsonl"   # run from app/ directory

def chat(prompt):
    payload = {"prompt": prompt}
    r = requests.post(SERVER, json=payload, timeout=60)
    r.raise_for_status()
    response = r.json()["response"]
    # append log (feedback initially empty)
    entry = {"prompt": prompt, "response": response, "feedback": ""}
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print("\n=== Assistant ===\n")
    print(response)
    print("\n(Logged entry to data/logs.jsonl)")

if __name__ == "__main__":
    print("Type your prompt. Empty line to quit.")
    while True:
        prompt = input("\nYou: ")
        if not prompt.strip():
            break
        try:
            chat(prompt)
        except Exception as e:
            print("Error:", e)
