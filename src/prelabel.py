import json, sys, httpx
from src.prompts import build_fewshot

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b-instruct-q4_K_M"
TEMPLATE = build_fewshot()


def generate(prompt: str) -> str:
    r = httpx.post(OLLAMA, json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "seed": 0},
    }, timeout=120)
    r.raise_for_status()
    return r.json()["response"]


def main(split: str, n: int, out_path: str):
    skip = 12 if split == "train" else 0
    texts = json.load(open("data/raw_texts.json"))[split][skip:skip + n]

    with open(out_path, "w") as f:
        for i, t in enumerate(texts, 1):
            try:
                o = json.loads(generate(TEMPLATE.format(text=t)).strip())
            except Exception:
                o = {}
            gold = {
                "sentiment": o.get("sentiment", "") if o.get("sentiment") in
                             ("positive", "neutral", "negative") else "",
                "confidence": "",                      # навмисно порожнє
                "topics": [x for x in o.get("topics", []) if isinstance(x, str)],
                "issue_count": o["issue_count"] if isinstance(o.get("issue_count"), int)
                               and not isinstance(o.get("issue_count"), bool) else 0,
            }
            f.write(json.dumps({"text": t, "gold": gold}, ensure_ascii=False) + "\n")
            print(f"\r{i}/{len(texts)}", end="", flush=True)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
