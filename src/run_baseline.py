
import json, sys, httpx
from src.prompts import MINIMAL, V3_FULL
from src.eval_json import evaluate

OLLAMA = "http://localhost:11434/api/generate"


def generate(model: str, prompt: str) -> str:
    r = httpx.post(OLLAMA, json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "seed": 0},
    }, timeout=120)
    r.raise_for_status()
    return r.json()["response"]


def main(model: str, template_name: str):
    from src.prompts import build_fewshot
    template = {"minimal": MINIMAL, "v3": V3_FULL, "fewshot": build_fewshot()}[template_name]
    rows = [json.loads(l) for l in open("data/test.jsonl")]

    preds, golds = [], []
    for i, row in enumerate(rows, 1):
        preds.append(generate(model, template.format(text=row["text"])))
        golds.append(row["gold"])
        print(f"\r{i}/{len(rows)}", end="", flush=True)
    print()

    metrics = evaluate(preds, golds)
    print(json.dumps(metrics, indent=2))

    tag = f"{model.replace(':', '_')}__{template_name}"
    with open(f"outputs/baseline_{tag}.json", "w") as f:
        json.dump({"model": model, "template": template_name,
                   "metrics": metrics, "raw": preds}, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])