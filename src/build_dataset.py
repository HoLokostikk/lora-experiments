import json
from src.prompts import label_from_stars

raw = json.load(open("data/raw_texts.json"))

for split, n in [("test", 60), ("train", 400), ("val", 60)]:
    items = raw[split][:n]
    with open(f"data/{split}.jsonl", "w") as f:
        for it in items:
            gold = label_from_stars(it["text"], it["stars"])
            f.write(json.dumps({"text": it["text"], "gold": gold}, ensure_ascii=False) + "\n")
    print(f"{split}: {len(items)}")

# перші 12 з train — для few-shot промпту
with open("data/fewshot.jsonl", "w") as f:
    for it in raw["train"][:12]:
        gold = label_from_stars(it["text"], it["stars"])
        f.write(json.dumps({"text": it["text"], "gold": gold}, ensure_ascii=False) + "\n")
print("fewshot: 12")