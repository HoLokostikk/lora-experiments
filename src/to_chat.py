import json
from src.prompts import MINIMAL

for split in ["train", "val"]:
    rows = [json.loads(l) for l in open(f"data/{split}.jsonl")]
    with open(f"data/{split}_chat.jsonl", "w") as f:
        for r in rows:
            msg = {"messages": [
                {"role": "user", "content": MINIMAL.format(text=r["text"])},
                {"role": "assistant", "content": json.dumps(r["gold"], ensure_ascii=False)},
            ]}
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    print(f"{split}_chat.jsonl: {len(rows)}")