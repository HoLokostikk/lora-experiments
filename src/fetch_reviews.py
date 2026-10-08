from datasets import load_dataset
from collections import defaultdict
import json, random

DATASET = "SetFit/amazon_reviews_multi_en"
TEXT_FIELD = "text"
LABEL_FIELD = "label"
PER_STAR = 120
EXPECTED_CLASSES = 5

ds = load_dataset(DATASET, split="test", streaming=True)

buckets, seen = defaultdict(list), set()
for row in ds:
    t = row[TEXT_FIELD].strip().replace("\n", " ")
    star = row[LABEL_FIELD]
    if not (40 < len(t) < 500) or t[:30] in seen:
        continue
    if len(buckets[star]) >= PER_STAR:
        if len(buckets) == EXPECTED_CLASSES and all(len(b) >= PER_STAR for b in buckets.values()):
            break
        continue
    seen.add(t[:30])
    buckets[star].append(t)

texts = [t for b in buckets.values() for t in b]
random.seed(0)
random.shuffle(texts)

with open("data/raw_texts.json", "w") as f:
    json.dump({"test": texts[:60], "train": texts[60:520], "val": texts[520:580]},
              f, indent=2, ensure_ascii=False)

print({k: len(v) for k, v in sorted(buckets.items())}, "→", len(texts))