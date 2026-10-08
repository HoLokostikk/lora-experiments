import json, sys

PATH = sys.argv[1]
TOPICS = ["delivery", "price", "quality", "support", "usability"]
SENTIMENT = {"1": "positive", "2": "neutral", "3": "negative"}
CONFIDENCE = {"1": "low", "2": "medium", "3": "high"}

rows = [json.loads(l) for l in open(PATH)]


def save():
    with open(PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ask(label, mapping, suggested=None):
    opts = "  ".join(f"{k}={v}" for k, v in mapping.items())
    hint = f" [Enter={suggested}]" if suggested else ""
    while True:
        v = input(f"  {label} ({opts}){hint}: ").strip()
        if not v and suggested:
            return suggested
        if v in mapping:
            return mapping[v]
        print("    цифра або Enter")


done = sum(1 for r in rows if r["gold"]["confidence"])
for i, row in enumerate(rows):
    g = row["gold"]
    if g["confidence"]:
        continue
    print("\n" + "=" * 70)
    print(f"[{i + 1}/{len(rows)}]  розмічено: {done}")
    print(row["text"])
    print("=" * 70)

    try:
        s = ask("sentiment", SENTIMENT, g["sentiment"] or None)
        c = ask("confidence", CONFIDENCE)          # без підказки — модель тут не вміє

        sug = "".join(t[0] for t in TOPICS if t in g["topics"])
        print("  topics: " + "  ".join(f"{t[0]}={t}" for t in TOPICS))
        raw = input(f"  (літери; Enter={sug or 'порожньо'}; '-' = очистити): ").strip().lower()
        if raw == "-":
            picked = []
        elif not raw:
            picked = [t for t in TOPICS if t in g["topics"]]
        else:
            picked = [t for t in TOPICS if t[0] in raw.split()]

        n = input(f"  issue_count [Enter={g['issue_count']}]: ").strip()
        row["gold"] = {"sentiment": s, "confidence": c,
                       "topics": picked, "issue_count": int(n) if n else g["issue_count"]}
        done += 1
        save()
    except (KeyboardInterrupt, EOFError):
        save()
        print(f"\n\nзбережено, розмічено {done}/{len(rows)}")
        sys.exit(0)

save()
print(f"\nготово: {done}/{len(rows)}")
