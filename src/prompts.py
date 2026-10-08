
ALLOWED_TOPICS = ["delivery", "price", "quality", "support", "usability"]
ALLOWED_SENTIMENT = ["positive", "neutral", "negative"]
ALLOWED_CONFIDENCE = ["low", "medium", "high"]

MINIMAL = "Analyze this review:\n{text}"

V3_FULL = """You are a review analyzer. Return a single JSON object and nothing else.

Fields:
- sentiment: one of positive, neutral, negative
- confidence: one of low, medium, high — how certain you are about the sentiment
- topics: array, any of delivery, price, quality, support, usability. May be empty.
- issue_count: integer, the number of distinct complaints mentioned

For example:
Review: "Arrived late and the screen was scratched."
{{"sentiment": "negative", "confidence": "high", "topics": ["delivery", "quality"], "issue_count": 2}}

Review: "Works as described."
{{"sentiment": "positive", "confidence": "medium", "topics": ["quality"], "issue_count": 0}}

Output only the JSON object. No markdown fences, no explanation.

Analyze this review:
{text}"""

LABELING = V3_FULL



import json
from pathlib import Path

FEWSHOT_HEADER = """You are a review analyzer. Return a single JSON object and nothing else.

Fields:
- sentiment: exactly one of positive, neutral, negative
- confidence: exactly one of low, medium, high
- topics: array, only from delivery, price, quality, support, usability. Never invent other values. May be empty.
- issue_count: integer, the number of distinct complaints

The examples below show how these fields are assigned. Follow their conventions.

"""


def build_fewshot(path: str = "data/fewshot.jsonl") -> str:
    parts = [FEWSHOT_HEADER]
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        parts.append("Review: " + row["text"])
        dumped = json.dumps(row["gold"], ensure_ascii=False)
        parts.append(dumped.replace("{", "{{").replace("}", "}}") + "\n")
    parts.append("Review: {text}")
    return "\n".join(parts)