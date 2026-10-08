import json
from pathlib import Path

ALLOWED_TOPICS = ["delivery", "price", "quality", "support", "usability"]
ALLOWED_SENTIMENT = ["positive", "neutral", "negative"]

TOPIC_LEXICON = {
    "delivery": ["ship", "shipped", "shipping", "arrive", "arrived", "delivery",
                 "delivered", "package", "packaging", "box", "late", "fast"],
    "price": ["price", "priced", "cheap", "expensive", "cost", "worth",
              "value", "money", "overpriced", "bargain", "deal"],
    "quality": ["quality", "material", "fabric", "sturdy", "flimsy", "broke",
                "broken", "durable", "cheap-feeling", "well-made", "defect"],
    "support": ["support", "service", "customer", "refund", "return",
                "replacement", "warranty", "seller", "contacted", "response"],
    "usability": ["easy", "difficult", "hard", "confusing", "instructions",
                  "simple", "intuitive", "setup", "install", "use", "using",
                  "works", "fit", "fits"],
}

MINIMAL = "Analyze this review:\n{text}"

V3_FULL = """You are a review analyzer. Return a single JSON object and nothing else.

Fields:
- stars: integer 1 to 5, the rating the reviewer most likely gave
- sentiment: one of positive, neutral, negative
- topics: array, any of delivery, price, quality, support, usability. May be empty.
- recommend: boolean, whether the reviewer would recommend the product

For example:
Review: "Arrived late and the screen was scratched. Returned it."
{{"stars": 1, "sentiment": "negative", "topics": ["delivery", "quality", "support"], "recommend": false}}

Review: "Works as described, nothing special but no complaints."
{{"stars": 4, "sentiment": "positive", "topics": ["quality"], "recommend": true}}

Output only the JSON object. No markdown fences, no explanation.

Analyze this review:
{text}"""


def label_from_stars(text: str, stars: int) -> dict:
    low = text.lower()
    topics = [t for t, words in TOPIC_LEXICON.items()
              if any(w in low for w in words)]
    return {
        "stars": stars,
        "sentiment": "negative" if stars <= 2 else ("neutral" if stars == 3 else "positive"),
        "topics": topics,
        "recommend": stars >= 4,
    }


def build_fewshot(path: str = "data/fewshot.jsonl") -> str:
    parts = ["""You are a review analyzer. Return a single JSON object and nothing else.

Fields:
- stars: integer 1 to 5
- sentiment: exactly one of positive, neutral, negative
- topics: array, only from delivery, price, quality, support, usability. Never invent other values.
- recommend: boolean

The examples below show how these fields are assigned. Follow their conventions.
"""]
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        parts.append("Review: " + row["text"])
        dumped = json.dumps(row["gold"], ensure_ascii=False)
        parts.append(dumped.replace("{", "{{").replace("}", "}}") + "\n")
    parts.append("Review: {text}")
    return "\n".join(parts)