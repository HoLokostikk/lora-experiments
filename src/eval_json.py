
import json
from src.prompts import ALLOWED_TOPICS, ALLOWED_SENTIMENT, ALLOWED_CONFIDENCE


def parse_strict(raw: str):
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return None


def check_schema(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != {"sentiment", "confidence", "topics", "issue_count"}:
        return False
    return (
        isinstance(obj["sentiment"], str)
        and isinstance(obj["confidence"], str)
        and isinstance(obj["topics"], list)
        and isinstance(obj["issue_count"], int)
        and not isinstance(obj["issue_count"], bool)
    )


def check_enums(obj) -> bool:
    return (
        obj["sentiment"] in ALLOWED_SENTIMENT
        and obj["confidence"] in ALLOWED_CONFIDENCE
        and all(t in ALLOWED_TOPICS for t in obj["topics"])
    )


def check_exact(obj, gold) -> bool:
    return (
        obj["sentiment"] == gold["sentiment"]
        and obj["confidence"] == gold["confidence"]
        and sorted(obj["topics"]) == sorted(gold["topics"])
        and obj["issue_count"] == gold["issue_count"]
    )


def evaluate(predictions: list[str], golds: list[dict]) -> dict:
    n = len(predictions)
    valid, schema, enums, exact = 0, 0, 0, 0
    per_field = {"sentiment": 0, "confidence": 0, "topics": 0, "issue_count": 0}

    for raw, gold in zip(predictions, golds):
        obj = parse_strict(raw)
        if obj is None:
            continue
        valid += 1
        if not check_schema(obj):
            continue
        schema += 1
        if not check_enums(obj):
            continue
        enums += 1
        for f in per_field:
            a, b = obj[f], gold[f]
            if f == "topics":
                a, b = sorted(a), sorted(b)
            if a == b:
                per_field[f] += 1
        if check_exact(obj, gold):
            exact += 1

    return {
        "json_valid": round(valid / n, 3),
        "schema_valid": round(schema / n, 3),
        "enum_valid": round(enums / n, 3),
        "exact": round(exact / n, 3),
        "per_field": {k: round(v / n, 3) for k, v in per_field.items()},
    }