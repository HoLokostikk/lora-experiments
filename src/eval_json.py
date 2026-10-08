
import json
from src.prompts import ALLOWED_TOPICS, ALLOWED_SENTIMENT


def parse_strict(raw: str):
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return None


def check_schema(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != {"stars", "sentiment", "topics", "recommend"}:
        return False
    return (
        isinstance(obj["stars"], int) and not isinstance(obj["stars"], bool)
        and isinstance(obj["sentiment"], str)
        and isinstance(obj["topics"], list)
        and isinstance(obj["recommend"], bool)
    )


def check_enums(obj) -> bool:
    return (
        1 <= obj["stars"] <= 5
        and obj["sentiment"] in ALLOWED_SENTIMENT
        and all(t in ALLOWED_TOPICS for t in obj["topics"])
    )


def check_exact(obj, gold) -> bool:
    return (
        obj["stars"] == gold["stars"]
        and obj["sentiment"] == gold["sentiment"]
        and sorted(obj["topics"]) == sorted(gold["topics"])
        and obj["recommend"] == gold["recommend"]
    )


def evaluate(predictions: list[str], golds: list[dict]) -> dict:
    n = len(predictions)
    valid, schema, enums, exact = 0, 0, 0, 0
    per_field = {"stars": 0, "sentiment": 0, "topics": 0, "recommend": 0}

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