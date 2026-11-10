"""Strict runtime validation for the four model task interfaces."""

from __future__ import annotations

import json
from typing import Any, Callable


VERDICTS = {"Supported", "Contradicted", "Insufficient Evidence"}
STANCES = {"supports", "contradicts", "context"}
QUALITIES = {"high", "medium", "low"}


class SchemaError(ValueError):
    """Raised when a model response does not satisfy its runtime contract."""


def loads_object(text: str) -> dict[str, Any]:
    """Parse one JSON object, tolerating a single Markdown JSON fence."""

    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].strip().casefold() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"response is not valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise SchemaError("response must be a JSON object")
    return parsed


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SchemaError(f"{name} keys mismatch; missing={missing}, extra={extra}")


def _string(value: Any, name: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{name} must be a string")
    result = value.strip()
    if nonempty and not result:
        raise SchemaError(f"{name} cannot be empty")
    return result


def validate_claim_extraction(value: dict[str, Any]) -> dict[str, Any]:
    _exact_keys(value, {"claims"}, "claim extraction")
    claims = value["claims"]
    if not isinstance(claims, list):
        raise SchemaError("claims must be a list")
    seen: set[str] = set()
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise SchemaError(f"claims[{index}] must be an object")
        _exact_keys(claim, {"claim_id", "claim_text", "context"}, f"claims[{index}]")
        claim_id = _string(claim["claim_id"], f"claims[{index}].claim_id")
        if claim_id in seen:
            raise SchemaError(f"duplicate claim_id: {claim_id}")
        seen.add(claim_id)
        claim["claim_text"] = _string(claim["claim_text"], f"claims[{index}].claim_text")
        claim["context"] = _string(claim["context"], f"claims[{index}].context", nonempty=False)
    return value


def validate_search_planning(value: dict[str, Any]) -> dict[str, Any]:
    _exact_keys(value, {"claim_id", "tool", "arguments"}, "search planning")
    _string(value["claim_id"], "claim_id")
    if value["tool"] != "web_search":
        raise SchemaError("tool must be web_search")
    arguments = value["arguments"]
    if not isinstance(arguments, dict):
        raise SchemaError("arguments must be an object")
    _exact_keys(arguments, {"query"}, "arguments")
    arguments["query"] = _string(arguments["query"], "arguments.query")
    return value


def validate_evidence_selection(value: dict[str, Any]) -> dict[str, Any]:
    _exact_keys(
        value,
        {"claim_id", "evidence", "needs_refinement", "refinement_reason"},
        "evidence selection",
    )
    _string(value["claim_id"], "claim_id")
    if not isinstance(value["needs_refinement"], bool):
        raise SchemaError("needs_refinement must be a boolean")
    reason = value["refinement_reason"]
    if reason is not None and not isinstance(reason, str):
        raise SchemaError("refinement_reason must be a string or null")
    evidence = value["evidence"]
    if not isinstance(evidence, list):
        raise SchemaError("evidence must be a list")
    expected = {"source_id", "title", "url", "excerpt", "stance", "quality"}
    seen: set[str] = set()
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise SchemaError(f"evidence[{index}] must be an object")
        _exact_keys(item, expected, f"evidence[{index}]")
        for key in ("source_id", "title", "url", "excerpt"):
            item[key] = _string(
                item[key], f"evidence[{index}].{key}", nonempty=key == "source_id"
            )
        if item["source_id"] in seen:
            raise SchemaError(f"duplicate evidence source_id: {item['source_id']}")
        seen.add(item["source_id"])
        if item["stance"] not in STANCES:
            raise SchemaError(f"invalid evidence stance: {item['stance']!r}")
        if item["quality"] not in QUALITIES:
            raise SchemaError(f"invalid evidence quality: {item['quality']!r}")
    return value


def validate_claim_evaluation(value: dict[str, Any]) -> dict[str, Any]:
    _exact_keys(value, {"claim_id", "label", "explanation", "source_ids"}, "claim evaluation")
    _string(value["claim_id"], "claim_id")
    if value["label"] not in VERDICTS:
        raise SchemaError(f"invalid verdict: {value['label']!r}")
    value["explanation"] = _string(value["explanation"], "explanation")
    if not isinstance(value["source_ids"], list) or not all(
        isinstance(item, str) and item.strip() for item in value["source_ids"]
    ):
        raise SchemaError("source_ids must be a list of nonempty strings")
    if len(set(value["source_ids"])) != len(value["source_ids"]):
        raise SchemaError("source_ids cannot contain duplicates")
    return value


Validator = Callable[[dict[str, Any]], dict[str, Any]]
