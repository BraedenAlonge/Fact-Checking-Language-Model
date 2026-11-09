"""Evaluate base and Tinker checkpoints on a fixed, stratified task suite.

The script is inert unless --execute is supplied. It measures strict output
schema compliance and task-specific behavior over deterministic held-out rows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "claim_extraction",
    "search_planning",
    "evidence_selection",
    "claim_evaluation",
)
WORD = re.compile(r"\b[\w'-]+\b")
NUMBER = re.compile(r"\b\d[\d,.]*\b")


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def expected_output(row: dict) -> dict:
    return json.loads(row["messages"][2]["content"])


def user_input(row: dict):
    content = row["messages"][1]["content"]
    return content if row["task"] == "claim_extraction" else json.loads(content)


def stratum(row: dict) -> str:
    task = row["task"]
    source = row.get("source_dataset", "unknown")
    expected = expected_output(row)
    user = user_input(row)
    if task == "claim_extraction":
        behavior = "empty" if not expected.get("claims") else "claims"
    elif task == "search_planning":
        behavior = "numeric" if NUMBER.search(user.get("claim_text", "")) else "text"
    elif task == "evidence_selection":
        behavior = "multi" if len(user.get("search_results", [])) > 1 else "single"
    else:
        behavior = expected.get("label", "missing")
    return f"{source}|{behavior}"


def build_suite(rows: list[dict], per_task: int, seed: int) -> list[dict]:
    selected: list[dict] = []
    for task in TASKS:
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row["task"] == task:
                groups[stratum(row)].append(row)
        rng = random.Random(f"{seed}:{task}")
        keys = sorted(groups)
        rng.shuffle(keys)
        for key in keys:
            random.Random(f"{seed}:{task}:{key}").shuffle(groups[key])
        offsets = {key: 0 for key in keys}
        active = list(keys)
        task_rows: list[dict] = []
        while active and len(task_rows) < per_task:
            next_active = []
            for key in active:
                offset = offsets[key]
                if offset < len(groups[key]) and len(task_rows) < per_task:
                    task_rows.append(groups[key][offset])
                    offsets[key] += 1
                if offsets[key] < len(groups[key]):
                    next_active.append(key)
            active = next_active
        if len(task_rows) != per_task:
            raise ValueError(f"Only {len(task_rows)} {task} rows available")
        selected.extend(task_rows)
    return selected


def is_string(value) -> bool:
    return isinstance(value, str)


def strict_schema(task: str, value) -> bool:
    if not isinstance(value, dict):
        return False
    if task == "claim_extraction":
        if set(value) != {"claims"} or not isinstance(value["claims"], list):
            return False
        return all(
            isinstance(item, dict)
            and set(item) == {"claim_id", "claim_text", "context"}
            and all(is_string(item[key]) for key in item)
            for item in value["claims"]
        )
    if task == "search_planning":
        return (
            set(value) == {"claim_id", "tool", "arguments"}
            and is_string(value["claim_id"])
            and value["tool"] == "web_search"
            and isinstance(value["arguments"], dict)
            and set(value["arguments"]) == {"query"}
            and is_string(value["arguments"]["query"])
        )
    if task == "evidence_selection":
        required = {"claim_id", "evidence", "needs_refinement", "refinement_reason"}
        evidence_keys = {"source_id", "title", "url", "excerpt", "stance", "quality"}
        return (
            set(value) == required
            and is_string(value["claim_id"])
            and isinstance(value["needs_refinement"], bool)
            and (value["refinement_reason"] is None or is_string(value["refinement_reason"]))
            and isinstance(value["evidence"], list)
            and all(
                isinstance(item, dict)
                and set(item) == evidence_keys
                and all(is_string(item[key]) for key in evidence_keys)
                and item["stance"] in {"supports", "contradicts", "context"}
                and item["quality"] in {"high", "medium", "low"}
                for item in value["evidence"]
            )
        )
    required = {"claim_id", "label", "explanation", "source_ids"}
    return (
        set(value) == required
        and is_string(value["claim_id"])
        and value["label"] in {"Supported", "Contradicted", "Insufficient Evidence"}
        and is_string(value["explanation"])
        and isinstance(value["source_ids"], list)
        and all(is_string(item) for item in value["source_ids"])
    )


def normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def token_set(value: str) -> set[str]:
    return {token.casefold() for token in WORD.findall(value)}


def prf(expected: set[str], predicted: set[str]) -> tuple[int, int, int, float]:
    tp = len(expected & predicted)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    denominator = 2 * tp + fp + fn
    return tp, fp, fn, (2 * tp / denominator if denominator else 1.0)


def score_row(row: dict, prediction: str) -> dict:
    task = row["task"]
    expected = expected_output(row)
    user = user_input(row)
    result = {
        "valid_json": False,
        "strict_schema": False,
        "exact_json": False,
    }
    if task == "claim_extraction":
        expected_claims = {
            normalized_text(item.get("claim_text", ""))
            for item in expected.get("claims", [])
            if item.get("claim_text")
        }
        result.update(
            claim_tp=0,
            claim_fp=0,
            claim_fn=len(expected_claims),
            claim_f1=0.0 if expected_claims else 1.0,
            no_claim_expected=not expected_claims,
            no_claim_false_positive=False,
            no_claim_correct=False,
        )
    elif task == "search_planning":
        result.update(
            query_exact=False,
            query_token_f1=0.0,
            target_numbers_preserved=False,
            all_claim_numbers_preserved=False,
        )
    elif task == "evidence_selection":
        expected_ids = {
            item.get("source_id") for item in expected.get("evidence", []) if item.get("source_id")
        }
        result.update(
            evidence_tp=0,
            evidence_fp=0,
            evidence_fn=len(expected_ids),
            evidence_f1=0.0 if expected_ids else 1.0,
            copied_items=0,
            predicted_items=0,
            matched_items=0,
            stance_correct=0,
            quality_correct=0,
            refinement_correct=False,
        )
    else:
        result.update(
            label_correct=False,
            expected_label=expected.get("label"),
            predicted_label=None,
            source_id_f1=0.0,
        )
    try:
        actual = json.loads(prediction)
    except (json.JSONDecodeError, TypeError):
        return result
    result["valid_json"] = True
    result["strict_schema"] = strict_schema(task, actual)
    result["exact_json"] = actual == expected
    if not isinstance(actual, dict):
        return result

    if task == "claim_extraction":
        actual_claims = {
            normalized_text(item.get("claim_text", ""))
            for item in actual.get("claims", [])
            if isinstance(item, dict) and item.get("claim_text")
        }
        tp, fp, fn, f1 = prf(expected_claims, actual_claims)
        result.update(
            claim_tp=tp,
            claim_fp=fp,
            claim_fn=fn,
            claim_f1=f1,
            no_claim_expected=not expected_claims,
            no_claim_false_positive=not expected_claims and bool(actual_claims),
            no_claim_correct=not expected_claims and not actual_claims,
        )
    elif task == "search_planning":
        expected_query = expected.get("arguments", {}).get("query", "")
        actual_query = actual.get("arguments", {}).get("query", "")
        if not is_string(actual_query):
            actual_query = ""
        _, _, _, query_f1 = prf(token_set(expected_query), token_set(actual_query))
        expected_numbers = set(NUMBER.findall(expected_query))
        claim_numbers = set(NUMBER.findall(user.get("claim_text", "")))
        actual_numbers = set(NUMBER.findall(actual_query))
        result.update(
            query_exact=normalized_text(actual_query) == normalized_text(expected_query),
            query_token_f1=query_f1,
            target_numbers_preserved=not expected_numbers or expected_numbers.issubset(actual_numbers),
            all_claim_numbers_preserved=not claim_numbers or claim_numbers.issubset(actual_numbers),
        )
    elif task == "evidence_selection":
        expected_ids = {
            item.get("source_id") for item in expected.get("evidence", []) if item.get("source_id")
        }
        actual_items = [item for item in actual.get("evidence", []) if isinstance(item, dict)]
        actual_ids = {item.get("source_id") for item in actual_items if item.get("source_id")}
        tp, fp, fn, evidence_f1 = prf(expected_ids, actual_ids)
        inputs = {item.get("source_id"): item for item in user.get("search_results", [])}
        expected_by_id = {
            item.get("source_id"): item for item in expected.get("evidence", [])
        }
        copied = 0
        matched = 0
        stance_correct = 0
        quality_correct = 0
        for item in actual_items:
            source = inputs.get(item.get("source_id"), {})
            if source and all(item.get(field, "") == source.get(field, "") for field in ("title", "url", "excerpt")):
                copied += 1
            expected_item = expected_by_id.get(item.get("source_id"))
            if expected_item:
                matched += 1
                stance_correct += item.get("stance") == expected_item.get("stance")
                quality_correct += item.get("quality") == expected_item.get("quality")
        result.update(
            evidence_tp=tp,
            evidence_fp=fp,
            evidence_fn=fn,
            evidence_f1=evidence_f1,
            copied_items=copied,
            predicted_items=len(actual_items),
            matched_items=matched,
            stance_correct=stance_correct,
            quality_correct=quality_correct,
            refinement_correct=actual.get("needs_refinement") == expected.get("needs_refinement"),
        )
    else:
        expected_ids = set(expected.get("source_ids", []))
        actual_ids = set(actual.get("source_ids", [])) if isinstance(actual.get("source_ids"), list) else set()
        _, _, _, source_f1 = prf(expected_ids, actual_ids)
        result.update(
            label_correct=actual.get("label") == expected.get("label"),
            expected_label=expected.get("label"),
            predicted_label=actual.get("label"),
            source_id_f1=source_f1,
        )
    return result


def summarize(predictions: list[dict], suite: list[dict]) -> dict:
    summary: dict = {
        "records": len(predictions),
        "suite_by_task": dict(Counter(row["task"] for row in suite)),
        "suite_by_stratum": dict(sorted(Counter(f"{row['task']}:{stratum(row)}" for row in suite).items())),
        "models": {},
    }
    labels = sorted({item["model"] for item in predictions})
    for label in labels:
        model_rows = [item for item in predictions if item["model"] == label]
        task_summary = {}
        for task in TASKS:
            rows = [item for item in model_rows if item["task"] == task]
            scores = [item["scores"] for item in rows]
            base = {
                "records": len(rows),
                "valid_json_rate": mean(score["valid_json"] for score in scores),
                "strict_schema_rate": mean(score["strict_schema"] for score in scores),
                "exact_json_rate": mean(score["exact_json"] for score in scores),
            }
            if task == "claim_extraction":
                tp = sum(score.get("claim_tp", 0) for score in scores)
                fp = sum(score.get("claim_fp", 0) for score in scores)
                fn = sum(score.get("claim_fn", 0) for score in scores)
                no_claim = [score for score in scores if score.get("no_claim_expected")]
                base.update(
                    claim_micro_f1=(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0),
                    no_claim_examples=len(no_claim),
                    no_claim_false_positive_rate=(
                        mean(score["no_claim_false_positive"] for score in no_claim) if no_claim else 0.0
                    ),
                    no_claim_accuracy=(
                        mean(score["no_claim_correct"] for score in no_claim) if no_claim else 0.0
                    ),
                )
            elif task == "search_planning":
                base.update(
                    mean_query_token_f1=mean(score.get("query_token_f1", 0.0) for score in scores),
                    target_number_preservation_rate=mean(score.get("target_numbers_preserved", False) for score in scores),
                    all_claim_number_preservation_rate=mean(score.get("all_claim_numbers_preserved", False) for score in scores),
                )
            elif task == "evidence_selection":
                tp = sum(score.get("evidence_tp", 0) for score in scores)
                fp = sum(score.get("evidence_fp", 0) for score in scores)
                fn = sum(score.get("evidence_fn", 0) for score in scores)
                predicted = sum(score.get("predicted_items", 0) for score in scores)
                matched = sum(score.get("matched_items", 0) for score in scores)
                base.update(
                    evidence_micro_f1=(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0),
                    evidence_copy_rate=(sum(score.get("copied_items", 0) for score in scores) / predicted if predicted else 1.0),
                    stance_accuracy=(sum(score.get("stance_correct", 0) for score in scores) / matched if matched else 1.0),
                    quality_accuracy=(sum(score.get("quality_correct", 0) for score in scores) / matched if matched else 1.0),
                    refinement_accuracy=mean(score.get("refinement_correct", False) for score in scores),
                )
            else:
                verdicts = ("Supported", "Contradicted", "Insufficient Evidence")
                confusion = Counter(
                    (score.get("expected_label"), score.get("predicted_label") or "invalid")
                    for score in scores
                )
                label_f1 = []
                recalls = {}
                for verdict in verdicts:
                    tp = confusion[(verdict, verdict)]
                    fp = sum(count for (expected_label, predicted_label), count in confusion.items() if predicted_label == verdict and expected_label != verdict)
                    fn = sum(count for (expected_label, predicted_label), count in confusion.items() if expected_label == verdict and predicted_label != verdict)
                    denominator = 2 * tp + fp + fn
                    label_f1.append(2 * tp / denominator if denominator else 0.0)
                    recalls[verdict] = tp / (tp + fn) if tp + fn else 0.0
                base.update(
                    label_accuracy=mean(score.get("label_correct", False) for score in scores),
                    label_macro_f1=mean(label_f1),
                    label_recall=recalls,
                    label_confusion={
                        f"{expected_label} -> {predicted_label}": count
                        for (expected_label, predicted_label), count in sorted(confusion.items())
                    },
                    mean_source_id_f1=mean(score.get("source_id_f1", 0.0) for score in scores),
                )
            task_summary[task] = {
                key: (
                    round(value, 4)
                    if isinstance(value, float)
                    else {nested_key: round(nested_value, 4) for nested_key, nested_value in value.items()}
                    if key == "label_recall"
                    else value
                )
                for key, value in base.items()
            }
        summary["models"][label] = task_summary
    return summary


async def sample_one(client, renderer, messages: list[dict], max_tokens: int) -> str:
    prompt = renderer.build_generation_prompt(messages[:2])
    result = await client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            stop=renderer.get_stop_sequences(),
        ),
    )
    parsed, _ = renderer.parse_response(result.sequences[0].tokens)
    return parsed["content"]


async def run(args, suite: list[dict]) -> None:
    tokenizer = get_tokenizer(args.model)
    renderer = renderers.get_renderer(args.renderer, tokenizer, model_name=args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = []
    if args.include_base:
        models.append(("base", None))
    for item in args.checkpoint:
        if "=" not in item:
            raise ValueError("--checkpoint must be LABEL=tinker://PATH")
        models.append(tuple(item.split("=", 1)))
    if not models:
        raise ValueError("Select --include-base and/or at least one --checkpoint")

    predictions: list[dict] = []
    semaphore = asyncio.Semaphore(args.concurrency)
    with tinker.ServiceClient(display_name="background-checkers-stratified-eval") as service:
        clients = {}
        for label, checkpoint in models:
            clients[label] = await (
                service.create_sampling_client(base_model=args.model)
                if checkpoint is None
                else service.create_sampling_client(model_path=checkpoint)
            )

        async def evaluate(label: str, row: dict) -> dict:
            async with semaphore:
                try:
                    prediction = await sample_one(clients[label], renderer, row["messages"], args.max_tokens)
                    error = None
                except Exception as exc:  # Preserve partial evaluation evidence.
                    prediction = ""
                    error = f"{type(exc).__name__}: {exc}"
            return {
                "model": label,
                "task": row["task"],
                "example_id": row["example_id"],
                "source_dataset": row.get("source_dataset"),
                "stratum": stratum(row),
                "expected": row["messages"][2]["content"],
                "prediction": prediction,
                "error": error,
                "scores": score_row(row, prediction),
            }

        for label, _checkpoint in models:
            print(f"Sampling {label} on {len(suite)} examples", flush=True)
            predictions.extend(await asyncio.gather(*(evaluate(label, row) for row in suite)))

    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = summarize(predictions, suite)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def rescore(args, suite: list[dict]) -> None:
    by_id = {row["example_id"]: row for row in suite}
    predictions = []
    with args.rescore_predictions.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            row = by_id.get(item["example_id"])
            if row is None:
                raise ValueError(f"Prediction is not in this suite: {item['example_id']}")
            item["scores"] = score_row(row, item.get("prediction", ""))
            predictions.append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "predictions.jsonl"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = summarize(predictions, suite)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--renderer", default="qwen3_5_disable_thinking")
    parser.add_argument("--eval-file", type=Path, default=ROOT / "data/tinker/pilot_eval.jsonl")
    parser.add_argument("--per-task", type=int, default=24)
    parser.add_argument("--seed", type=int, default=5870)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rescore-predictions", type=Path)
    args = parser.parse_args()
    if min(args.per_task, args.max_tokens, args.concurrency) < 1:
        parser.error("per-task, max-tokens, and concurrency must be positive")
    suite = build_suite(load_rows(args.eval_file), args.per_task, args.seed)
    plan = {
        "execute": args.execute,
        "examples": len(suite),
        "per_task": args.per_task,
        "include_base": args.include_base,
        "checkpoints": args.checkpoint,
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.rescore_predictions:
        rescore(args, suite)
        return
    if not args.execute:
        print("Plan only; pass --execute to make paid Tinker sampling calls.")
        return
    asyncio.run(run(args, suite))


if __name__ == "__main__":
    main()
