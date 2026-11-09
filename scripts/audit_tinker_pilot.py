"""Audit the prepared Tinker pilot for schema and goal-alignment signals."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "train": ROOT / "data/tinker/pilot_train.jsonl",
    "eval": ROOT / "data/tinker/pilot_eval.jsonl",
}
WORD = re.compile(r"\b[\w'-]+\b")
NUMBER = re.compile(r"\b\d[\d,.]*\b")
PREDICTIVE = re.compile(
    r"\b(?:may|might|could|will|likely|probably|expected to|should)\b", re.IGNORECASE
)


def ratio(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def main() -> None:
    report: dict = {"files": {}, "cross_split": {}}
    fingerprints: dict[str, set[str]] = {}
    ids: dict[str, set[tuple[str, str, str]]] = {}

    for split, path in FILES.items():
        task_counts: Counter = Counter()
        sources: dict[str, Counter] = defaultdict(Counter)
        metrics: dict[str, Counter] = defaultdict(Counter)
        labels: Counter = Counter()
        stances: Counter = Counter()
        qualities: Counter = Counter()
        message_fingerprints: set[str] = set()
        source_ids: set[tuple[str, str, str]] = set()

        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                task = row["task"]
                task_counts[task] += 1
                sources[task][row.get("source_dataset", "unknown")] += 1
                source_ids.add(
                    (
                        task,
                        row.get("source_dataset", "unknown"),
                        str(row.get("source_id", "")),
                    )
                )
                message_fingerprints.add(
                    json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
                )

                assistant = json.loads(row["messages"][2]["content"])
                user = (
                    row["messages"][1]["content"]
                    if task == "claim_extraction"
                    else json.loads(row["messages"][1]["content"])
                )

                if task == "search_planning":
                    claim = user.get("claim_text", "")
                    query = assistant.get("arguments", {}).get("query", "")
                    query_words = WORD.findall(query)
                    claim_numbers = set(NUMBER.findall(claim))
                    query_numbers = set(NUMBER.findall(query))
                    metrics[task]["records"] += 1
                    metrics[task]["query_words_total"] += len(query_words)
                    metrics[task]["query_at_most_3_words"] += len(query_words) <= 3
                    metrics[task]["query_at_most_5_words"] += len(query_words) <= 5
                    metrics[task]["claims_with_numbers"] += bool(claim_numbers)
                    metrics[task]["number_not_preserved"] += bool(
                        claim_numbers and not claim_numbers.issubset(query_numbers)
                    )

                elif task == "claim_extraction":
                    claims = assistant.get("claims", [])
                    metrics[task]["records"] += 1
                    metrics[task]["empty_outputs"] += not claims
                    metrics[task]["claims_total"] += len(claims)
                    for claim in claims:
                        text = claim.get("claim_text", "")
                        context = claim.get("context", "")
                        metrics[task]["claims_with_predictive_modal"] += bool(
                            PREDICTIVE.search(text)
                        )
                        metrics[task]["claims_with_newline"] += "\n" in text
                        metrics[task]["contexts_with_source_instruction"] += (
                            "summarize the following" in context.lower()
                        )

                elif task == "evidence_selection":
                    results = user.get("search_results", [])
                    selected = assistant.get("evidence", [])
                    by_source = {item.get("source_id"): item for item in results}
                    metrics[task]["records"] += 1
                    metrics[task]["input_results_total"] += len(results)
                    metrics[task]["selected_total"] += len(selected)
                    metrics[task]["single_input_result"] += len(results) == 1
                    metrics[task]["single_selected_result"] += len(selected) == 1
                    metrics[task]["needs_refinement"] += bool(
                        assistant.get("needs_refinement")
                    )
                    for item in selected:
                        metrics[task]["blank_selected_url"] += not item.get("url")
                        original = by_source.get(item.get("source_id"), {})
                        copied = all(
                            item.get(field, "") == original.get(field, "")
                            for field in ("title", "url", "excerpt")
                        )
                        metrics[task]["copy_mismatch"] += not copied
                        stances[item.get("stance", "missing")] += 1
                        qualities[item.get("quality", "missing")] += 1

                elif task == "claim_evaluation":
                    evidence = user.get("evidence", [])
                    explanation = assistant.get("explanation", "")
                    metrics[task]["records"] += 1
                    metrics[task]["evidence_total"] += len(evidence)
                    metrics[task]["single_evidence"] += len(evidence) == 1
                    metrics[task]["empty_evidence"] += not evidence
                    metrics[task]["templated_explanation"] += any(
                        phrase in explanation
                        for phrase in (
                            "The supplied evidence supports the complete claim.",
                            "The supplied evidence directly conflicts with the claim.",
                            "The supplied evidence does not settle the complete claim.",
                        )
                    )
                    labels[assistant.get("label", "missing")] += 1

        search = metrics["search_planning"]
        extraction = metrics["claim_extraction"]
        evidence = metrics["evidence_selection"]
        evaluation = metrics["claim_evaluation"]
        task_report = {
            "search_planning": {
                **search,
                "mean_query_words": round(
                    search["query_words_total"] / search["records"], 2
                ),
                "at_most_3_word_rate": ratio(
                    search["query_at_most_3_words"], search["records"]
                ),
                "number_omission_rate_when_present": ratio(
                    search["number_not_preserved"], search["claims_with_numbers"]
                ),
            },
            "claim_extraction": {
                **extraction,
                "predictive_modal_claim_rate": ratio(
                    extraction["claims_with_predictive_modal"], extraction["claims_total"]
                ),
                "newline_claim_rate": ratio(
                    extraction["claims_with_newline"], extraction["claims_total"]
                ),
                "instruction_context_rate": ratio(
                    extraction["contexts_with_source_instruction"],
                    extraction["claims_total"],
                ),
            },
            "evidence_selection": {
                **evidence,
                "single_input_rate": ratio(
                    evidence["single_input_result"], evidence["records"]
                ),
                "single_selected_rate": ratio(
                    evidence["single_selected_result"], evidence["records"]
                ),
                "blank_url_rate": ratio(
                    evidence["blank_selected_url"], evidence["selected_total"]
                ),
                "copy_mismatch_rate": ratio(
                    evidence["copy_mismatch"], evidence["selected_total"]
                ),
                "stances": dict(stances),
                "qualities": dict(qualities),
            },
            "claim_evaluation": {
                **evaluation,
                "single_evidence_rate": ratio(
                    evaluation["single_evidence"], evaluation["records"]
                ),
                "templated_explanation_rate": ratio(
                    evaluation["templated_explanation"], evaluation["records"]
                ),
                "labels": dict(labels),
            },
        }
        report["files"][split] = {
            "records": sum(task_counts.values()),
            "task_counts": dict(task_counts),
            "source_counts_by_task": {
                task: dict(counter) for task, counter in sorted(sources.items())
            },
            "metrics": task_report,
        }
        fingerprints[split] = message_fingerprints
        ids[split] = source_ids

    report["cross_split"] = {
        "exact_message_overlap": len(fingerprints["train"] & fingerprints["eval"]),
        "task_dataset_original_id_overlap": len(ids["train"] & ids["eval"]),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
