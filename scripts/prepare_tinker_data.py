"""Create deterministic, task-balanced Tinker pilot datasets.

The complete converted corpus is intentionally much larger than a sensible
first training run.  This script reservoir-samples each task independently,
keeps the repository's train and eval splits separate, and writes ordinary
conversation JSONL that tinker-cookbook can consume directly.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
DEFAULT_OUT = ROOT / "data" / "tinker"
TASKS = (
    "claim_extraction",
    "search_planning",
    "evidence_selection",
    "claim_evaluation",
)


def reservoir_sample(path: Path, size: int, seed: str) -> tuple[list[dict], int]:
    rng = random.Random(seed)
    sample: list[str] = []
    seen = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            seen += 1
            if len(sample) < size:
                sample.append(line)
                continue
            replacement = rng.randrange(seen)
            if replacement < size:
                sample[replacement] = line

    rows = [json.loads(line) for line in sample]
    for row in rows:
        if row.get("task") != path.stem:
            raise ValueError(f"Task mismatch in {path}: {row.get('task')!r}")
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"Invalid messages in {path}: {row.get('example_id')!r}")
    return rows, seen


def summarize(rows: list[dict]) -> dict:
    by_task = Counter(row["task"] for row in rows)
    by_source = Counter(row.get("source_dataset", "unknown") for row in rows)
    labels: Counter = Counter()
    for row in rows:
        if row["task"] == "claim_evaluation":
            output = json.loads(row["messages"][-1]["content"])
            labels[output["label"]] += 1
    return {
        "total": len(rows),
        "by_task": dict(sorted(by_task.items())),
        "by_source": dict(sorted(by_source.items())),
        "claim_evaluation_labels": dict(sorted(labels.items())),
    }


def write_jsonl(path: Path, rows: list[dict]) -> str:
    digest = sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-per-task", type=int, default=2500)
    parser.add_argument("--eval-per-task", type=int, default=250)
    parser.add_argument("--seed", type=int, default=5870)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.train_per_task < 1 or args.eval_per_task < 1:
        parser.error("sample sizes must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict] = {}
    source_sizes: dict[str, int] = {}

    for split, per_task in (("train", args.train_per_task), ("eval", args.eval_per_task)):
        rows: list[dict] = []
        for task in TASKS:
            path = PROCESSED / split / f"{task}.jsonl"
            selected, seen = reservoir_sample(
                path, per_task, seed=f"{args.seed}:{split}:{task}"
            )
            rows.extend(selected)
            source_sizes[f"{split}/{task}"] = seen

        random.Random(f"{args.seed}:{split}:shuffle").shuffle(rows)
        output = args.out_dir / f"pilot_{split}.jsonl"
        digest = write_jsonl(output, rows)
        outputs[split] = {
            "path": str(output.relative_to(ROOT)),
            "sha256": digest,
            **summarize(rows),
        }

    manifest = {
        "seed": args.seed,
        "requested_per_task": {
            "train": args.train_per_task,
            "eval": args.eval_per_task,
        },
        "source_sizes": source_sizes,
        "outputs": outputs,
    }
    manifest_path = args.out_dir / "pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
