"""Create deterministic, task- and source-balanced Tinker pilot datasets.

The complete converted corpus is intentionally much larger than a sensible
first training run. This script reservoir-samples within every source dataset,
then draws round-robin across sources for each task. It keeps the repository's
train and eval splits separate and writes ordinary conversation JSONL that
tinker-cookbook can consume directly.
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


def source_balanced_sample(path: Path, size: int, seed: str) -> tuple[list[dict], int]:
    """Sample a task without allowing its largest source to dominate.

    Each source gets an independent reservoir of at most ``size`` rows. The
    final sample cycles across those reservoirs until the requested size is
    reached. Small sources are therefore fully represented before large sources
    fill the remaining slots.
    """

    samples: dict[str, list[str]] = {}
    source_rngs: dict[str, random.Random] = {}
    source_seen: Counter = Counter()
    seen = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source_dataset", "unknown")
            if source not in source_rngs:
                source_rngs[source] = random.Random(f"{seed}:{source}")
            seen += 1
            source_seen[source] += 1
            sample = samples.setdefault(source, [])
            if len(sample) < size:
                sample.append(line)
                continue
            replacement = source_rngs[source].randrange(source_seen[source])
            if replacement < size:
                sample[replacement] = line

    source_order = sorted(samples)
    random.Random(f"{seed}:source-order").shuffle(source_order)
    for source in source_order:
        random.Random(f"{seed}:{source}:shuffle").shuffle(samples[source])

    selected: list[str] = []
    offsets = {source: 0 for source in source_order}
    active = list(source_order)
    while active and len(selected) < size:
        next_active = []
        for source in active:
            offset = offsets[source]
            if offset < len(samples[source]) and len(selected) < size:
                selected.append(samples[source][offset])
                offsets[source] += 1
            if offsets[source] < len(samples[source]):
                next_active.append(source)
        active = next_active

    rows = [json.loads(line) for line in selected]
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
            selected, seen = source_balanced_sample(
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
        "sampling_strategy": "equal round-robin across source_dataset within each task",
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
