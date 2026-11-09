"""Tokenize prepared chat JSONL locally and estimate Tinker training cost."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from tinker_cookbook import renderers
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    return sorted(values)[round((len(values) - 1) * fraction)]


def inspect_file(
    path: Path,
    renderer: renderers.Renderer,
    max_length: int,
) -> dict:
    lengths: list[int] = []
    by_task: dict[str, list[int]] = {}
    over_limit = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            datum = conversation_to_datum(
                row["messages"],
                renderer,
                max_length=None,
                train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            length = datum.model_input.length
            lengths.append(length)
            by_task.setdefault(row["task"], []).append(length)
            over_limit += length > max_length

    return {
        "records": len(lengths),
        "tokens": sum(lengths),
        "mean_tokens": round(statistics.fmean(lengths), 1),
        "p50_tokens": percentile(lengths, 0.50),
        "p95_tokens": percentile(lengths, 0.95),
        "max_tokens": max(lengths, default=0),
        "over_max_length": over_limit,
        "by_task": {
            task: {
                "records": len(task_lengths),
                "tokens": sum(task_lengths),
                "mean_tokens": round(statistics.fmean(task_lengths), 1),
                "p95_tokens": percentile(task_lengths, 0.95),
                "max_tokens": max(task_lengths),
            }
            for task, task_lengths in sorted(by_task.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--renderer", default="qwen3_5_disable_thinking")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--train-price-per-million", type=float, default=0.737)
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)
    renderer = renderers.get_renderer(args.renderer, tokenizer, model_name=args.model)
    report = {
        "model": args.model,
        "renderer": args.renderer,
        "max_length": args.max_length,
        "train_price_per_million_tokens_usd": args.train_price_per_million,
        "files": {},
    }
    for path in args.paths:
        details = inspect_file(path, renderer, args.max_length)
        details["one_epoch_train_cost_usd"] = round(
            details["tokens"] / 1_000_000 * args.train_price_per_million, 4
        )
        report["files"][str(path)] = details
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
