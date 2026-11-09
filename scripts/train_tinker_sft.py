"""Run a guarded Tinker SFT smoke test or pilot on the prepared data.

The command is inert unless --execute is supplied.  Defaults intentionally run
only five optimizer steps and disable periodic checkpoints; the cookbook still
saves one final checkpoint so the result can be sampled and evaluated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from itertools import islice
from pathlib import Path

import chz
import datasets
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import (
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)


ROOT = Path(__file__).resolve().parents[1]


def read_conversations(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        lines = islice(handle, limit) if limit is not None else handle
        for line in lines:
            row = json.loads(line)
            rows.append({"messages": row["messages"]})
    if not rows:
        raise ValueError(f"No conversations found in {path}")
    return rows


@chz.chz
class PairedConversationFileBuilder(ChatDatasetBuilder):
    train_path: Path
    eval_path: Path
    eval_limit: int = 256
    shuffle_seed: int = 5870

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        train_rows = read_conversations(self.train_path)
        eval_rows = read_conversations(self.eval_path, self.eval_limit)
        train_hf = datasets.Dataset.from_list(train_rows).shuffle(seed=self.shuffle_seed)
        eval_hf = datasets.Dataset.from_list(eval_rows)
        renderer = self.renderer

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"],
                renderer,
                self.common_config.max_length,
                renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )

        batch_size = self.common_config.batch_size
        return (
            SupervisedDatasetFromHFDataset(train_hf, batch_size=batch_size, map_fn=map_fn),
            SupervisedDatasetFromHFDataset(eval_hf, batch_size=batch_size, map_fn=map_fn),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--renderer", default="qwen3_5_disable_thinking")
    parser.add_argument("--train-file", type=Path, default=ROOT / "data/tinker/pilot_train.jsonl")
    parser.add_argument("--eval-file", type=Path, default=ROOT / "data/tinker/pilot_eval.jsonl")
    parser.add_argument("--eval-limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--session-name")
    parser.add_argument("--log-path", type=Path)
    args = parser.parse_args()

    for path in (args.train_file, args.eval_file):
        if not path.is_file():
            parser.error(f"missing prepared dataset: {path}")
    if min(args.batch_size, args.max_steps, args.eval_limit) < 1:
        parser.error("batch size, max steps, and eval limit must be positive")
    if args.eval_every < 1 or args.save_every < 0:
        parser.error("eval-every must be positive and save-every cannot be negative")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = args.log_path or ROOT / "runs" / f"qwen35-4b-smoke-{timestamp}"
    plan = {
        "execute": args.execute,
        "model": args.model,
        "renderer": args.renderer,
        "train_file": str(args.train_file),
        "eval_file": str(args.eval_file),
        "eval_limit": args.eval_limit,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "log_path": str(log_path),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print("Plan only; pass --execute to create a paid Tinker training session.")
        return

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model,
        renderer_name=args.renderer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    builder = PairedConversationFileBuilder(
        common_config=common_config,
        train_path=args.train_file,
        eval_path=args.eval_file,
        eval_limit=args.eval_limit,
    )
    config = train.Config(
        log_path=str(log_path),
        model_name=args.model,
        recipe_name="background_checkers_multitask_sft",
        renderer_name=args.renderer,
        dataset_builder=builder,
        learning_rate=args.learning_rate,
        num_epochs=1,
        lora_rank=args.lora_rank,
        save_every=args.save_every,
        eval_every=args.eval_every,
        infrequent_eval_every=0,
        max_steps=args.max_steps,
        session_name=args.session_name or f"background-checkers-{'smoke' if args.max_steps <= 5 else 'pilot'}-{timestamp}",
    )
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
