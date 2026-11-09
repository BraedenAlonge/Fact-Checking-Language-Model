"""Compare a Tinker checkpoint with its base model on held-out task examples."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

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


def held_out_examples(path: Path) -> list[dict]:
    selected: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            selected.setdefault(row["task"], row)
            if len(selected) == len(TASKS):
                break
    missing = set(TASKS) - selected.keys()
    if missing:
        raise ValueError(f"Missing held-out tasks: {sorted(missing)}")
    return [selected[task] for task in TASKS]


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


async def run(args) -> None:
    tokenizer = get_tokenizer(args.model)
    renderer = renderers.get_renderer(args.renderer, tokenizer, model_name=args.model)
    examples = held_out_examples(args.eval_file)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tinker.ServiceClient(display_name="background-checkers-smoke-comparison") as service:
        base_client = await service.create_sampling_client(base_model=args.model)
        tuned_client = await service.create_sampling_client(model_path=args.checkpoint)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in examples:
                base = await sample_one(base_client, renderer, row["messages"], args.max_tokens)
                tuned = await sample_one(tuned_client, renderer, row["messages"], args.max_tokens)
                result = {
                    "task": row["task"],
                    "example_id": row["example_id"],
                    "expected": row["messages"][2]["content"],
                    "base": base,
                    "tuned": tuned,
                    "base_valid_json": _valid_json(base),
                    "tuned_valid_json": _valid_json(tuned),
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(
                    f"{row['task']}: base_json={result['base_valid_json']} "
                    f"tuned_json={result['tuned_valid_json']}",
                    flush=True,
                )


def _valid_json(value: str) -> bool:
    try:
        json.loads(value)
        return True
    except json.JSONDecodeError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--renderer", default="qwen3_5_disable_thinking")
    parser.add_argument("--eval-file", type=Path, default=ROOT / "data/tinker/pilot_eval.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=768)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
