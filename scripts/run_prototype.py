"""Run the guarded end-to-end Background Checkers prototype."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from background_checkers.model import DEFAULT_CHECKPOINT, TinkerTaskModel
from background_checkers.pipeline import PipelineConfig, PrototypePipeline
from background_checkers.retrieval import PageRetriever
from background_checkers.search import create_search_provider

def read_input(args) -> str:
    if args.text is not None:
        return args.text.strip()
    return args.file.read_text(encoding="utf-8").strip()


def print_summary(report: dict) -> None:
    print("\nBACKGROUND CHECKERS PROTOTYPE REPORT")
    print("=" * 36)
    print(f"Claims processed: {report['claim_extraction']['processed']}")
    if not report["claims"]:
        print("No independently checkable factual claims were extracted.")
        return
    for index, result in enumerate(report["claims"], start=1):
        claim = result["claim"]
        print(f"\n{index}. {claim['claim_text']}")
        if result["status"] != "complete":
            print(f"   ERROR: {result['error']}")
            continue
        evaluation = result["evaluation"]
        print(f"   Verdict: {evaluation['label']}")
        print(f"   Explanation: {evaluation['explanation']}")
        if result["citations"]:
            print("   Sources:")
            for citation in result["citations"]:
                print(
                    f"   - [{citation['source_id']}] {citation['title']}: "
                    f"{citation['url']}"
                )
        else:
            print("   Sources: none selected")


async def execute(args, text: str, output_path: Path) -> None:
    provider = create_search_provider(args.provider)
    retriever = PageRetriever(
        timeout=args.fetch_timeout,
        max_bytes=args.max_page_bytes,
        concurrency=args.fetch_concurrency,
    )
    config = PipelineConfig(
        checkpoint=args.checkpoint,
        max_claims=args.max_claims,
        search_results_per_attempt=args.search_results,
        max_refinements=args.max_refinements,
        max_total_results_per_claim=args.max_total_results,
        max_evidence_per_claim=args.max_evidence,
        fetch_pages=not args.no_fetch,
    )
    async with TinkerTaskModel(
        checkpoint=args.checkpoint,
        max_tokens=args.max_model_tokens,
        correction_attempts=args.correction_attempts,
    ) as model:
        pipeline = PrototypePipeline(model, provider, retriever, config)
        report = await pipeline.run(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print_summary(report)
    print(f"\nFull JSON report: {output_path.resolve()}")


def main() -> None:
    # Windows PowerShell may expose a legacy console encoding. Keep a successful
    # run from failing only because a source excerpt contains IPA or other
    # Unicode characters that the console cannot represent. The JSON report is
    # always written as UTF-8 and preserves the original characters.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(
        description="Run claim extraction, search, retrieval, evidence selection, and evaluation."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Text to fact-check")
    input_group.add_argument("--file", type=Path, help="UTF-8 text file to fact-check")
    parser.add_argument("--execute", action="store_true", help="Allow paid Tinker and live search calls")
    parser.add_argument("--provider", choices=("ddgs", "brave"), default="ddgs")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-claims", type=int, default=8)
    parser.add_argument("--search-results", type=int, default=8)
    parser.add_argument("--max-total-results", type=int, default=12)
    parser.add_argument("--max-refinements", type=int, default=1)
    parser.add_argument("--max-evidence", type=int, default=3)
    parser.add_argument("--max-model-tokens", type=int, default=1024)
    parser.add_argument("--correction-attempts", type=int, default=1)
    parser.add_argument("--fetch-timeout", type=float, default=12.0)
    parser.add_argument("--fetch-concurrency", type=int, default=4)
    parser.add_argument("--max-page-bytes", type=int, default=2_000_000)
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Use search snippets only instead of retrieving result pages",
    )
    args = parser.parse_args()
    if args.file is not None and not args.file.is_file():
        parser.error(f"input file does not exist: {args.file}")
    positive = (
        args.max_claims,
        args.search_results,
        args.max_total_results,
        args.max_evidence,
        args.max_model_tokens,
        args.fetch_concurrency,
        args.max_page_bytes,
    )
    if min(positive) < 1 or args.max_refinements < 0 or args.correction_attempts < 0:
        parser.error("limits must be positive; refinement/correction counts cannot be negative")
    text = read_input(args)
    if not text:
        parser.error("input text cannot be empty")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = args.output or ROOT / "prototype_outputs" / f"report-{timestamp}.json"
    plan = {
        "execute": args.execute,
        "provider": args.provider,
        "checkpoint": args.checkpoint,
        "input_characters": len(text),
        "max_claims": args.max_claims,
        "search_results_per_attempt": args.search_results,
        "max_refinements": args.max_refinements,
        "fetch_pages": not args.no_fetch,
        "output": str(output_path),
    }
    print(json.dumps(plan, indent=2))
    if not args.execute:
        print("Plan only; pass --execute to allow paid Tinker and live web-search calls.")
        return
    asyncio.run(execute(args, text, output_path))


if __name__ == "__main__":
    main()
