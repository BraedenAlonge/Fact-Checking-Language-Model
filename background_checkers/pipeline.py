"""End-to-end orchestration for the Background Checkers prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from .model import DEFAULT_CHECKPOINT, TaskModel
from .retrieval import PageRetriever, RetrievedSource, domain_for_url
from .search import SearchError, SearchProvider, canonicalize_url


@dataclass
class PipelineConfig:
    checkpoint: str = DEFAULT_CHECKPOINT
    max_claims: int = 12
    search_results_per_attempt: int = 8
    max_refinements: int = 1
    max_total_results_per_claim: int = 12
    max_evidence_per_claim: int = 3
    fetch_pages: bool = True

    def __post_init__(self) -> None:
        numeric = {
            "max_claims": self.max_claims,
            "search_results_per_attempt": self.search_results_per_attempt,
            "max_total_results_per_claim": self.max_total_results_per_claim,
            "max_evidence_per_claim": self.max_evidence_per_claim,
        }
        for name, value in numeric.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_refinements < 0:
            raise ValueError("max_refinements cannot be negative")


def enforce_evidence_contract(
    claim_id: str,
    selection: dict[str, Any],
    sources: list[RetrievedSource],
    max_evidence: int,
) -> dict[str, Any]:
    """Drop invented/duplicate sources and restore exact input source fields."""

    by_id = {source.source_id: source for source in sources}
    verified_page_available = any(source.fetch_status == "page_text" for source in sources)
    evidence = []
    seen_domains: set[str] = set()
    dropped = 0
    for item in selection.get("evidence", []):
        source = by_id.get(item.get("source_id"))
        if source is None:
            dropped += 1
            continue
        if verified_page_available and source.fetch_status != "page_text":
            dropped += 1
            continue
        domain = source.domain or domain_for_url(source.url)
        if domain and domain in seen_domains:
            dropped += 1
            continue
        if domain:
            seen_domains.add(domain)
        evidence.append(
            {
                "source_id": source.source_id,
                "title": source.title,
                "url": source.url,
                "excerpt": source.excerpt,
                "stance": item["stance"],
                "quality": item["quality"],
            }
        )
        if len(evidence) >= max_evidence:
            break
    needs_refinement = bool(selection.get("needs_refinement")) or not evidence
    reason = selection.get("refinement_reason")
    if not evidence and not reason:
        reason = "No usable independent evidence was selected."
    elif dropped and needs_refinement and not reason:
        reason = "Some selected results were unknown or duplicated the same source domain."
    return {
        "claim_id": claim_id,
        "evidence": evidence,
        "needs_refinement": needs_refinement,
        "refinement_reason": reason,
        "runtime_dropped_evidence_items": dropped,
    }


def enforce_evaluation_contract(
    claim_id: str,
    evaluation: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    allowed = {item["source_id"] for item in evidence}
    source_ids = [item for item in evaluation.get("source_ids", []) if item in allowed]
    if not evidence:
        return {
            "claim_id": claim_id,
            "label": "Insufficient Evidence",
            "explanation": "No usable evidence was selected for this claim.",
            "source_ids": [],
        }
    if evaluation["label"] != "Insufficient Evidence" and not source_ids:
        return {
            "claim_id": claim_id,
            "label": "Insufficient Evidence",
            "explanation": "The model did not cite any usable selected evidence for its verdict.",
            "source_ids": [],
        }
    return {
        "claim_id": claim_id,
        "label": evaluation["label"],
        "explanation": evaluation["explanation"],
        "source_ids": list(dict.fromkeys(source_ids)),
    }


class PrototypePipeline:
    def __init__(
        self,
        model: TaskModel,
        search_provider: SearchProvider,
        retriever: PageRetriever,
        config: PipelineConfig | None = None,
    ) -> None:
        self.model = model
        self.search_provider = search_provider
        self.retriever = retriever
        self.config = config or PipelineConfig()

    async def _process_claim(self, claim: dict[str, Any]) -> dict[str, Any]:
        claim_id = claim["claim_id"]
        sources: list[RetrievedSource] = []
        seen_urls: set[str] = set()
        used_queries: set[str] = set()
        attempts: list[dict[str, Any]] = []
        selection: dict[str, Any] = {
            "claim_id": claim_id,
            "evidence": [],
            "needs_refinement": True,
            "refinement_reason": "No search has been run.",
        }

        for attempt_index in range(self.config.max_refinements + 1):
            search_claim = dict(claim)
            if attempts:
                previous = attempts[-1]
                search_claim["context"] = (
                    f"{claim.get('context', '')} Previous query: {previous['query']}. "
                    f"Refinement reason: {previous.get('refinement_reason') or 'inadequate evidence'}. "
                    "Create a different query that addresses the missing evidence."
                ).strip()
            planning = await self.model.plan_search(search_claim)
            query = planning["arguments"]["query"].strip()
            query_key = query.casefold()
            if query_key in used_queries:
                query = f"{claim['claim_text']} {claim.get('context', '')}".strip()[:600]
                query_key = query.casefold()
            used_queries.add(query_key)
            attempt_report: dict[str, Any] = {
                "attempt": attempt_index + 1,
                "query": query,
                "provider": self.search_provider.name,
                "search_error": None,
                "fallback_query": None,
                "new_source_ids": [],
            }
            search_results = []
            try:
                search_results = await self.search_provider.search(
                    query, self.config.search_results_per_attempt
                )
            except SearchError as exc:
                attempt_report["search_error"] = str(exc)
                fallback_query = claim["claim_text"].strip()[:600]
                if fallback_query.casefold() != query.casefold():
                    attempt_report["fallback_query"] = fallback_query
                    try:
                        search_results = await self.search_provider.search(
                            fallback_query, self.config.search_results_per_attempt
                        )
                    except SearchError as fallback_exc:
                        attempt_report["search_error"] += f"; fallback failed: {fallback_exc}"

            if search_results:
                unique_results = []
                for result in search_results:
                    url = canonicalize_url(result.url)
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    unique_results.append(result)
                    if len(sources) + len(unique_results) >= self.config.max_total_results_per_claim:
                        break
                retrieved = await self.retriever.retrieve(
                    unique_results,
                    claim["claim_text"],
                    query,
                    fetch_pages=self.config.fetch_pages,
                    source_offset=len(sources),
                )
                sources.extend(source for source in retrieved if source.excerpt)
                attempt_report["new_source_ids"] = [source.source_id for source in retrieved]

            if sources:
                raw_selection = await self.model.select_evidence(
                    claim, [source.model_dict() for source in sources]
                )
                selection = enforce_evidence_contract(
                    claim_id,
                    raw_selection,
                    sources,
                    self.config.max_evidence_per_claim,
                )
            else:
                selection = {
                    "claim_id": claim_id,
                    "evidence": [],
                    "needs_refinement": True,
                    "refinement_reason": "Search returned no usable evidence text.",
                    "runtime_dropped_evidence_items": 0,
                }
            attempt_report["needs_refinement"] = selection["needs_refinement"]
            attempt_report["refinement_reason"] = selection["refinement_reason"]
            attempts.append(attempt_report)
            if not selection["needs_refinement"]:
                break

        evidence = selection["evidence"]
        raw_evaluation = await self.model.evaluate_claim(claim, evidence)
        evaluation = enforce_evaluation_contract(claim_id, raw_evaluation, evidence)
        cited = set(evaluation["source_ids"])
        citations = [
            {
                "source_id": item["source_id"],
                "title": item["title"],
                "url": item["url"],
            }
            for item in evidence
            if item["source_id"] in cited
        ]
        return {
            "claim": claim,
            "status": "complete",
            "search_attempts": attempts,
            "retrieved_sources": [source.report_dict() for source in sources],
            "evidence_selection": selection,
            "evaluation": evaluation,
            "citations": citations,
        }

    async def run(self, text: str) -> dict[str, Any]:
        normalized = text.strip()
        if not normalized:
            raise ValueError("input text cannot be empty")
        extraction = await self.model.extract_claims(normalized)
        claims = extraction["claims"][: self.config.max_claims]
        results = []
        pipeline_errors = []
        for claim in claims:
            try:
                results.append(await self._process_claim(claim))
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                pipeline_errors.append(
                    {"claim_id": claim.get("claim_id"), "error": message}
                )
                results.append(
                    {
                        "claim": claim,
                        "status": "error",
                        "error": message,
                        "search_attempts": [],
                        "retrieved_sources": [],
                        "evidence_selection": None,
                        "evaluation": None,
                        "citations": [],
                    }
                )
        return {
            "schema_version": "background-checkers-prototype-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_checkpoint": self.config.checkpoint,
            "search_provider": self.search_provider.name,
            "configuration": asdict(self.config),
            "input_text": normalized,
            "claim_extraction": {
                "total_generated": len(extraction["claims"]),
                "processed": len(claims),
                "truncated": len(extraction["claims"]) > len(claims),
            },
            "claims": results,
            "errors": pipeline_errors,
        }
