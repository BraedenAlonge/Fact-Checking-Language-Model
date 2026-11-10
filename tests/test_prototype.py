from __future__ import annotations

import unittest

from background_checkers.pipeline import (
    PipelineConfig,
    PrototypePipeline,
    enforce_evaluation_contract,
    enforce_evidence_contract,
)
from background_checkers.retrieval import PageRetriever, RetrievedSource, select_excerpt
from background_checkers.schemas import (
    SchemaError,
    loads_object,
    validate_claim_evaluation,
    validate_claim_extraction,
    validate_evidence_selection,
    validate_search_planning,
)
from background_checkers.search import (
    SearchResult,
    StaticSearchProvider,
    canonicalize_url,
)


class SchemaTests(unittest.TestCase):
    def test_json_fence_is_accepted(self) -> None:
        self.assertEqual(loads_object('```json\n{"claims":[]}\n```'), {"claims": []})

    def test_all_gold_shapes_validate(self) -> None:
        validate_claim_extraction(
            {
                "claims": [
                    {"claim_id": "c1", "claim_text": "A factual claim.", "context": "Context"}
                ]
            }
        )
        validate_search_planning(
            {"claim_id": "c1", "tool": "web_search", "arguments": {"query": "factual claim"}}
        )
        validate_evidence_selection(
            {
                "claim_id": "c1",
                "evidence": [
                    {
                        "source_id": "s1",
                        "title": "Title",
                        "url": "https://example.com",
                        "excerpt": "Text",
                        "stance": "supports",
                        "quality": "high",
                    }
                ],
                "needs_refinement": False,
                "refinement_reason": None,
            }
        )
        validate_claim_evaluation(
            {
                "claim_id": "c1",
                "label": "Supported",
                "explanation": "The evidence supports the claim.",
                "source_ids": ["s1"],
            }
        )

    def test_extra_schema_fields_are_rejected(self) -> None:
        with self.assertRaises(SchemaError):
            validate_claim_extraction({"claims": [], "commentary": "not allowed"})


class RuntimeRuleTests(unittest.TestCase):
    def test_canonical_url_removes_tracking_and_fragment(self) -> None:
        value = canonicalize_url(
            "HTTPS://Example.COM/path/?utm_source=test&keep=yes#section"
        )
        self.assertEqual(value, "https://example.com/path?keep=yes")

    def test_canonical_url_rejects_invalid_port(self) -> None:
        self.assertEqual(canonicalize_url("https://example.com:not-a-port/path"), "")

    def test_excerpt_prefers_relevant_text(self) -> None:
        text = (
            "A paragraph about cooking and ingredients.\n"
            "The observatory recorded the comet on March 14, 2025 near Jupiter.\n"
            "A paragraph about an unrelated sports result."
        )
        excerpt = select_excerpt(text, "The comet was recorded near Jupiter.", "comet Jupiter")
        self.assertIn("observatory recorded the comet", excerpt)

    def test_runtime_drops_invented_and_same_domain_evidence(self) -> None:
        sources = [
            RetrievedSource("s1", "One", "https://a.example/1", "web", "First", "a.example", "q", "static", "snippet_only"),
            RetrievedSource("s2", "Two", "https://a.example/2", "web", "Second", "a.example", "q", "static", "snippet_only"),
        ]
        selection = {
            "claim_id": "wrong",
            "evidence": [
                {"source_id": "s1", "title": "invented", "url": "invented", "excerpt": "invented", "stance": "supports", "quality": "high"},
                {"source_id": "s2", "title": "Two", "url": "https://a.example/2", "excerpt": "Second", "stance": "supports", "quality": "high"},
                {"source_id": "s9", "title": "Unknown", "url": "", "excerpt": "", "stance": "context", "quality": "low"},
            ],
            "needs_refinement": False,
            "refinement_reason": None,
        }
        enforced = enforce_evidence_contract("c1", selection, sources, 3)
        self.assertEqual([item["source_id"] for item in enforced["evidence"]], ["s1"])
        self.assertEqual(enforced["evidence"][0]["title"], "One")
        self.assertEqual(enforced["runtime_dropped_evidence_items"], 2)

    def test_empty_evidence_forces_insufficient(self) -> None:
        enforced = enforce_evaluation_contract(
            "c1",
            {
                "claim_id": "c1",
                "label": "Supported",
                "explanation": "Unsupported output.",
                "source_ids": ["s9"],
            },
            [],
        )
        self.assertEqual(enforced["label"], "Insufficient Evidence")
        self.assertEqual(enforced["source_ids"], [])

    def test_uncited_positive_verdict_forces_insufficient(self) -> None:
        evidence = [
            {
                "source_id": "s1",
                "title": "One",
                "url": "https://a.example/1",
                "excerpt": "Evidence",
                "stance": "supports",
                "quality": "high",
            }
        ]
        enforced = enforce_evaluation_contract(
            "c1",
            {
                "claim_id": "c1",
                "label": "Supported",
                "explanation": "The claim is supported.",
                "source_ids": ["invented"],
            },
            evidence,
        )
        self.assertEqual(enforced["label"], "Insufficient Evidence")
        self.assertEqual(enforced["source_ids"], [])

    def test_verified_page_text_wins_over_search_snippet(self) -> None:
        sources = [
            RetrievedSource("s1", "Fetched", "https://a.example/1", "web", "Verified", "a.example", "q", "static", "page_text"),
            RetrievedSource("s2", "Snippet", "https://b.example/2", "web", "Unverified", "b.example", "q", "static", "snippet_fallback:HTTPStatusError"),
        ]
        selection = {
            "claim_id": "c1",
            "evidence": [
                {"source_id": "s1", "title": "Fetched", "url": "https://a.example/1", "excerpt": "Verified", "stance": "supports", "quality": "high"},
                {"source_id": "s2", "title": "Snippet", "url": "https://b.example/2", "excerpt": "Unverified", "stance": "supports", "quality": "medium"},
            ],
            "needs_refinement": False,
            "refinement_reason": None,
        }
        enforced = enforce_evidence_contract("c1", selection, sources, 3)
        self.assertEqual([item["source_id"] for item in enforced["evidence"]], ["s1"])
        self.assertEqual(enforced["runtime_dropped_evidence_items"], 1)


class FakeModel:
    def __init__(self, no_claims: bool = False) -> None:
        self.no_claims = no_claims

    async def extract_claims(self, text):
        del text
        if self.no_claims:
            return {"claims": []}
        return {
            "claims": [
                {
                    "claim_id": "c1",
                    "claim_text": "The observatory recorded the comet in 2025.",
                    "context": "Astronomy report",
                }
            ]
        }

    async def plan_search(self, claim):
        return {
            "claim_id": claim["claim_id"],
            "tool": "web_search",
            "arguments": {"query": "observatory comet 2025"},
        }

    async def select_evidence(self, claim, search_results):
        del claim
        first = search_results[0]
        return {
            "claim_id": "c1",
            "evidence": [
                {
                    "source_id": first["source_id"],
                    "title": first["title"],
                    "url": first["url"],
                    "excerpt": first["excerpt"],
                    "stance": "supports",
                    "quality": "medium",
                }
            ],
            "needs_refinement": False,
            "refinement_reason": None,
        }

    async def evaluate_claim(self, claim, evidence):
        return {
            "claim_id": claim["claim_id"],
            "label": "Supported" if evidence else "Insufficient Evidence",
            "explanation": "The selected source supports the claim." if evidence else "No evidence.",
            "source_ids": [item["source_id"] for item in evidence],
        }


class CountingStaticProvider(StaticSearchProvider):
    def __init__(self, results):
        super().__init__(results)
        self.calls = 0

    async def search(self, query, limit):
        self.calls += 1
        return await super().search(query, limit)


class FailOnceProvider(CountingStaticProvider):
    async def search(self, query, limit):
        from background_checkers.search import SearchError

        self.calls += 1
        if self.calls == 1:
            raise SearchError("transient provider failure")
        return await StaticSearchProvider.search(self, query, limit)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_offline_pipeline(self) -> None:
        provider = CountingStaticProvider(
            [
                SearchResult(
                    "Observatory report",
                    "https://science.example/comet",
                    "The observatory recorded the comet in March 2025.",
                    "static",
                )
            ]
        )
        pipeline = PrototypePipeline(
            FakeModel(),
            provider,
            PageRetriever(),
            PipelineConfig(fetch_pages=False, max_refinements=0),
        )
        report = await pipeline.run("A passage containing one astronomy claim.")
        self.assertEqual(provider.calls, 1)
        self.assertEqual(report["claims"][0]["evaluation"]["label"], "Supported")
        self.assertEqual(report["claims"][0]["citations"][0]["source_id"], "s1")

    async def test_no_claims_skips_search(self) -> None:
        provider = CountingStaticProvider([])
        pipeline = PrototypePipeline(
            FakeModel(no_claims=True),
            provider,
            PageRetriever(),
            PipelineConfig(fetch_pages=False),
        )
        report = await pipeline.run("Thanks for your help!")
        self.assertEqual(report["claims"], [])
        self.assertEqual(provider.calls, 0)

    async def test_search_failure_retries_with_raw_claim(self) -> None:
        provider = FailOnceProvider(
            [
                SearchResult(
                    "Observatory report",
                    "https://science.example/comet",
                    "The observatory recorded the comet in March 2025.",
                    "static",
                )
            ]
        )
        pipeline = PrototypePipeline(
            FakeModel(),
            provider,
            PageRetriever(),
            PipelineConfig(fetch_pages=False, max_refinements=0),
        )
        report = await pipeline.run("A passage containing one astronomy claim.")
        attempt = report["claims"][0]["search_attempts"][0]
        self.assertEqual(provider.calls, 2)
        self.assertEqual(
            attempt["fallback_query"], "The observatory recorded the comet in 2025."
        )
        self.assertEqual(report["claims"][0]["evaluation"]["label"], "Supported")


if __name__ == "__main__":
    unittest.main()
