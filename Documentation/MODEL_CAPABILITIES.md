# Current Model Capabilities

Last updated: 2026-11-09

## Selected prototype model

The current candidate is a LoRA adaptation of `Qwen/Qwen3.5-4B`, trained with the non-thinking renderer for direct structured outputs.

```text
tinker://7e61d5a7-3f20-593b-a4fa-691ca4cb308e:train:0/sampler_weights/000200
```

This step-200 checkpoint was selected because it has the best held-out NLL and the strongest repaired no-claim behavior while retaining good search-planning and evidence-selection performance.

## Scope

The model is a structured, four-stage fact-checking prototype. It is not itself a web-browsing fact checker. This repository now supplies a command-line application that performs search, page retrieval, text extraction, source management, deterministic validation, and report generation around the model.

| Capability | Current behavior |
| --- | --- |
| Claim extraction | Identifies individual factual claims and rejects most opinions, commands, and conversational filler. |
| Search planning | Produces one structured web-search request for a claim. |
| Evidence selection | Selects relevant items from supplied search results without rewriting their source fields. |
| Claim evaluation | Assigns `Supported`, `Contradicted`, or `Insufficient Evidence` using supplied evidence. |
| Structured output | Produces machine-readable JSON for all four tasks. |

The model was trained and evaluated with a configured maximum sequence length of 8,192 tokens.

## Claim extraction

The model accepts prose such as an LLM response, article paragraph, or debate statement and returns independently checkable claims:

```json
{
  "claims": [
    {
      "claim_id": "c1",
      "claim_text": "The independently checkable claim.",
      "context": "Context needed to interpret it."
    }
  ]
}
```

For passages containing no independently checkable factual claim, the target output is:

```json
{"claims":[]}
```

Measured on the fixed stratified suite:

- strict schema compliance: 95.8%;
- normalized exact-text claim micro-F1: 0.416;
- no-claim accuracy: 88.9%, or 8 of 9 cases; and
- explicit false-positive claim rate on those negative cases: 0%.

The one remaining no-claim failure was an invalid output rather than an extracted false claim. Exact-text F1 is deliberately strict: semantically equivalent claims with different wording or boundaries count as mismatches. The model may still over-split, under-split, omit, or paraphrase claims.

## Search planning

For one supplied claim, the model produces a structured search request:

```json
{
  "claim_id": "c1",
  "tool": "web_search",
  "arguments": {
    "query": "concise verification query"
  }
}
```

Measured performance:

- strict schema compliance: 100%;
- mean target-query token F1: 0.611; and
- target-number preservation: 83.3%.

The output is a proposed query, not a search result. A real search provider must execute it, and some queries will need deterministic or model-guided refinement. Target overlap is not a substitute for human relevance scoring.

## Evidence selection

Given a claim and normalized search results, the model can:

- choose relevant candidates;
- reject irrelevant candidates;
- copy source IDs, titles, URLs, and excerpts;
- label evidence stance as `supports`, `contradicts`, or `context`;
- label source quality as `high`, `medium`, or `low`; and
- report whether another search is needed.

Measured performance:

- strict schema compliance: 100%;
- selected source-ID F1: 1.000;
- exact source-field copy rate: 100%;
- stance accuracy: 83.3%;
- quality accuracy: 83.3%; and
- refinement accuracy: 79.2%.

This is currently the strongest task. The evaluation evidence remains benchmark-heavy and cleaner than live web results. SEO spam, inaccessible pages, syndicated reporting, low-quality sources, and complicated multi-source conflicts are not adequately covered yet.

## Claim evaluation

Given one claim and selected evidence, the model returns:

```json
{
  "claim_id": "c1",
  "label": "Supported",
  "explanation": "Why the supplied evidence supports the complete claim.",
  "source_ids": ["s1", "s2"]
}
```

Measured performance:

- strict schema compliance: 100%;
- verdict accuracy: 79.2%, or 19 of 24 cases;
- verdict macro-F1: 0.793; and
- cited source-ID F1: 1.000.

Verdict classification is the main current regression. It performs below the first pilot checkpoint on the small fixed suite and needs targeted error review, particularly around `Contradicted` versus `Insufficient Evidence`.

## Intended prototype workflow

```text
Input text
   -> model extracts claims
   -> model proposes a search query
   -> external application searches and retrieves pages
   -> model selects evidence from normalized results
   -> model evaluates the claim and returns source IDs
```

The search and retrieval wrapper is now implemented in `background_checkers/` with a guarded entry point at `scripts/run_prototype.py`. It is suitable for a controlled project demonstration of the complete concept and emits a cited JSON audit report. It is not yet a hardened service, local web UI, or browser extension. Setup and usage are documented in `PROTOTYPE_RUNTIME.md`.

## What the model cannot do by itself

The model itself does not:

- browse or search the internet;
- retrieve, parse, or archive webpages;
- determine whether information is current;
- guarantee that selected sources are independent;
- reliably identify duplicate or syndicated reporting;
- verify quotations against complete source pages;
- guarantee objective truth;
- provide trustworthy verdicts without supplied evidence;
- enforce runtime schemas outside its generated text; or
- implement the browser extension described in the project proposal.

The surrounding prototype now supplies the first-pass search, retrieval, validation, citation, and failure-handling layers. Their limitations—including lightweight HTML extraction, incomplete prompt-injection defenses, and approximate source-independence checks—are documented in `PROTOTYPE_RUNTIME.md`.

The model's internal parametric knowledge must not be treated as evidence. Search, retrieval, source normalization, deterministic validation, citations, and failure handling belong in the surrounding application.

## Evaluation limitations

The automated comparison suite contains 96 examples, 24 per task. One example therefore changes a task rate by 4.17 percentage points. It is useful for iteration but is not a production release gate.

Before treating the model as more than a prototype:

1. Human-review at least 50 examples per task.
2. Create a frozen project-specific evaluation set.
3. Review claim-boundary mismatches semantically rather than only by exact text.
4. Review every verdict confusion.
5. Test with noisy live-web retrieval results.
6. Evaluate source independence, duplicate reporting, and citation correctness.

Complete experiment details and comparisons are recorded in `PROJECT_PROGRESS.md` and `EVALUATION_RESULTS.json`.
