# End-to-End Prototype Runtime

Last updated: 2026-11-09

The repository now contains a working command-line prototype of the complete Background Checkers pipeline:

```text
input text
  -> claim extraction
  -> search planning
  -> live web search
  -> bounded page retrieval and text extraction
  -> evidence selection
  -> deterministic evidence checks
  -> claim evaluation
  -> cited JSON report
```

The four model stages use the selected repaired checkpoint:

```text
tinker://7e61d5a7-3f20-593b-a4fa-691ca4cb308e:train:0/sampler_weights/000200
```

This is a controlled research prototype, not a production fact-checking service. Its output is a preliminary evidence-consistency check and always needs source inspection for consequential use.

## Main components

| Path | Responsibility |
| --- | --- |
| `scripts/run_prototype.py` | Guarded command-line entry point and JSON report writer. |
| `background_checkers/model.py` | Tinker sampler lifecycle, task prompts, output parsing, and one schema-correction retry. |
| `background_checkers/schemas.py` | Exact runtime validation for all four task outputs. |
| `background_checkers/search.py` | DDGS and Brave search providers, URL normalization, and result deduplication. |
| `background_checkers/retrieval.py` | Bounded concurrent retrieval, public-address checks, HTML text extraction, and relevant excerpts. |
| `background_checkers/pipeline.py` | Claim-by-claim orchestration, refinement limits, evidence contracts, citations, and audit report. |
| `tests/test_prototype.py` | Offline tests for schemas, fallbacks, source integrity, and the complete orchestration path. |

## Install and authenticate

The existing project environment already contains Tinker. To recreate the prototype dependencies:

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-prototype.txt
```

Authenticate without putting a Tinker key in the repository:

```powershell
.\.venv\Scripts\tinker.exe auth login
.\.venv\Scripts\tinker.exe auth status
```

The default search provider is DDGS and needs no key. It is convenient for a demo but can fail transiently. For a more stable provider, obtain a Brave Search API key and set it only in the current shell:

```powershell
$env:BRAVE_SEARCH_API_KEY = "your-key"
```

The `.env.example` file documents the variable name. The runtime does not automatically load `.env`, and `.env` is gitignored.

## Run the prototype

First print the plan without making paid Tinker or live search calls:

```powershell
.\.venv\Scripts\python.exe scripts\run_prototype.py `
  --text "The Eiffel Tower was completed in 1889."
```

Then explicitly allow execution:

```powershell
.\.venv\Scripts\python.exe scripts\run_prototype.py `
  --execute `
  --text "The Eiffel Tower was completed in 1889."
```

Use Brave when configured:

```powershell
.\.venv\Scripts\python.exe scripts\run_prototype.py `
  --execute --provider brave `
  --file .\input.txt `
  --output .\prototype_outputs\input-report.json
```

The default output location is `prototype_outputs/report-<timestamp>.json`. That directory is ignored because reports can contain the submitted text, retrieved excerpts, and service identifiers.

Useful limits include `--max-claims`, `--search-results`, `--max-refinements`, `--max-evidence`, `--fetch-concurrency`, `--fetch-timeout`, and `--max-page-bytes`. `--no-fetch` provides a lower-confidence snippet-only mode. The defaults bound model calls, network work, page size, and concurrency so one input cannot fan out without limit.

## Runtime integrity rules

The application does not trust model-generated citations directly. It:

- rejects model-selected source IDs that were not retrieved;
- restores titles, URLs, and excerpts from the retrieval record instead of accepting rewritten model fields;
- keeps at most one item per source domain and at most three evidence items by default;
- prefers successfully fetched page text over unverified search snippets;
- validates redirects and rejects private, loopback, link-local, and otherwise non-public destinations;
- makes only one search refinement by default and records every attempt;
- isolates a failed claim so other claims can still complete; and
- forces `Insufficient Evidence` when no usable evidence remains.

The JSON report preserves search queries, retrieval status, model decisions, dropped evidence counts, errors, verdicts, and citations for review.

## Verification performed

Run all offline checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The first live end-to-end smoke test processed “The Eiffel Tower was completed in 1889,” searched the web, retrieved usable Wikipedia and Britannica page text, and returned a supported verdict with citations. An official site blocked retrieval and fell back to its search snippet; the runtime was subsequently tightened so fetched page evidence is preferred and that fallback is not selected when fetched evidence exists.

## Known limitations and next steps

- HTML extraction is deliberately lightweight; PDFs, JavaScript-only pages, paywalls, and complex layouts need specialized handling.
- Prompt-injection text inside retrieved pages is not yet classified or stripped beyond treating page content as data in the prompts.
- A formal robots/publisher-policy layer and cache/archive policy are not implemented.
- Domain uniqueness is only a practical approximation of editorial independence; syndicated or commonly owned sources can still appear independent.
- DDGS is an unofficial convenience provider. Brave requires a separate account, key, and its own usage/cost review.
- The interface is a guarded CLI and JSON report, not yet a local web UI or browser extension.
- Model performance still needs a frozen, human-reviewed project evaluation set. Runtime infrastructure does not remove the need to review claim boundaries, query relevance, evidence fidelity, and verdicts.

The next application layer can wrap `PrototypePipeline` in a small local API/UI without changing the task schemas or report format.
