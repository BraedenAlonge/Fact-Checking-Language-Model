# Background Checkers - Project Progress and TODO

Last updated: 2026-11-09

This is the live project record for the Background Checkers data and training work. Update it whenever a dataset, schema, training configuration, evaluation result, or project decision changes.

## Project goal

Train a lightweight model to support a four-stage fact-checking pipeline:

1. Extract independently checkable factual claims while excluding opinions, predictions, commands, and conversational filler.
2. Create concise web-search requests that preserve names, dates, locations, quantities, and attribution.
3. Select faithful, relevant evidence from search results without inventing or rewriting source content.
4. Evaluate each claim as `Supported`, `Contradicted`, or `Insufficient Evidence` using only the supplied evidence.

The intended product is a preliminary consistency check with inspectable sources, not a claim of objective truth. The runtime should use several independent sources when available and refine inadequate searches.

## Current status

| Area | Status | Notes |
| --- | --- | --- |
| Documentation review | Complete | README, proposal, architecture, dataset plan, schema, dataset catalog, and build notes reviewed. |
| Raw data acquisition | Complete with one exclusion | Public converter inputs are local. LLM-AggreFact remains gated. |
| Dataset conversion | Complete, first repair pass | 2,075,390 records regenerated after removing weak targets and adding ClaimBuster. |
| Tinker environment | Complete | Project-local Python 3.11 environment; authentication and credits work. |
| Tinker smoke test | Complete | Five LoRA optimizer steps validated the complete workflow. |
| Preliminary training | Complete | A 625-step, one-epoch pilot completed for about $3.33 before credits. |
| Post-run data repair | Complete, first pass | Claim negatives, source balancing, query filtering, and prompt-leak fixes applied. |
| Evaluation harness | Partial | Structural audit and fixed four-example comparison exist; task metrics and human review remain. |

## Work completed

### Repository and documentation

- Read `README.md` and the text documentation.
- Extracted and reviewed both project PDFs.
- Confirmed that this repository covers data acquisition and curation, not yet the browser extension or complete fact-checking runtime.
- Confirmed the initial plan to train one multitask LoRA adapter before considering task-specific adapters.
- Pushed the pre-training snapshot to `origin/main` as commit `b2e6a9b`.
- Added this live progress document and kept generated data, credentials, virtual environments, and run logs out of Git.

### Data acquisition and conversion

- Fixed the downloader to fetch the FActScore human-annotation archive required by the converter.
- Downloaded approximately 1.96 GiB of public raw data.
- Added the 23,533-row ClaimBuster Parquet mirror after Zenodo's file host was inaccessible from this machine. Mirror metadata identifies the license as CC BY-SA 4.0.
- ClaimBuster's public mirror has no official split. A stable text hash now reserves about 10% for evaluation and keeps identical text in one split.
- LLM-AggreFact remains excluded because its Hugging Face dataset is gated.
- Rebuilt and count-validated the processed collection at below-normal process priority:

| Split/task | Records |
| --- | ---: |
| train/claim_extraction | 22,983 |
| train/search_planning | 18,907 |
| train/evidence_selection | 810,221 |
| train/claim_evaluation | 810,221 |
| eval/claim_extraction | 7,966 |
| eval/search_planning | 3,104 |
| eval/evidence_selection | 200,994 |
| eval/claim_evaluation | 200,994 |

### Tinker setup

- Installed `uv` 0.13.7.
- Created `.venv` with Python 3.11.18.
- Installed and import-tested Tinker SDK/CLI 0.39.0 and Tinker Cookbook 0.5.9.
- Verified credentials, organization access, and that `Qwen/Qwen3.5-4B` is trainable and sampleable.
- Never displayed or read the API key. `TinkerAPI.txt` is gitignored but should be deleted if it contains a redundant plaintext key.

### Initial pilot and smoke test

- Created deterministic 10,000-row train and 1,000-row eval pilots, balanced across the four tasks.
- The original pilot contained 4,007,604 training tokens and 290,502 eval tokens.
- Five-step smoke session: `025f612e-7bd8-5fdf-900c-7052a74c08b0`.
- Smoke cost estimate: about $0.078 before credits.
- Smoke checkpoints:
  - `tinker://025f612e-7bd8-5fdf-900c-7052a74c08b0:train:0/weights/final`
  - `tinker://025f612e-7bd8-5fdf-900c-7052a74c08b0:train:0/sampler_weights/final`
- Base and tuned generations all parsed as JSON. Five steps improved schema use but were not enough for reliable task behavior.

### Preliminary one-epoch training

The user authorized a preliminary paid run after the initial repository state was pushed.

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-4B` |
| Renderer | `qwen3_5_disable_thinking` |
| Data | Original 10,000-row task-balanced pilot, before the repair pass |
| LoRA rank | 16 |
| Learning rate | 1e-4 with linear decay |
| Batch size | 16 |
| Maximum length | 8,192 tokens |
| Steps | 625, one epoch |
| Evaluation | 256 held-out examples every 100 steps |

- Tinker session: `b51a81da-3eb7-5530-a680-13b43e96acfe`.
- Training tokens: 4,007,604.
- Evaluation tokens: 510,734 over seven evaluations.
- Estimated charge: about $3.33 before credits; the billing endpoint may lag.
- Local logs: `runs/qwen35-4b-pilot-20261109-0125/`.
- Final checkpoints:
  - `tinker://b51a81da-3eb7-5530-a680-13b43e96acfe:train:0/weights/final`
  - `tinker://b51a81da-3eb7-5530-a680-13b43e96acfe:train:0/sampler_weights/final`

Held-out NLL by evaluation step:

| Step | NLL |
| ---: | ---: |
| 0 | 1.2373 |
| 100 | 0.6337 |
| 200 | **0.5523** |
| 300 | 0.5727 |
| 400 | 0.6338 |
| 500 | 0.6264 |
| 600 | 0.6603 |

The run learned quickly, then plateaued and overfit after approximately step 200. Only the final checkpoint was saved, so the best step cannot be recovered. Future substantive runs should save every 100 steps and select the best held-out checkpoint rather than automatically using the final one.

The post-training four-example comparison is at `runs/qwen35-4b-pilot-20261109-0125/sample_comparison.jsonl`. The tuned model exactly matched the expected search-planning, evidence-selection, and claim-evaluation examples. It still extracted six advice/command statements from a no-claim example. This failure directly motivated the claim-extraction repair.

### First post-training data repair

- Added ClaimBuster factual and non-factual sentences:
  - labels 0 and 1 become one-claim examples;
  - label 2 becomes `{"claims":[]}`;
  - 13,844 non-factual examples are in train and 1,572 in eval.
- Removed RAGTruth from claim extraction because it labels hallucination spans, not factual claim boundaries. It remains useful for evidence selection and evaluation.
- Removed FELM writing-recommendation segments from claim extraction because they mix advice, slogans, opinions, and facts without claim-boundary labels.
- Normalized whitespace in extracted RAGTruth sentences and replaced source-generation prompts with neutral context.
- Filtered FEVEROUS query targets to at least four word tokens. For claims containing numbers, retained only targets preserving every number.
  - kept: 11,961;
  - removed as too short: 73,862;
  - removed for omitted numbers: 18,434.
- Changed pilot preparation from task-only balancing to equal round-robin sampling across source datasets within each task.
- Corrected the overlap audit to include the official source split because AVeriTeC reuses integer IDs independently in train and dev.
- Added optional `--save-every` and `--session-name` arguments to the guarded Tinker runner.

## Current patched-pilot audit

The rebuilt pilot remains 10,000 train and 1,000 eval examples, balanced equally across the four tasks. It is now also source-aware within each task.

| Signal | Before repair | After repair |
| --- | ---: | ---: |
| Empty claim-extraction outputs in 2,500 train rows | 0 | 627 |
| Predictive/modal extraction claims | 9.1% | 1.32% |
| Extraction claims containing newlines | 6.4% | 0% |
| Extraction contexts containing source instructions | 31.7% | 0% |
| Search queries of three words or fewer | 66.5% | 1.8% |
| Mean search-query length | 3.67 words | 8.50 words |
| Numeric claims whose target omits a number | 89.9% | 43.1% |
| Evidence-selection rows with one candidate | 75.1% | 49.4% |
| Evaluation rows using a fixed explanation template | 71.2% | 52.5% |
| Selected-evidence copy mismatches | 0 | 0 |
| Exact train/eval message overlap | 0 | 0 |
| Full task/dataset/split/source-ID overlap | 0 | 0 |

Patched pilot sizing:

- Training: 4,280,978 tokens; mean 428.1; maximum 6,416; no 8,192-token overflows.
- Evaluation: 364,251 tokens; mean 364.3; maximum 3,959; no overflows.
- One training epoch at $0.737 per million tokens: approximately $3.16 before evaluation.
- With the same evaluation cadence as the completed run, a repeat is expected to cost roughly $3.5-$3.6 before credits.

The data was not catastrophically bad, and the completed run proved that it can teach the schemas and three of the four sampled behaviors. It is materially better after this repair, but it is not yet ready to support a trustworthy end-user fact checker without a project-specific evaluator and further evidence/query improvements.

## Remaining risks and limitations

- Numeric preservation still fails in 43.1% of sampled numeric search targets, mostly outside the newly filtered FEVEROUS slice. The audit treats every number as necessary, so human review should distinguish real failures from irrelevant numbers.
- Claim extraction still needs human review of predictions, opinions, attribution, and multi-claim boundaries. ClaimBuster sentences are short debate utterances, not full LLM responses.
- About half of evidence-selection training rows still have one candidate. Ranking, rejecting duplicates, source independence, and low-quality-source handling remain underrepresented.
- About half of evaluation explanations still use generated templates, and most benchmark evidence is Wikipedia-like rather than open-web multi-source evidence.
- Some source text still contains decoding artifacts or duplicated tokens.
- The eval set is benchmark-derived. A smaller project-specific, human-reviewed set is required.
- Dataset licenses include attribution, share-alike, and non-commercial terms. Publishing the data or adapter requires a license review.
- Smoke and pilot checkpoints currently have no expiration and therefore incur small ongoing storage costs. Do not delete them until the user decides whether they are still useful.

## Decisions

- Continue with one multitask adapter for now.
- Continue using `Qwen/Qwen3.5-4B` and its non-thinking renderer for structured outputs.
- Train only on the final assistant message.
- Preserve official public train/eval boundaries; use a stable hash only when a source lacks a split.
- Balance both tasks and source datasets for pilot runs.
- Treat step 200 as evidence for early checkpoint selection, not as a universal optimum. The repaired data needs its own learning curve.
- Do not start another paid run until the evaluator can score more than four hand-selected examples.

## Current action items

### Priority 0 - evaluation before another paid run

- [ ] Implement exact schema validation, including field names and allowed enums.
- [ ] Build a fixed base-versus-tuned runner over a representative held-out set.
- [ ] Measure claim-extraction precision, recall, F1, and no-claim false-positive rate.
- [ ] Define and measure search relevance plus required name/date/number preservation.
- [ ] Measure evidence-selection precision and exact excerpt-copy compliance.
- [ ] Measure verdict accuracy, macro F1, and Contradicted-versus-Insufficient-Evidence confusion.
- [ ] Human-review at least 50 examples per task.
- [ ] Select the best periodic checkpoint by held-out metrics and task scores.

### Priority 0 - remaining data improvements

- [ ] Review numeric-query omissions and patch AVeriTeC/ClaimDecomp targets where necessary.
- [ ] Human-review a stratified sample of extraction negatives, modal claims, and claim boundaries.
- [ ] Add multi-result evidence examples with irrelevant, duplicate, syndicated, conflicting, low-quality, and partial evidence.
- [ ] Add explicit source-independence examples and selecting up to three independent sources when available.
- [ ] Diversify evidence-specific evaluation explanations and reduce fixed templates.
- [ ] Repair or filter obvious text-decoding artifacts and duplicated tokens.

### Priority 1 - next training iteration

- [x] Regenerate the source-balanced pilot after the first data repair.
- [ ] Run structural and task-level evaluation against the base and existing tuned checkpoint.
- [ ] If evaluation is satisfactory, run the repaired pilot with `--save-every 100`.
- [ ] Compare LoRA rank 16 versus 32 only after data and evaluation are stable.
- [ ] Record actual billed usage after Tinker's billing endpoint catches up.
- [ ] Decide whether to set checkpoint TTLs or delete obsolete smoke artifacts.

### Priority 1 - runtime application

- [ ] Define and enforce exact runtime JSON schemas.
- [ ] Select a search provider and normalized result format.
- [ ] Implement page retrieval, content extraction, and inaccessible-page handling.
- [ ] Implement deterministic source-domain and independence rules outside the model.
- [ ] Implement query-refinement limits and failure handling.
- [ ] Build the end-to-end text-to-cited-results pipeline before browser-extension work.

## Reproduction commands

From the repository root in PowerShell:

```powershell
# Download/rebuild local data (the .venv supplies pyarrow for ClaimBuster)
.\.venv\Scripts\python.exe scripts\download_raw.py
.\.venv\Scripts\python.exe scripts\build_factcheck_data.py

# Recreate and audit the deterministic source-balanced pilot
.\.venv\Scripts\python.exe scripts\prepare_tinker_data.py
.\.venv\Scripts\python.exe scripts\audit_tinker_pilot.py

# Recalculate token counts and estimated cost
.\.venv\Scripts\python.exe scripts\estimate_tinker_tokens.py `
  data\tinker\pilot_train.jsonl data\tinker\pilot_eval.jsonl

# Print the guarded smoke configuration without spending credits
.\.venv\Scripts\python.exe scripts\train_tinker_sft.py

# Example future paid pilot; requires an explicit --execute guard
.\.venv\Scripts\python.exe scripts\train_tinker_sft.py `
  --execute --max-steps 625 --eval-every 100 --save-every 100
```

## Generated and ignored artifacts

- `data/raw/`
- `data/processed/**/*.jsonl`
- `data/tinker/`
- `.venv/`
- `runs/`

Tracked rebuild metadata and scripts include `Documentation/DATA_BUILD.txt`, `data/processed/stats.json`, and the scripts under `scripts/`.
