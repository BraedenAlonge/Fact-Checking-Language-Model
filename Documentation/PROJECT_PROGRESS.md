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

## Executive assessment

The project has moved beyond data acquisition and connectivity testing into a measured research prototype. The complete local data pipeline is reproducible, two full pilot adaptations have run, and a fixed stratified evaluator now compares the base model and multiple checkpoints. The repaired step-200 checkpoint is the current candidate model.

What is working:

- structured outputs are generally learned reliably;
- the repaired model recognizes most no-claim inputs instead of turning every utterance into a factual claim;
- search queries retain more discriminating context;
- evidence source selection and exact copying are strong on the current suite; and
- periodic checkpoints and task-level evaluation now prevent blindly choosing the final training step.

What is not yet ready:

- claim boundaries remain difficult and automated exact-match F1 is modest;
- verdict performance regressed relative to the first pilot on the small fixed suite;
- evidence data is still more benchmark/Wikipedia-like than a real open-web workflow;
- the evaluation suite is automated and only 96 examples, not a human-reviewed release gate; and
- the search, retrieval, page-extraction, source-independence, citation, and browser-extension runtime is not implemented in this repository.

The current result is suitable for experiment iteration and manual demos, but not yet for unattended or user-facing factuality decisions. The next highest-value work is error review and a human-scored project-specific evaluation set, followed by targeted data repair. Another training run should wait until those steps identify a concrete failure distribution.

The selected checkpoint's supported behaviors, measured task performance, example schemas, intended application workflow, and explicit non-capabilities are summarized in `Documentation/MODEL_CAPABILITIES.md`.

## Current status

| Area | Status | Notes |
| --- | --- | --- |
| Documentation review | Complete | README, proposal, architecture, dataset plan, schema, dataset catalog, and build notes reviewed. |
| Raw data acquisition | Complete with one exclusion | Public converter inputs are local. LLM-AggreFact remains gated. |
| Dataset conversion | Complete, first repair pass | 2,075,390 records regenerated after removing weak targets and adding ClaimBuster. |
| Tinker environment | Complete | Project-local Python 3.11 environment; authentication and credits work. |
| Tinker smoke test | Complete | Five LoRA optimizer steps validated the complete workflow. |
| Preliminary training | Complete | Original and repaired 625-step pilots completed for about $3.33 and $3.60 before credits. |
| Post-run data repair | Complete, first pass | Claim negatives, source balancing, query filtering, and prompt-leak fixes applied. |
| Evaluation harness | Automated preliminary coverage complete | Fixed 96-example stratified suite compares schemas and task metrics; human review remains. |

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

## Expanded evaluation and repaired-data training

### Stratified evaluation suite

Added `scripts/evaluate_tinker_checkpoint.py`, guarded by `--execute`. It selects 96 deterministic held-out examples, 24 per task, and stratifies them by:

- source plus empty/nonempty output for claim extraction;
- source plus numeric/nonnumeric claim for search planning;
- source plus single/multiple candidate results for evidence selection; and
- source plus verdict label for claim evaluation.

It reports strict nested-schema validity, exact JSON, claim micro-F1, no-claim false positives and accuracy, query token overlap and number preservation, selected-source F1, exact evidence copying, stance/quality accuracy, refinement accuracy, verdict accuracy/macro-F1/confusion, and source-ID F1. Predictions and summaries are retained in ignored run directories for error analysis. The scorer was self-tested against gold outputs, which produce perfect scores on all gold-aligned metrics.

### Second one-epoch run

The repaired source-balanced pilot was trained with the same model and hyperparameters as the first run for a controlled comparison. Periodic state and sampler checkpoints were saved every 100 steps.

- Tinker session: `7e61d5a7-3f20-593b-a4fa-691ca4cb308e`.
- Training tokens: 4,280,978.
- Evaluation: 85,500 tokens per evaluation, seven evaluations, 598,500 total.
- Estimated billed tokens: 4,879,478.
- Estimated charge: approximately $3.60 before credits and sampling.
- Local logs: `runs/qwen35-4b-repaired-pilot-20261109-125133/`.
- Final sampler checkpoint: `tinker://7e61d5a7-3f20-593b-a4fa-691ca4cb308e:train:0/sampler_weights/final`.
- Selected checkpoint: `tinker://7e61d5a7-3f20-593b-a4fa-691ca4cb308e:train:0/sampler_weights/000200`.

Held-out NLL:

| Step | NLL |
| ---: | ---: |
| 0 | 1.1260 |
| 100 | 0.2215 |
| 200 | **0.1511** |
| 300 | 0.1590 |
| 400 | 0.1668 |
| 500 | 0.1562 |
| 600 | 0.1596 |

The repaired run learned much faster and overfit much less severely than the original run. Step 200 is the NLL minimum, and later checkpoints remain close instead of degrading to the original run's 0.6603 final-region NLL.

### Fixed-suite model comparison

All values below use the exact same 96 examples. The first run was trained on the original pilot; the three repaired checkpoints were trained on the patched pilot.

| Metric | Base | First final | Repaired 200 | Repaired 500 | Repaired final |
| --- | ---: | ---: | ---: | ---: | ---: |
| Claim strict schema | 4.2% | 100% | 95.8% | 95.8% | 95.8% |
| Claim micro-F1 | 0.000 | 0.460 | 0.416 | 0.253 | 0.452 |
| No-claim accuracy (9 cases) | 33.3% | 0% | **88.9%** | 77.8% | 66.7% |
| No-claim false-positive rate | 0%* | 100% | **0%** | 11.1% | 22.2% |
| Search strict schema | 0% | 100% | 100% | 100% | 100% |
| Query token F1 | 0.000 | **0.616** | 0.611 | 0.589 | 0.596 |
| Target-number preservation | 66.7% | 79.2% | 83.3% | 83.3% | 83.3% |
| Evidence source-ID F1 | 0.755 | 0.760 | **1.000** | **1.000** | **1.000** |
| Evidence exact-copy rate | 59.4% | 95.9% | **100%** | **100%** | **100%** |
| Evidence stance accuracy | 10.0% | **93.3%** | 83.3% | 80.0% | 86.7% |
| Verdict accuracy | 79.2% | **87.5%** | 79.2% | 83.3% | 79.2% |
| Verdict macro-F1 | 0.796 | **0.877** | 0.793 | 0.842 | 0.796 |
| Verdict source-ID F1 | 0.167 | **1.000** | **1.000** | **1.000** | **1.000** |

`*` The base model often emitted invalid claim schemas, so its apparent no-claim false-positive rate is not success; its no-claim accuracy was only 33.3%.

Step 200 is selected as the best balanced repaired checkpoint. It has the best held-out NLL, the highest repaired no-claim accuracy, zero false-positive claims on valid negative cases, strong query behavior, and perfect evidence source selection/copying. Step 500 improves verdict macro-F1 but sharply lowers claim-extraction F1. The final checkpoint improves evidence exact-match details but loses two negative extraction cases. This selection prioritizes avoiding fabricated checkable claims, a central project requirement.

## Remaining risks and limitations

- Numeric preservation still fails in 43.1% of sampled numeric search targets, mostly outside the newly filtered FEVEROUS slice. The audit treats every number as necessary, so human review should distinguish real failures from irrelevant numbers.
- Claim extraction still needs human review of predictions, opinions, attribution, and multi-claim boundaries. ClaimBuster sentences are short debate utterances, not full LLM responses.
- About half of evidence-selection training rows still have one candidate. Ranking, rejecting duplicates, source independence, and low-quality-source handling remain underrepresented.
- About half of evaluation explanations still use generated templates, and most benchmark evidence is Wikipedia-like rather than open-web multi-source evidence.
- Some source text still contains decoding artifacts or duplicated tokens.
- The eval set is benchmark-derived. A smaller project-specific, human-reviewed set is required.
- The 96-example automated suite is useful for iteration but too small for definitive model selection; one example changes a task rate by 4.17 percentage points.
- Repaired training improved extraction negatives and evidence selection but reduced verdict accuracy versus the first checkpoint on this suite. Verdict data and label confusions need targeted review.
- Dataset licenses include attribution, share-alike, and non-commercial terms. Publishing the data or adapter requires a license review.
- Smoke and pilot checkpoints currently have no expiration and therefore incur small ongoing storage costs. Do not delete them until the user decides whether they are still useful.
- The two earlier checkpoint pairs plus seven repaired-run pairs are estimated at roughly 5.25 GB total, or about $0.53/month at $0.10/GB-month, until TTLs are set or obsolete artifacts are deleted.

## Decisions

- Continue with one multitask adapter for now.
- Continue using `Qwen/Qwen3.5-4B` and its non-thinking renderer for structured outputs.
- Train only on the final assistant message.
- Preserve official public train/eval boundaries; use a stable hash only when a source lacks a split.
- Balance both tasks and source datasets for pilot runs.
- Treat step 200 as evidence for early checkpoint selection, not as a universal optimum. The repaired data needs its own learning curve.
- Use repaired step 200 as the current candidate checkpoint, while retaining step 500 and final for error analysis.
- Do not start another paid run until the 96-example errors and a human-reviewed project-specific set are analyzed.

## Current action items

### Priority 0 - evaluation before another paid run

- [x] Implement exact schema validation, including field names and allowed enums.
- [x] Build a fixed base-versus-tuned runner over a stratified held-out set.
- [x] Measure automated claim-extraction F1 and no-claim false-positive rate.
- [x] Measure query-target overlap and date/number preservation.
- [x] Measure evidence-selection source F1 and exact excerpt-copy compliance.
- [x] Measure verdict accuracy, macro F1, and label confusion.
- [ ] Define human search-relevance and claim-boundary rubrics.
- [ ] Human-review at least 50 examples per task.
- [x] Select the best periodic checkpoint by held-out metrics and task scores.

### Priority 0 - remaining data improvements

- [ ] Review numeric-query omissions and patch AVeriTeC/ClaimDecomp targets where necessary.
- [ ] Human-review a stratified sample of extraction negatives, modal claims, and claim boundaries.
- [ ] Add multi-result evidence examples with irrelevant, duplicate, syndicated, conflicting, low-quality, and partial evidence.
- [ ] Add explicit source-independence examples and selecting up to three independent sources when available.
- [ ] Diversify evidence-specific evaluation explanations and reduce fixed templates.
- [ ] Repair or filter obvious text-decoding artifacts and duplicated tokens.

### Priority 1 - next training iteration

- [x] Regenerate the source-balanced pilot after the first data repair.
- [x] Run structural and task-level evaluation against the base and existing tuned checkpoint.
- [x] Run the repaired pilot with `--save-every 100`.
- [ ] Review every error in the 96-example suite, especially verdict confusion and claim-boundary mismatches.
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

# Plan a 96-example-per-model stratified evaluation without API calls
.\.venv\Scripts\python.exe scripts\evaluate_tinker_checkpoint.py `
  --include-base --output-dir runs\evaluation-plan
```

## Generated and ignored artifacts

- `data/raw/`
- `data/processed/**/*.jsonl`
- `data/tinker/`
- `.venv/`
- `runs/`

These artifacts remain local because they are multi-gigabyte, reproducible, may carry source-license obligations, or may contain private credentials and service identifiers. They are intentionally not force-added to Git. Tracked rebuild metadata and reproducibility records include `Documentation/DATA_BUILD.txt`, `Documentation/PILOT_MANIFEST.json`, `Documentation/EVALUATION_RESULTS.json`, `data/processed/stats.json`, and the scripts under `scripts/`.
