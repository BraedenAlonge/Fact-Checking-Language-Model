# Background Checkers - Project Progress and TODO

Last updated: 2026-11-09

This is the live project record for the Background Checkers data and training work. Update it whenever a dataset, schema, training configuration, evaluation result, or project decision changes.

## Project goal

Train a lightweight model to support a four-stage fact-checking pipeline:

1. Extract independently checkable factual claims while excluding opinions, predictions, commands, and conversational filler.
2. Create concise web-search requests that preserve names, dates, locations, quantities, and attribution.
3. Select faithful, relevant evidence from search results without inventing or rewriting source content.
4. Evaluate each claim as `Supported`, `Contradicted`, or `Insufficient Evidence` using only the supplied evidence.

The intended user-facing result is a preliminary consistency check with inspectable sources, not a claim of objective truth. The runtime should use several independent sources when available and refine inadequate searches.

## Current status

| Area | Status | Notes |
| --- | --- | --- |
| Documentation review | Complete | README, proposal, architecture, dataset plan, schema, dataset catalog, and build notes reviewed. |
| Raw data acquisition | Complete with documented exclusions | Required converter inputs downloaded. ClaimBuster returned HTTP 403 and LLM-AggreFact is gated. |
| Dataset conversion | Complete | Eight train/eval JSONL files generated and count-validated. |
| Tinker environment | Complete | Project-local Python 3.11 environment with Tinker SDK and cookbook. Authentication works. |
| Balanced pilot preparation | Complete | 10,000 train and 1,000 held-out eval examples, balanced equally across the four tasks. |
| Tinker smoke training | Complete | Five LoRA optimizer steps on Qwen3.5-4B completed successfully. |
| Data-goal alignment review | Complete for the pilot | Several important gaps were found; see below. |
| Full pilot training | Not started | Do not start until the priority data fixes and evaluation harness are complete. |

## Work completed

### Documentation and repository context

- Read `README.md` and all text documentation.
- Extracted and reviewed both PDFs:
  - `CSC5870 Project Proposal.pdf`
  - `CSC5870_ System Design and Architecture.pdf`
- Confirmed that this repository contains data acquisition and curation, not the complete browser extension or runtime fact-checking application.
- Confirmed the initial plan to train one multitask LoRA adapter before considering separate task adapters.

### Data acquisition and conversion

- Fixed `scripts/download_raw.py` to download the official FActScore human-annotation `data.zip`, which the converter required but the downloader omitted.
- Downloaded approximately 1.96 GiB of raw public data.
- Validated the FEVER, VitaminC, FActScore, and SciFact archives before conversion.
- Ran `scripts/build_factcheck_data.py` at below-normal local process priority.
- Generated 2,153,148 records, approximately 3.44 GiB total:

| Split/task | Records |
| --- | ---: |
| train/claim_extraction | 11,800 |
| train/search_planning | 98,941 |
| train/evidence_selection | 810,221 |
| train/claim_evaluation | 810,221 |
| eval/claim_extraction | 7,341 |
| eval/search_planning | 12,636 |
| eval/evidence_selection | 200,994 |
| eval/claim_evaluation | 200,994 |

- Verified every output file's line count against `data/processed/stats.json`.
- Verified representative JSON records and task names.
- The expected unavailable sources remain excluded:
  - ClaimBuster: Zenodo returned HTTP 403.
  - LLM-AggreFact: Hugging Face returned HTTP 401 because the dataset is gated.

### Tinker setup

- Installed `uv` 0.13.7.
- Created `.venv` with Python 3.11.18.
- Installed and import-tested:
  - Tinker SDK/CLI 0.39.0
  - Tinker Cookbook 0.5.9
- Verified Tinker credentials and organization access without displaying the API key.
- Verified that `Qwen/Qwen3.5-4B` is trainable and sampleable for the account.
- Verified that there were no existing checkpoint-storage objects before the smoke run.

### Pilot preparation and token sizing

- Added `scripts/prepare_tinker_data.py`.
- Created deterministic, task-balanced pilot files under ignored `data/tinker/`:
  - `pilot_train.jsonl`: 10,000 examples, 2,500 per task.
  - `pilot_eval.jsonl`: 1,000 examples, 250 per task, sampled only from official eval sources.
- Confirmed zero exact-message overlap and zero task/dataset/source-ID overlap between the pilot train and eval files.
- Added `scripts/estimate_tinker_tokens.py`.
- Tokenized locally with the Qwen3.5 non-thinking renderer:
  - Training: 4,007,604 tokens, mean 400.8 tokens/example, maximum 4,043.
  - Evaluation: 290,502 tokens, mean 290.5 tokens/example, maximum 1,970.
  - No pilot example exceeds the configured 8,192-token maximum.
- At the observed published rate of $0.737 per million training tokens, one training epoch over the pilot is approximately $2.95 before evaluation and sampling.

### Tinker smoke run

Configuration:

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-4B` |
| Renderer | `qwen3_5_disable_thinking` |
| LoRA rank | 16 |
| Learning rate | 1e-4 with linear decay |
| Batch size | 16 examples |
| Maximum length | 8,192 tokens |
| Optimizer steps | 5 |
| Held-out NLL subset | 256 examples |

Run results:

- Tinker session: `025f612e-7bd8-5fdf-900c-7052a74c08b0`
- Training completed successfully.
- Training tokens across five steps: 32,685.
- Pre-training held-out evaluation tokens: 72,962.
- Estimated training/evaluation charge: approximately $0.078 before credits. The billing endpoint can lag by several hours.
- Initial held-out metrics, measured before the first optimizer update:
  - NLL: 1.237305
  - bits per byte: 0.239442
- Final resumable state:
  - `tinker://025f612e-7bd8-5fdf-900c-7052a74c08b0:train:0/weights/final`
  - Size: 437,565,045 bytes.
- Final sampling checkpoint:
  - `tinker://025f612e-7bd8-5fdf-900c-7052a74c08b0:train:0/sampler_weights/final`
  - Size: 145,936,229 bytes.
- Both checkpoints are private and currently have no expiration. Together they use approximately 583.5 MB, projecting to roughly $0.054 per month at the published $0.10/GB-month storage rate before credits.
- Local logs:
  - `runs/qwen35-4b-smoke-20261109-011558/`

The initial evaluation occurred at step zero, so it is a baseline measurement and must not be reported as a post-training improvement. The per-batch training losses are also not directly comparable because each batch contains different tasks and sequence lengths.

### Smoke generation comparison

Compared the base model and five-step checkpoint on one held-out example per task. All eight responses parsed as valid JSON.

| Task | Base behavior | Five-step checkpoint behavior | Conclusion |
| --- | --- | --- | --- |
| Claim extraction | Returned a JSON list with the wrong schema and extracted commands as claims. | Returned a `claims` object but still extracted commands and used `claim` instead of `claim_text`. | Schema movement, but behavior remains incorrect. |
| Search planning | Produced a useful query but omitted the required wrapper fields. | Produced the required `claim_id`, `tool`, and `arguments.query` schema and a more informative query than the dataset target. | Clear smoke-level improvement. |
| Evidence selection | Used incorrect field names and `refutes`. | Added claim/source IDs, but used unsupported stance `neutral` and incorrect `query_refinement` field. | Partial schema learning only. |
| Claim evaluation | Returned only the label. | Returned `claim_id` and label, but omitted explanation and `source_ids`. | Partial schema learning only. |

This run validates connectivity, data rendering, optimization, checkpointing, and sampling. Five steps are not enough to judge model quality.

## Data alignment review

The following measurements use the deterministic pilot, which is large enough to expose dominant patterns while avoiding another multi-gigabyte full-corpus parse.

### What aligns well

- The four tasks and their main output labels match the planned pipeline.
- The pilot is balanced at 25% per task.
- The claim-evaluation pilot contains all three verdicts:
  - 1,191 Supported
  - 597 Contradicted
  - 712 Insufficient Evidence
- Selected evidence copying was exact in the audit sample: zero title, URL, or excerpt mismatches.
- Train/eval isolation passed for exact messages and source IDs in the pilot.
- Missing evidence is represented and maps to Insufficient Evidence rather than Contradicted.
- The evaluation claim-extraction subset contains many no-claim cases, which is useful for measuring false positives.
- Real source provenance, source splits, source IDs, and licenses are retained on records.
- No examples exceed the current training context limit.

### Gaps that conflict with project goals

#### Claim extraction

- The 2,500 sampled training records contain zero `{"claims":[]}` outputs. The model is evaluated on no-claim passages but is not trained on them.
- 9.1% of extracted training claims contain a predictive/modal term such as `may`, `might`, `could`, `will`, or `likely`. This heuristic needs human confirmation, but the smoke example proved that commands/advice are currently being extracted as claims.
- 6.4% of extracted claims contain embedded newlines, often joining a headline to the first sentence.
- 31.7% of claim contexts contain a source-generation instruction such as “Summarize the following,” which is not useful runtime context and may teach prompt leakage.
- Training claim extraction is dominated by RAGTruth (82% of the sampled task), while FActScore contributes only 4.7% and FELM only 1.9%.

#### Search planning

- FEVEROUS supplies 92% of sampled training search-planning records.
- Mean query length is only 3.67 words.
- 66.5% of queries are three words or fewer.
- When a claim contains a number, 89.9% of queries omit at least one number.
- A sampled claim about HMCS Sudbury's era, length, and speed was reduced to `HMCS Sudbury`. This directly conflicts with the instruction to preserve dates, quantities, and other necessary context.
- AVeriTeC, ClaimDecomp, and FreshQA better match open-web planning but are heavily underrepresented.

#### Evidence selection

- 75.1% of sampled training records contain only one candidate search result.
- 75.2% select only one result.
- The mean number of candidate results is 1.21, so most records do not teach ranking, rejection, source independence, or comparison among pages.
- 95.9% of selected excerpts are marked high quality; no meaningful low-quality source distribution was observed.
- Duplicate/syndicated-source handling and the “three independent sources when available” behavior are present in the synthetic gold schema examples but not adequately represented in training.
- VitaminC and FEVER dominate this task, making it closer to evidence-conditioned Wikipedia verification than open-web evidence selection.

#### Claim evaluation

- 74.5% of sampled training records use exactly one evidence item.
- 71.2% of explanations use one of three generated templates. This encourages formula copying instead of concise evidence-specific reasoning.
- VitaminC and FEVER account for about 92% of sampled claim-evaluation training records.
- AVeriTeC, WiCE, SciFact, and real multi-source/open-web evidence are underrepresented.
- The label distribution is usable, but dataset and evidence-structure diversity are not yet sufficient for the intended runtime.

#### General risks

- Some source text contains decoding artifacts such as replacement characters and duplicated Wikipedia tokens.
- The full converted corpus is extremely task-imbalanced. The balanced pilot fixes top-level task frequency but not within-task source dominance.
- The current eval collection is derived mostly from public benchmark splits. A smaller project-specific, human-reviewed evaluation set is still required.
- Several source licenses are non-commercial, attribution, or share-alike. Course use is acceptable, but publishing data or a resulting adapter requires a license review.
- `TinkerAPI.txt` exists in the workspace and is gitignored but has not been read. If it contains an API key, delete it now that Tinker's credential store works.

## Decisions

- Start with one multitask adapter, consistent with the dataset plan.
- Use `Qwen/Qwen3.5-4B` for the first experiments because it is lightweight, supports the required context, is supported by the cookbook, and has a low training rate.
- Use the Qwen3.5 non-thinking renderer so the application receives direct structured outputs. The rendered sequence still contains the model family's empty thinking delimiters, but the renderer strips them when parsing the assistant content.
- Train loss only on the final assistant message.
- Keep the official public train/eval split boundary intact.
- Use task-balanced sampling for initial experiments.
- Do not run the full 10,000-example pilot until the priority data problems below are fixed and an end-to-end evaluation script exists.

## Current action items

### Priority 0 - data corrections before the next substantive run

- [ ] Add real training negatives for claim extraction, including opinions, commands, predictions, conversational filler, and passages with no checkable claims.
- [ ] Human-review a stratified sample of modal/predictive claim-extraction outputs and remove or relabel incorrect claims.
- [ ] Remove source-generation instructions from claim context and fix headline/newline concatenation.
- [ ] Redesign or filter FEVEROUS search-planning targets so queries preserve discriminating dates, quantities, locations, and attribution.
- [ ] Increase AVeriTeC, ClaimDecomp, and other open-web-style search-planning representation.
- [ ] Add multi-result evidence-selection examples with irrelevant, duplicate, syndicated, conflicting, low-quality, and partial evidence.
- [ ] Add examples that teach source independence and selecting up to three independent sources when available.
- [ ] Diversify claim-evaluation explanations and reduce the fixed-template rate.
- [ ] Repair or filter obvious text-decoding artifacts and duplicated tokens.
- [ ] Create source-aware and label-aware sampling so VitaminC, FEVER, FEVEROUS, and RAGTruth do not overwhelm smaller but more runtime-relevant datasets.

### Priority 0 - evaluation before a full pilot

- [ ] Implement exact schema validation for model generations, including field names and allowed enums.
- [ ] Measure claim extraction precision, recall, and F1 on a human-reviewed subset.
- [ ] Measure no-claim false-positive rate separately.
- [ ] Define a human relevance score for search queries, including explicit date/number preservation checks.
- [ ] Measure evidence-selection precision and exact excerpt-copy compliance.
- [ ] Measure verdict accuracy, macro F1, and Contradicted-versus-Insufficient-Evidence confusion.
- [ ] Build a fixed base-versus-tuned comparison runner over the same held-out examples.
- [ ] Human-review at least 50 examples per task before the next paid run.
- [ ] Run post-training evaluation after the final step; do not rely only on the cookbook's step-zero NLL.

### Priority 1 - training iteration

- [ ] Regenerate the balanced pilot after the data corrections.
- [ ] Repeat the five-step smoke test and require all four task outputs to pass schema validation.
- [ ] Decide whether to train the 10,000-example pilot for one epoch. Current raw training estimate is about $2.95, excluding evaluations and samples.
- [ ] Compare LoRA rank 16 versus 32 only after the data and evaluator are stable.
- [ ] Record actual billed usage after Tinker's billing endpoint has caught up.
- [ ] Evaluate checkpoint storage and delete obsolete smoke checkpoints when they are no longer useful.

### Priority 1 - runtime application work

- [ ] Define the exact runtime JSON schemas and enforce them in the tool executor.
- [ ] Select an internet search provider and define its normalized result format.
- [ ] Implement page retrieval, content extraction, and inaccessible-page handling.
- [ ] Implement source-domain and source-independence logic outside the model where deterministic rules are safer.
- [ ] Implement query refinement limits and failure handling.
- [ ] Build the end-to-end text-to-cited-results pipeline before browser-extension work.

## Reproduction commands

From the repository root in PowerShell:

```powershell
# Rebuild all local data
python scripts\download_raw.py
python scripts\build_factcheck_data.py

# Recreate the deterministic pilot
.\.venv\Scripts\python.exe scripts\prepare_tinker_data.py

# Recalculate token counts and estimated cost
.\.venv\Scripts\python.exe scripts\estimate_tinker_tokens.py `
  data\tinker\pilot_train.jsonl data\tinker\pilot_eval.jsonl

# Audit the pilot
.\.venv\Scripts\python.exe scripts\audit_tinker_pilot.py

# Print the smoke configuration without spending credits
.\.venv\Scripts\python.exe scripts\train_tinker_sft.py

# A paid run requires the explicit guard
.\.venv\Scripts\python.exe scripts\train_tinker_sft.py --execute
```

## Files added or changed during this work

- `.gitignore`
- `scripts/download_raw.py`
- `scripts/prepare_tinker_data.py`
- `scripts/estimate_tinker_tokens.py`
- `scripts/audit_tinker_pilot.py`
- `scripts/train_tinker_sft.py`
- `scripts/sample_tinker_checkpoint.py`
- `Documentation/DATA_BUILD.txt` (regenerated by the converter)
- `Documentation/PROJECT_PROGRESS.md` (this document)
- `data/processed/stats.json` (regenerated)

Generated and ignored artifacts:

- `data/raw/`
- `data/processed/**/*.jsonl`
- `data/tinker/`
- `.venv/`
- `runs/`
