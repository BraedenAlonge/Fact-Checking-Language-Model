# Background Checkers

This repository contains the data, model experiments, evaluation harness, and end-to-end command-line prototype for Background Checkers, a CSC 5870 project exploring how a lightweight language model can help users review factual claims in LLM-generated or other source text.

The prototype now extracts checkable claims, searches the web, retrieves source pages, selects evidence, evaluates each claim, and writes cited audit reports. It is a controlled CLI prototype rather than the proposed browser extension or a production factuality service.

## Quick start: run the model

Run commands from the repository root in PowerShell. The supported environment is Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-prototype.txt
.\.venv\Scripts\tinker.exe auth login
.\.venv\Scripts\tinker.exe auth status
```

First run the offline tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The safest way to use the selected model is through the complete pipeline. Without `--execute`, the command only prints its plan and makes no Tinker or live-search calls:

```powershell
.\.venv\Scripts\python.exe scripts\run_prototype.py `
  --text "The Eiffel Tower was completed in 1889."
```

After reviewing the plan, explicitly allow the paid Tinker and live-search calls:

```powershell
.\.venv\Scripts\python.exe scripts\run_prototype.py `
  --execute `
  --text "The Eiffel Tower was completed in 1889."
```

For a longer input, use `--file` and optionally choose the output path:

```powershell
.\.venv\Scripts\python.exe scripts\run_prototype.py `
  --execute `
  --file .\input.txt `
  --output .\prototype_outputs\input-report.json
```

The default DDGS search provider needs no key. To use Brave Search instead, set `BRAVE_SEARCH_API_KEY` in the current shell and add `--provider brave`. Prototype reports are written under `prototype_outputs/` by default. See `Documentation/PROTOTYPE_RUNTIME.md` for every runtime option, safety rule, and limitation.

## Dataset scope

The planned dataset is a multi-task JSONL collection covering four stages of the proposed fact-checking pipeline:

1. **Claim extraction** - identify independently checkable factual claims while excluding opinions, predictions, commands, and conversational filler.
2. **Search planning** - turn each claim into a concise web-search request that preserves important context such as names, dates, and locations.
3. **Evidence selection** - identify relevant excerpts from retrieved sources, retain source metadata, assess evidence stance and quality, and request search refinement when results are inadequate.
4. **Claim evaluation** - classify a claim as `Supported`, `Contradicted`, or `Insufficient Evidence` using only the supplied evidence.

Each JSONL line is a complete chat-style training record. The assistant response is serialized JSON so that a fine-tuned model can learn both the task behavior and the required machine-readable output schema.

## Current status

Public datasets have been converted into the four tasks. Training and evaluation JSONL files are in `data/processed/`. Counts and the mapping rules are in `Documentation/DATA_BUILD.txt`.

| Split | Claim extraction | Search planning | Evidence selection | Claim evaluation |
| --- | ---: | ---: | ---: | ---: |
| Train | 22,983 | 18,907 | 810,221 | 810,221 |
| Eval | 7,966 | 3,104 | 200,994 | 200,994 |

The 25-example gold set remains a schema and behavior review file. Its evidence is synthetic and its URLs use the reserved `.example` domain. It is not part of the training files.

## How to navigate the repository

Use this table as the starting point:

| If you want to... | Start here |
| --- | --- |
| Run the complete fact-checking demo | `scripts/run_prototype.py` and `Documentation/PROTOTYPE_RUNTIME.md` |
| Understand what the selected checkpoint can do | `Documentation/MODEL_CAPABILITIES.md` |
| See completed work and the current TODO list | `Documentation/PROJECT_PROGRESS.md` |
| Inspect the runtime implementation | `background_checkers/pipeline.py`, then `model.py`, `search.py`, and `retrieval.py` |
| Understand the JSON task contracts | `Preliminary_Data/GOLD_SET_SCHEMA.txt` and `background_checkers/schemas.py` |
| Rebuild or inspect the datasets | `Documentation/DATA_BUILD.txt`, `scripts/download_raw.py`, and `scripts/build_factcheck_data.py` |
| Prepare or audit a Tinker training pilot | `scripts/prepare_tinker_data.py` and `scripts/audit_tinker_pilot.py` |
| Train or evaluate a checkpoint | `scripts/train_tinker_sft.py` and `scripts/evaluate_tinker_checkpoint.py` |
| Run offline regression tests | `tests/test_prototype.py` |

The main directory structure is:

```text
Documentation/
  CSC5870 Project Proposal.pdf
  CSC5870_ System Design and Architecture.pdf
  DATASET_PLAN.txt
  USEFUL_DATASETS.txt
  DATA_BUILD.txt
  PROJECT_PROGRESS.md
  MODEL_CAPABILITIES.md
  PROTOTYPE_RUNTIME.md
  PILOT_MANIFEST.json
  EVALUATION_RESULTS.json
Preliminary_Data/
  GOLD_SET_SCHEMA.txt
  gold_examples_v1.jsonl
data/processed/
  stats.json
  train/
  eval/
scripts/
  download_raw.py
  build_factcheck_data.py
  prepare_tinker_data.py
  audit_tinker_pilot.py
  estimate_tinker_tokens.py
  train_tinker_sft.py
  evaluate_tinker_checkpoint.py
  sample_tinker_checkpoint.py
  run_prototype.py
background_checkers/
  schemas.py
  model.py
  search.py
  retrieval.py
  pipeline.py
tests/
  test_prototype.py
requirements-prototype.txt
README.md
```

- `Documentation/` contains the broader project proposal, system design, dataset plan, the catalog of public datasets, and `DATA_BUILD.txt`, which records how those datasets were converted.
- `data/processed/` contains the converted train and eval JSONL. The JSONL files are large and are gitignored; `stats.json` records the counts. Rebuild them through the project environment with `.\.venv\Scripts\python.exe scripts\download_raw.py` and `.\.venv\Scripts\python.exe scripts\build_factcheck_data.py`.
- `Documentation/PROJECT_PROGRESS.md` is the live record of acquisition, audits, Tinker runs, decisions, and current action items.
- `Documentation/MODEL_CAPABILITIES.md` describes what the selected prototype checkpoint can and cannot do, with measured task-level results and example schemas.
- `Documentation/PROTOTYPE_RUNTIME.md` explains the complete CLI pipeline, installation, search-provider choices, safety rules, usage, and current limitations.
- `Preliminary_Data/GOLD_SET_SCHEMA.txt` defines the shared record format, task-specific output schemas, labeling rules, and review checklist.
- `Preliminary_Data/gold_examples_v1.jsonl` contains the initial 25 examples.

## Script guide and common workflows

Every script should be run from the repository root with `.\.venv\Scripts\python.exe`. Use `--help` to inspect all available arguments.

| Script | Purpose | External work or cost |
| --- | --- | --- |
| `run_prototype.py` | Run text through extraction, search, retrieval, evidence selection, evaluation, and citation reporting. | Plan-only by default; `--execute` enables Tinker and live search. |
| `download_raw.py` | Download the public source datasets used by the converters. | Network and substantial disk use. |
| `build_factcheck_data.py` | Convert local raw sources into the four processed task datasets. | Local CPU, memory, and disk use. |
| `prepare_tinker_data.py` | Build deterministic, source-balanced pilot train/eval JSONL files. | Local only. |
| `audit_tinker_pilot.py` | Check pilot balance, overlap, copying, prompt leakage, and other quality signals. | Local only. |
| `estimate_tinker_tokens.py` | Tokenize prepared JSONL and estimate training cost. | Local only; the estimate is not a billing quote. |
| `train_tinker_sft.py` | Plan or launch a Tinker LoRA training run. | Plan-only by default; `--execute` starts a paid run. |
| `evaluate_tinker_checkpoint.py` | Build the fixed evaluation suite, rescore saved predictions, or sample models. | Plan/rescoring is local; `--execute` starts paid sampling. |
| `sample_tinker_checkpoint.py` | Compare base and tuned outputs on four examples. | **No execution guard:** invoking it immediately makes Tinker sampling calls. |

### Rebuild and audit training data

The raw and generated JSONL data is intentionally gitignored. Reproduce it with:

```powershell
.\.venv\Scripts\python.exe scripts\download_raw.py
.\.venv\Scripts\python.exe scripts\build_factcheck_data.py
.\.venv\Scripts\python.exe scripts\prepare_tinker_data.py
.\.venv\Scripts\python.exe scripts\audit_tinker_pilot.py
```

`download_raw.py` downloads roughly two gigabytes, and conversion can use meaningful CPU, memory, and disk. Do not start those steps casually on a resource-constrained machine.

### Estimate, train, and evaluate

Estimate tokens and cost before training:

```powershell
.\.venv\Scripts\python.exe scripts\estimate_tinker_tokens.py `
  data\tinker\pilot_train.jsonl data\tinker\pilot_eval.jsonl
```

Print a training plan without creating a session:

```powershell
.\.venv\Scripts\python.exe scripts\train_tinker_sft.py `
  --max-steps 625 --eval-every 100 --save-every 100
```

Print the fixed-suite evaluation plan without sampling:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tinker_checkpoint.py `
  --include-base `
  --checkpoint "tinker://your-checkpoint" `
  --output-dir runs\evaluation-plan
```

Only add `--execute` after reviewing the printed plan, confirming the checkpoint and data paths, and deciding that the expected Tinker cost is acceptable. Training/evaluation outputs go under the ignored `runs/` directory.

## Data-quality principles

Dataset acquisition and curation should:

- preserve enough context to keep each claim's original meaning;
- include negative, boundary, and adversarial cases rather than only straightforward successes;
- copy evidence excerpts faithfully and never invent sources, titles, URLs, or quotations;
- distinguish independent sources from duplicate or syndicated pages;
- prefer authoritative and primary sources where appropriate;
- label missing, partial, weak, ambiguous, irrelevant, or conflicting evidence as `Insufficient Evidence` rather than `Contradicted`;
- validate every record for JSON syntax and schema compliance;
- detect duplicate and near-duplicate examples;
- record how examples were acquired, generated, and reviewed; and
- keep a human-reviewed evaluation set isolated from training data.

The labels describe agreement with the retrieved evidence or online consensus, not guaranteed objective truth. Source provenance and user-visible citations are therefore essential parts of the data design.

## Acquisition roadmap

1. Review and approve the current gold-set schemas and labeling decisions.
2. Implement automated JSON and schema validation.
3. Convert the useful public datasets catalogued in `Documentation/USEFUL_DATASETS.txt`. Use each source's full training split, and keep archived evidence text with its provenance.
4. Human-review the label mapping and a sample from each converted dataset. The public sources are too large to review line by line.
5. Check task coverage, label mapping, source independence, and duplicates across sources.
6. Freeze each source's development and test split, plus Factcheck-Bench, before model training begins.

The converted files are the training and evaluation data. `Documentation/DATA_BUILD.txt` lists the counts and the schema adaptations. The 25 gold examples remain a schema check. A slice is left out when it cannot be mapped onto the four tasks and the three labels, not in order to keep the collection small.
