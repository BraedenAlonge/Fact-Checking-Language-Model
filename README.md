# Background Checkers

This repository contains the data, model experiments, evaluation harness, and end-to-end command-line prototype for Background Checkers, a CSC 5870 project exploring how a lightweight language model can help users review factual claims in LLM-generated or other source text.

The prototype now extracts checkable claims, searches the web, retrieves source pages, selects evidence, evaluates each claim, and writes cited audit reports. It is a controlled CLI prototype rather than the proposed browser extension or a production factuality service.

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

## Repository contents

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
