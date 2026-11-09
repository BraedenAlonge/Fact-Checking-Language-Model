# Background Checkers - Data Acquisition

This repository contains the **data-acquisition and dataset-curation work** for Background Checkers, a CSC 5870 project exploring how a lightweight language model can help users review factual claims in LLM-generated or other source text.

It is not the complete fact-checking application. The broader project is intended to extract checkable claims, search the web for relevant sources, select useful evidence, and compare each claim with that evidence. This repository focuses on acquiring and preparing the high-quality training and evaluation data needed to support those behaviors.

## Dataset scope

The planned dataset is a multi-task JSONL collection covering four stages of the proposed fact-checking pipeline:

1. **Claim extraction** - identify independently checkable factual claims while excluding opinions, predictions, commands, and conversational filler.
2. **Search planning** - turn each claim into a concise web-search request that preserves important context such as names, dates, and locations.
3. **Evidence selection** - identify relevant excerpts from retrieved sources, retain source metadata, assess evidence stance and quality, and request search refinement when results are inadequate.
4. **Claim evaluation** - classify a claim as `Supported`, `Contradicted`, or `Insufficient Evidence` using only the supplied evidence.

Each JSONL line is a complete chat-style training record. The assistant response is serialized JSON so that a fine-tuned model can learn both the task behavior and the required machine-readable output schema.

## Current status

The repository currently includes a 25-example, human-review-oriented gold set:

| Task | Examples |
| --- | ---: |
| Claim extraction | 7 |
| Search planning | 5 |
| Evidence selection | 6 |
| Claim evaluation | 7 |

These examples are intended for **schema and behavior review**. Evidence packets are synthetic but internally consistent, and their URLs use the reserved `.example` domain. The current set must not be treated as a factual web-evidence corpus.

After the schemas, task boundaries, and labeling rules are approved, the evidence-selection and claim-evaluation data should be rebuilt or expanded with archived content from real sources, including retrieval dates and provenance.

## Repository contents

```text
Documentation/
  CSC5870 Project Proposal.pdf
  CSC5870_ System Design and Architecture.pdf
  DATASET_PLAN.txt
  USEFUL_DATASETS.txt
Preliminary_Data/
  GOLD_SET_SCHEMA.txt
  gold_examples_v1.jsonl
README.md
```

- `Documentation/` contains the broader project proposal, system design, dataset plan, and the catalog of public datasets to convert.
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

The 25 gold examples are a schema check, not a size target. Training uses the full useful portion of the public datasets. A slice is left out when it cannot be mapped onto the four tasks and the three labels, not in order to keep the collection small.
