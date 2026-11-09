"""Convert downloaded fact-checking datasets into the project's four tasks.

The output is chat-style JSONL. Development and test splits, FEVER 2.0,
Factcheck-Bench, and FreshQA are written to data/processed/eval. Training
splits are written to data/processed/train.

Run scripts/download_raw.py first. This script only reads data/raw.
"""

from __future__ import annotations

import csv
import json
import re
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

LABELS = ("Supported", "Contradicted", "Insufficient Evidence")

PROMPTS = {
    "claim_extraction": (
        "Extract independently checkable factual claims. Exclude opinions, "
        "predictions, commands, and conversational filler. Split distinct "
        "assertions when they can be checked separately. Preserve necessary "
        "context. Return valid JSON only."
    ),
    "search_planning": (
        "Create one concise web-search tool call for the supplied claim. "
        "Preserve names, dates, places, quantities, and attribution needed to "
        "verify the claim. Do not answer the claim. Return valid JSON only."
    ),
    "evidence_selection": (
        "Select only evidence relevant to the supplied claim. Copy titles, "
        "URLs, and excerpts exactly from the supplied results. Assign stance "
        "and quality. Do not invent information. Mark whether query refinement "
        "is needed. Return valid JSON only."
    ),
    "claim_evaluation": (
        "Evaluate the claim using only the supplied evidence. Use exactly one "
        "label: Supported, Contradicted, or Insufficient Evidence. Supported "
        "requires the complete claim to be supported. Contradicted requires "
        "reliable direct conflict. Otherwise use Insufficient Evidence. "
        "Return valid JSON only."
    ),
}

VERDICT = {
    "supports": "Supported",
    "supported": "Supported",
    "support": "Supported",
    "refutes": "Contradicted",
    "refuted": "Contradicted",
    "contradict": "Contradicted",
    "not enough info": "Insufficient Evidence",
    "not enough information": "Insufficient Evidence",
    "not enough evidence": "Insufficient Evidence",
    "nei": "Insufficient Evidence",
    "conflicting evidence/cherrypicking": "Insufficient Evidence",
    "partially_supported": "Insufficient Evidence",
    "not_supported": "Insufficient Evidence",
}

NOTES: list[str] = []


def clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut or text[:limit]).rstrip() + "..."


def detok(text: str) -> str:
    return (
        text.replace("-LRB-", "(")
        .replace("-RRB-", ")")
        .replace("-LSB-", "[")
        .replace("-RSB-", "]")
        .replace("-LCB-", "{")
        .replace("-RCB-", "}")
        .replace("-COLON-", ":")
    )


def quality_for(url: str, default: str = "medium") -> str:
    host = (url or "").lower()
    high = (
        "wikipedia.org",
        ".gov",
        ".edu",
        "who.int",
        "nih.gov",
        "nature.com",
        "sciencemag.org",
        "thelancet.com",
        "nejm.org",
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "bbc.co.uk",
    )
    if any(piece in host for piece in high):
        return "high"
    return default


def host_title(url: str, fallback: str) -> str:
    if not url:
        return fallback
    host = urlparse(url).netloc
    return host or fallback


def verdict_of(label: str) -> str | None:
    return VERDICT.get(re.sub(r"\s+", " ", (label or "")).strip().lower())


def quote_explanation(label: str, evidence: list[dict]) -> str:
    if label == "Insufficient Evidence" and not evidence:
        return "No evidence was supplied for this claim."
    if not evidence:
        return "The supplied evidence does not settle the complete claim."
    snippet = clip(evidence[0]["excerpt"], 220)
    source = evidence[0]["source_id"]
    if label == "Supported":
        return f"The supplied evidence supports the complete claim. {source} states: \"{snippet}\""
    if label == "Contradicted":
        return f"The supplied evidence directly conflicts with the claim. {source} states: \"{snippet}\""
    return f"The supplied evidence does not settle the complete claim. {source} states: \"{snippet}\""


class Store:
    def __init__(self) -> None:
        self.handles: dict[tuple[str, str], object] = {}
        self.counts: Counter = Counter()
        self.verdicts: Counter = Counter()
        self.seen: set[str] = set()

    def add(self, bucket: str, record: dict) -> None:
        key = record["example_id"]
        if key in self.seen:
            record["example_id"] = f"{key}-{self.counts[(bucket, record['task'])]}"
        self.seen.add(record["example_id"])
        slot = (bucket, record["task"])
        if slot not in self.handles:
            path = OUT / bucket / f"{record['task']}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handles[slot] = path.open("w", encoding="utf-8", newline="\n")
        self.handles[slot].write(json.dumps(record, ensure_ascii=False) + "\n")
        self.counts[slot] += 1
        assistant = json.loads(record["messages"][2]["content"])
        if record["task"] == "claim_evaluation":
            self.verdicts[(bucket, record["source_dataset"], assistant["label"])] += 1

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def record(
    store: Store,
    *,
    bucket: str,
    task: str,
    dataset: str,
    split: str,
    source_id: str,
    license_name: str,
    user,
    assistant: dict,
    tags: list[str] | None = None,
) -> None:
    user_text = user if isinstance(user, str) else json.dumps(user, ensure_ascii=False)
    built = {
        "example_id": f"{dataset}-{split}-{source_id}-{task}",
        "task": task,
        "source_dataset": dataset,
        "source_split": split,
        "source_id": str(source_id),
        "license": license_name,
        "tags": tags or [],
        "messages": [
            {"role": "system", "content": PROMPTS[task]},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
        ],
    }
    if task == "evidence_selection":
        pool = {item["excerpt"] for item in user["search_results"]}
        for item in assistant["evidence"]:
            if item["excerpt"] not in pool:
                raise ValueError(f"excerpt was not copied from results: {source_id}")
            if item["stance"] not in ("supports", "contradicts", "context"):
                raise ValueError(item["stance"])
            if item["quality"] not in ("high", "medium", "low"):
                raise ValueError(item["quality"])
    if task == "claim_evaluation":
        if assistant["label"] not in LABELS:
            raise ValueError(assistant["label"])
        ids = {item["source_id"] for item in user["evidence"]}
        if not set(assistant["source_ids"]).issubset(ids):
            raise ValueError(f"source id missing from evidence: {source_id}")
    if task == "search_planning":
        if not assistant["arguments"]["query"].strip():
            return
    if task == "claim_extraction":
        for claim in assistant["claims"]:
            if not claim["claim_text"].strip():
                return
    store.add(bucket, built)


def emit_verification(
    store: Store,
    *,
    bucket: str,
    dataset: str,
    split: str,
    source_id: str,
    license_name: str,
    claim_text: str,
    context: str,
    results: list[dict],
    selected: list[dict],
    label: str,
    explanation: str,
    needs_refinement: bool,
    refinement_reason: str | None,
    tags: list[str],
) -> None:
    claim = {"claim_id": "c1", "claim_text": claim_text, "context": context}
    record(
        store,
        bucket=bucket,
        task="evidence_selection",
        dataset=dataset,
        split=split,
        source_id=source_id,
        license_name=license_name,
        user={"claim": claim, "search_results": results},
        assistant={
            "claim_id": "c1",
            "evidence": selected,
            "needs_refinement": needs_refinement,
            "refinement_reason": refinement_reason,
        },
        tags=tags,
    )
    record(
        store,
        bucket=bucket,
        task="claim_evaluation",
        dataset=dataset,
        split=split,
        source_id=source_id,
        license_name=license_name,
        user={
            "claim": claim,
            "evidence": [
                {
                    "source_id": item["source_id"],
                    "excerpt": item["excerpt"],
                    "quality": item["quality"],
                }
                for item in selected
            ],
        },
        assistant={
            "claim_id": "c1",
            "label": label,
            "explanation": clip(explanation, 700),
            "source_ids": [item["source_id"] for item in selected],
        },
        tags=tags,
    )


def result(source_id: str, title: str, url: str, excerpt: str, source_type: str) -> dict:
    return {
        "source_id": source_id,
        "title": title or host_title(url, "Source"),
        "url": url or "",
        "source_type": source_type,
        "excerpt": excerpt,
    }


def selected_from(result_item: dict, stance: str, quality: str | None = None) -> dict:
    return {
        "source_id": result_item["source_id"],
        "title": result_item["title"],
        "url": result_item["url"],
        "excerpt": result_item["excerpt"],
        "stance": stance,
        "quality": quality or quality_for(result_item["url"]),
    }


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def convert_averitec(store: Store) -> None:
    license_name = "CC BY-NC 4.0"
    buckets = {"train": "train", "dev": "eval", "test": "eval"}
    skipped_unlabeled = 0
    for split, bucket in buckets.items():
        rows = json.loads((RAW / "averitec" / f"{split}.json").read_text(encoding="utf-8"))
        for index, row in enumerate(rows):
            claim_text = (row.get("claim") or "").strip()
            if not claim_text:
                continue
            label = verdict_of(row.get("label") or "")
            if label is None:
                skipped_unlabeled += 1
                continue
            context_bits = [
                bit
                for bit in (
                    f"Speaker: {row.get('speaker')}" if row.get("speaker") else "",
                    f"Date: {row.get('claim_date')}" if row.get("claim_date") else "",
                    f"Location: {row.get('location_ISO_code')}" if row.get("location_ISO_code") else "",
                )
                if bit
            ]
            context = ". ".join(context_bits) or "Real-world claim checked against web sources."
            questions = []
            for question in row.get("questions") or []:
                text = (question.get("question") or "").strip()
                if text:
                    questions.append(text)
            for q_index, query in enumerate(dict.fromkeys(questions)):
                record(
                    store,
                    bucket=bucket,
                    task="search_planning",
                    dataset="averitec",
                    split=split,
                    source_id=f"{index}-q{q_index}",
                    license_name=license_name,
                    user={"claim_id": "c1", "claim_text": claim_text, "context": context},
                    assistant={
                        "claim_id": "c1",
                        "tool": "web_search",
                        "arguments": {"query": query},
                    },
                    tags=["averitec", "annotated_question"],
                )
            results = []
            seen_urls = set()
            for question in row.get("questions") or []:
                q_text = (question.get("question") or "").strip()
                for answer in question.get("answers") or []:
                    if (answer.get("answer_type") or "").lower() == "unanswerable":
                        continue
                    answer_text = (answer.get("answer") or "").strip()
                    if not answer_text:
                        continue
                    url = (answer.get("source_url") or answer.get("cached_source_url") or "").strip()
                    if url and url in seen_urls:
                        continue
                    if url:
                        seen_urls.add(url)
                    excerpt = answer_text if len(answer_text) > 12 or not q_text else f"{q_text} — {answer_text}"
                    results.append(
                        result(
                            f"s{len(results) + 1}",
                            host_title(url, "Web source"),
                            url,
                            clip(excerpt, 700),
                            answer.get("source_medium") or "web",
                        )
                    )
            selected = [selected_from(item, "context") for item in results]
            reason = None
            refine = label == "Insufficient Evidence" or len(results) == 0
            if refine:
                reason = "The retrieved answers do not settle the claim." if results else "No usable source answers were recorded."
            explanation = (row.get("justification") or "").strip() or quote_explanation(label, selected)
            emit_verification(
                store,
                bucket=bucket,
                dataset="averitec",
                split=split,
                source_id=str(index),
                license_name=license_name,
                claim_text=claim_text,
                context=context,
                results=results,
                selected=selected,
                label=label,
                explanation=explanation,
                needs_refinement=refine,
                refinement_reason=reason,
                tags=["averitec", label.lower().replace(" ", "_")],
            )
    if skipped_unlabeled:
        NOTES.append(
            "AVeriTeC's test file has "
            f"{skipped_unlabeled} claims and no verdicts or evidence, so those claims were not converted. Train and dev were converted."
        )


def fever_evidence_sets(evidence) -> list[list[tuple[str, int]]]:
    sets = []
    if not isinstance(evidence, list):
        return sets
    for evidence_set in evidence:
        if not isinstance(evidence_set, list):
            continue
        items = []
        for item in evidence_set:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            page, sid = item[2], item[3]
            if not page or sid is None:
                continue
            try:
                items.append((str(page), int(sid)))
            except (TypeError, ValueError):
                continue
        if items:
            sets.append(items)
    return sets


def load_fever_rows() -> list[tuple[str, str, dict]]:
    specs = [
        ("train", "train", RAW / "fever" / "train.jsonl"),
        ("dev", "eval", RAW / "fever" / "shared_task_dev.jsonl"),
        ("fever2", "eval", RAW / "fever" / "fever2-fixers-dev.jsonl"),
    ]
    rows = []
    for split, bucket, path in specs:
        for row in iter_jsonl(path):
            if (row.get("claim") or "").strip():
                rows.append((split, bucket, row))
    return rows


def build_fever_sentence_index(needed: dict[str, set[int]]) -> dict[tuple[str, int], str]:
    cache = RAW / "fever" / "needed_sentences.jsonl"
    found: dict[tuple[str, int], str] = {}
    if cache.exists() and cache.stat().st_size > 0:
        for row in iter_jsonl(cache):
            found[(row["page"], int(row["sid"]))] = row["text"]
        missing = [
            (page, sid)
            for page, sids in needed.items()
            for sid in sids
            if (page, sid) not in found
        ]
        if not missing:
            print(f"FEVER sentence cache hit ({len(found)} sentences)", flush=True)
            return found
    found = {}
    zip_path = RAW / "fever" / "wiki-pages.zip"
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.startswith("wiki-pages/wiki-") and name.endswith(".jsonl")]
        for name in tqdm(members, desc="FEVER wiki"):
            with archive.open(name) as handle:
                for raw_line in handle:
                    page = json.loads(raw_line)
                    page_id = page.get("id") or ""
                    wanted = needed.get(page_id)
                    if not wanted or not page.get("lines"):
                        continue
                    for line in page["lines"].split("\n"):
                        if "\t" not in line:
                            continue
                        idx, text = line.split("\t", 1)
                        if not idx.isdigit() or int(idx) not in wanted:
                            continue
                        cleaned = detok(text).strip()
                        if cleaned:
                            found[(page_id, int(idx))] = cleaned
    with cache.open("w", encoding="utf-8", newline="\n") as handle:
        for (page, sid), text in found.items():
            handle.write(json.dumps({"page": page, "sid": sid, "text": text}, ensure_ascii=False) + "\n")
    print(f"FEVER sentences kept: {len(found)}", flush=True)
    return found


def convert_fever(store: Store) -> None:
    license_name = "CC BY-SA 3.0"
    rows = load_fever_rows()
    needed: dict[str, set[int]] = defaultdict(set)
    for _split, _bucket, row in rows:
        for evidence_set in fever_evidence_sets(row.get("evidence")):
            for page, sid in evidence_set:
                needed[page].add(sid)
    sentences = build_fever_sentence_index(needed)
    missed = 0
    for split, bucket, row in rows:
        label = verdict_of(row.get("label") or "")
        if label is None:
            continue
        claim_text = row["claim"].strip()
        sets = fever_evidence_sets(row.get("evidence"))
        resolved_sets = []
        for evidence_set in sets:
            resolved = []
            complete = True
            for page, sid in evidence_set:
                text = sentences.get((page, sid))
                if text:
                    resolved.append((page, sid, text))
                else:
                    missed += 1
                    complete = False
            if complete and resolved:
                resolved_sets.append(resolved)
        tags = ["fever" if split != "fever2" else "fever2"]
        if split == "fever2":
            tags.append("adversarial")
        if not resolved_sets:
            if label != "Insufficient Evidence":
                continue
            emit_verification(
                store,
                bucket=bucket,
                dataset="fever",
                split=split,
                source_id=str(row.get("id")),
                license_name=license_name,
                claim_text=claim_text,
                context="Claim checked against English Wikipedia.",
                results=[],
                selected=[],
                label=label,
                explanation="No Wikipedia evidence was supplied for this claim.",
                needs_refinement=True,
                refinement_reason="No evidence sentences were available.",
                tags=tags + ["empty_evidence"],
            )
            continue
        for set_index, evidence_set in enumerate(resolved_sets):
            results = []
            for page, _sid, text in evidence_set:
                title = page.replace("_", " ")
                results.append(
                    result(
                        f"s{len(results) + 1}",
                        title,
                        f"https://en.wikipedia.org/wiki/{page}",
                        clip(text, 700),
                        "wikipedia",
                    )
                )
            stance = {"Supported": "supports", "Contradicted": "contradicts"}.get(label, "context")
            selected = [selected_from(item, stance, "high") for item in results]
            refine = label == "Insufficient Evidence"
            emit_verification(
                store,
                bucket=bucket,
                dataset="fever",
                split=split,
                source_id=f"{row.get('id')}-e{set_index}",
                license_name=license_name,
                claim_text=claim_text,
                context="Claim checked against English Wikipedia.",
                results=results,
                selected=selected,
                label=label,
                explanation=quote_explanation(label, selected),
                needs_refinement=refine,
                refinement_reason="The Wikipedia sentences do not settle the claim." if refine else None,
                tags=tags,
            )
    if missed:
        NOTES.append(f"FEVER evidence sentences missing from the Wikipedia dump: {missed}. Those sentences were left out.")


def convert_vitaminc(store: Store) -> None:
    license_name = "CC BY-SA 3.0"
    buckets = {"train": "train", "dev": "eval", "test": "eval"}
    archive = zipfile.ZipFile(RAW / "vitaminc" / "vitaminc.zip")
    label_sets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for split in buckets:
        with archive.open(f"vitaminc/{split}.jsonl") as handle:
            for raw_line in handle:
                row = json.loads(raw_line)
                label = verdict_of(row.get("label") or "")
                claim = (row.get("claim") or "").strip()
                if label and claim:
                    label_sets[(split, claim)].add(label)
    mixed = {key for key, labels in label_sets.items() if "Supported" in labels and "Contradicted" in labels}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for split, bucket in buckets.items():
        with archive.open(f"vitaminc/{split}.jsonl") as handle:
            for raw_line in handle:
                row = json.loads(raw_line)
                label = verdict_of(row.get("label") or "")
                claim = (row.get("claim") or "").strip()
                evidence = clip(row.get("evidence") or "", 700)
                page = (row.get("page") or "Wikipedia").strip()
                if not label or not claim or not evidence:
                    continue
                title = page.replace("_", " ")
                url = "https://en.wikipedia.org/wiki/" + page.replace(" ", "_")
                item = result("s1", title, url, evidence, "wikipedia")
                stance = {"Supported": "supports", "Contradicted": "contradicts"}.get(label, "context")
                chosen = selected_from(item, stance, "high")
                refine = label == "Insufficient Evidence"
                tags = ["vitaminc", row.get("revision_type") or "revision"]
                emit_verification(
                    store,
                    bucket=bucket,
                    dataset="vitaminc",
                    split=split,
                    source_id=str(row.get("unique_id")),
                    license_name=license_name,
                    claim_text=claim,
                    context=f"Wikipedia page: {title}.",
                    results=[item],
                    selected=[chosen],
                    label=label,
                    explanation=quote_explanation(label, [chosen]),
                    needs_refinement=refine,
                    refinement_reason="This Wikipedia text does not settle the claim." if refine else None,
                    tags=tags,
                )
                if (split, claim) in mixed and label in ("Supported", "Contradicted"):
                    bucket_rows = grouped[(split, claim)]
                    if sum(1 for saved in bucket_rows if saved["label"] == label) < 2:
                        bucket_rows.append({"label": label, "item": item, "bucket": bucket})
    for conflict_index, ((split, claim), items) in enumerate(grouped.items()):
        results = []
        selected = []
        for saved in items:
            copied = dict(saved["item"])
            copied["source_id"] = f"s{len(results) + 1}"
            results.append(copied)
            stance = "supports" if saved["label"] == "Supported" else "contradicts"
            selected.append(selected_from(copied, stance, "high"))
        emit_verification(
            store,
            bucket=items[0]["bucket"],
            dataset="vitaminc",
            split=split,
            source_id=f"conflict-{conflict_index}",
            license_name=license_name,
            claim_text=claim,
            context="The same claim is paired with Wikipedia texts that disagree.",
            results=results,
            selected=selected,
            label="Insufficient Evidence",
            explanation="The supplied Wikipedia texts disagree, so they do not settle the claim.",
            needs_refinement=True,
            refinement_reason="The sources conflict.",
            tags=["vitaminc", "conflicting_evidence"],
        )
    archive.close()


def wice_keep(sentence: str, supporting: bool) -> bool:
    lowered = sentence.strip().lower()
    if supporting:
        return True
    if lowered.startswith("(meta data)") or lowered in {
        "sections",
        "you are not signed in.",
        "share this article",
        "contributing editor",
    }:
        return False
    return len(sentence.split()) >= 8


def wice_pool(evidence: list[str], support_indexes: list[int]) -> tuple[list[dict], list[dict]]:
    support = set(support_indexes)
    ordered = []
    for index in support_indexes:
        if 0 <= index < len(evidence):
            ordered.append(index)
    for index, sentence in enumerate(evidence):
        if index in support:
            continue
        if wice_keep(sentence, False):
            ordered.append(index)
        if len(ordered) >= 12:
            break
    results = []
    selected = []
    for index in ordered[:12]:
        sentence = re.sub(r"\s+", " ", evidence[index]).strip()
        if not sentence:
            continue
        item = result(f"s{len(results) + 1}", "Cited web page", "", clip(sentence, 500), "web")
        results.append(item)
        if index in support:
            selected.append(item)
    return results, selected


def convert_wice(store: Store) -> None:
    license_name = "ODC-BY"
    buckets = {"train": "train", "dev": "eval", "test": "eval"}
    for split, bucket in buckets.items():
        subclaims = defaultdict(list)
        for row in iter_jsonl(RAW / "wice" / "subclaim" / f"{split}.jsonl"):
            meta = row.get("meta") or {}
            parent = str(meta.get("id") or "").rsplit("-", 1)[0]
            text = (row.get("claim") or "").strip()
            if parent and text:
                suffix = str(meta.get("id") or "").rsplit("-", 1)[-1]
                subclaims[parent].append((int(suffix) if suffix.isdigit() else 0, text, meta))
        for row in iter_jsonl(RAW / "wice" / "claim" / f"{split}.jsonl"):
            meta = row.get("meta") or {}
            parent = str(meta.get("id") or "")
            claim_text = (row.get("claim") or "").strip()
            if not claim_text:
                continue
            context_text = (meta.get("claim_context") or "").strip()
            title = meta.get("claim_title") or "Wikipedia"
            section = meta.get("claim_section") or ""
            passage = f"{context_text}\n{claim_text}".strip() if context_text else claim_text
            parts = sorted(subclaims.get(parent, []))
            claims = [
                {
                    "claim_id": f"c{index + 1}",
                    "claim_text": text,
                    "context": f"{title}. {section}".strip(),
                }
                for index, (_order, text, _meta) in enumerate(parts)
            ]
            if not claims:
                claims = [{"claim_id": "c1", "claim_text": claim_text, "context": f"{title}. {section}".strip()}]
            record(
                store,
                bucket=bucket,
                task="claim_extraction",
                dataset="wice",
                split=split,
                source_id=parent or claim_text[:40],
                license_name=license_name,
                user=passage,
                assistant={"claims": claims},
                tags=["wice", "claim_splitting"],
            )
            label = verdict_of(row.get("label") or "")
            if label is None:
                continue
            support_sets = row.get("supporting_sentences") or []
            support_indexes = []
            if support_sets and isinstance(support_sets[0], list):
                support_indexes = [int(idx) for idx in support_sets[0]]
            evidence = [str(item) for item in (row.get("evidence") or [])]
            results, chosen_results = wice_pool(evidence, support_indexes)
            stance = "supports" if label == "Supported" else "context"
            selected = [selected_from(item, stance, "medium") for item in chosen_results]
            refine = label != "Supported"
            reason = None
            if row.get("label") == "partially_supported":
                reason = "The cited text supports only part of the claim."
            elif row.get("label") == "not_supported":
                reason = "The cited text does not support the claim."
            elif refine:
                reason = "The cited text does not settle the claim."
            emit_verification(
                store,
                bucket=bucket,
                dataset="wice",
                split=split,
                source_id=parent or claim_text[:40],
                license_name=license_name,
                claim_text=claim_text,
                context=f"{title}. {section}".strip(),
                results=results,
                selected=selected,
                label=label,
                explanation=quote_explanation(label, selected),
                needs_refinement=refine,
                refinement_reason=reason,
                tags=["wice", row.get("label") or "wice"],
            )


def convert_scifact(store: Store) -> None:
    license_name = "CC BY-NC 2.0"
    corpus = {}
    with tarfile.open(RAW / "scifact" / "data.tar.gz") as archive:
        corpus_handle = archive.extractfile("data/corpus.jsonl")
        for raw_line in corpus_handle:
            row = json.loads(raw_line)
            corpus[str(row["doc_id"])] = row
        files = {"train": "data/claims_train.jsonl", "dev": "data/claims_dev.jsonl"}
        for split, name in files.items():
            bucket = "train" if split == "train" else "eval"
            handle = archive.extractfile(name)
            for raw_line in handle:
                row = json.loads(raw_line)
                claim_text = (row.get("claim") or "").strip()
                if not claim_text:
                    continue
                evidence = row.get("evidence") or {}
                rationale = {}
                for doc_id, rationales in evidence.items():
                    for item in rationales or []:
                        stance_label = (item.get("label") or "").upper()
                        for sentence_id in item.get("sentences") or []:
                            rationale.setdefault(str(doc_id), []).append((int(sentence_id), stance_label))
                results = []
                selected = []
                saw_support = False
                saw_contradict = False
                for doc_id in row.get("cited_doc_ids") or list(evidence):
                    doc = corpus.get(str(doc_id))
                    if not doc:
                        continue
                    abstract = doc.get("abstract") or []
                    marked = {sid: label for sid, label in rationale.get(str(doc_id), [])}
                    url = f"https://www.semanticscholar.org/paper/{doc_id}"
                    for sid, sentence in enumerate(abstract):
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        item = result(f"s{len(results) + 1}", doc.get("title") or "Scientific abstract", url, sentence, "scientific abstract")
                        results.append(item)
                        if sid in marked:
                            stance_label = marked[sid]
                            if stance_label == "SUPPORT":
                                saw_support = True
                                selected.append(selected_from(item, "supports", "high"))
                            elif stance_label == "CONTRADICT":
                                saw_contradict = True
                                selected.append(selected_from(item, "contradicts", "high"))
                conflict = saw_support and saw_contradict
                if conflict:
                    label = "Insufficient Evidence"
                elif saw_support:
                    label = "Supported"
                elif saw_contradict:
                    label = "Contradicted"
                else:
                    label = "Insufficient Evidence"
                refine = conflict or not selected
                reason = None
                if conflict:
                    reason = "The abstracts disagree."
                elif not selected:
                    reason = "The cited abstracts do not settle the claim."
                emit_verification(
                    store,
                    bucket=bucket,
                    dataset="scifact",
                    split=split,
                    source_id=str(row.get("id")),
                    license_name=license_name,
                    claim_text=claim_text,
                    context="Scientific claim checked against cited abstracts.",
                    results=results,
                    selected=selected,
                    label=label,
                    explanation=quote_explanation(label, selected),
                    needs_refinement=refine,
                    refinement_reason=reason,
                    tags=["scifact"],
                )


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    position = 0
    for part in re.split(r"(?<=[.!?])\s+", text):
        if not part:
            continue
        found = text.find(part, position)
        if found < 0:
            found = position
        if len(part.split()) >= 6:
            spans.append((found, found + len(part), part.strip()))
        position = found + len(part)
    return spans


def rag_label_for(span: tuple[int, int], annotations: list[dict]) -> str:
    start, end = span
    kinds = set()
    for annotation in annotations:
        if annotation.get("end", 0) <= start or annotation.get("start", 0) >= end:
            continue
        kinds.add(annotation.get("label_type") or "")
    if "Evident Conflict" in kinds:
        return "Contradicted"
    if kinds:
        return "Insufficient Evidence"
    return "Supported"


def convert_ragtruth(store: Store) -> None:
    license_name = "MIT"
    sources = {}
    for row in iter_jsonl(RAW / "ragtruth" / "source_info.jsonl"):
        sources[str(row["source_id"])] = row
    for row in iter_jsonl(RAW / "ragtruth" / "response.jsonl"):
        source = sources.get(str(row.get("source_id")))
        if not source or source.get("task_type") not in ("QA", "Summary"):
            continue
        response = row.get("response") or ""
        spans = sentence_spans(response)
        if not spans:
            continue
        bucket = "train" if row.get("split") == "train" else "eval"
        split = row.get("split") or "train"
        prompt = clip(source.get("prompt") or "", 240)
        claims = [
            {"claim_id": f"c{index + 1}", "claim_text": text, "context": prompt or source.get("task_type")}
            for index, (_start, _end, text) in enumerate(spans)
        ]
        record(
            store,
            bucket=bucket,
            task="claim_extraction",
            dataset="ragtruth",
            split=split,
            source_id=str(row.get("id")),
            license_name=license_name,
            user=response,
            assistant={"claims": claims},
            tags=["ragtruth", source.get("task_type"), source.get("source") or "source"],
        )
        passage = clip(str(source.get("source_info") or ""), 1600)
        if not passage:
            continue
        quality = "high" if source.get("task_type") == "Summary" else "medium"
        for index, (start, end, text) in enumerate(spans):
            label = rag_label_for((start, end), row.get("labels") or [])
            stance = {"Supported": "supports", "Contradicted": "contradicts"}.get(label, "context")
            item = result("s1", source.get("source") or "Retrieved source", "", passage, source.get("task_type") or "retrieved")
            chosen = [selected_from(item, stance, quality)]
            refine = label != "Supported"
            if label == "Supported":
                explanation = "The retrieved source supports this sentence, and the sentence was not marked as conflicting with or missing from that source."
            elif label == "Contradicted":
                explanation = "The retrieved source directly conflicts with this sentence."
            else:
                explanation = "The retrieved source does not provide enough evidence for this sentence."
            emit_verification(
                store,
                bucket=bucket,
                dataset="ragtruth",
                split=split,
                source_id=f"{row.get('id')}-s{index}",
                license_name=license_name,
                claim_text=text,
                context=prompt or "Sentence from a retrieved-context response.",
                results=[item],
                selected=chosen,
                label=label,
                explanation=explanation,
                needs_refinement=refine,
                refinement_reason="The retrieved source does not settle this sentence." if refine else None,
                tags=["ragtruth", label.lower().replace(" ", "_")],
            )


def convert_felm(store: Store, held_out_prompts: set[str]) -> None:
    license_name = "CC BY-NC-SA 4.0"
    keep_extraction = {"wk", "science", "writing_rec"}
    for row in iter_jsonl(RAW / "felm" / "all.jsonl"):
        domain = row.get("domain") or ""
        if domain not in keep_extraction:
            continue
        prompt = (row.get("prompt") or "").strip()
        if prompt.casefold() in held_out_prompts:
            continue
        response = (row.get("response") or "").strip()
        segments = [segment.strip() for segment in (row.get("segmented_response") or []) if segment and segment.strip()]
        if not response or not segments:
            continue
        claims = [
            {"claim_id": f"c{index + 1}", "claim_text": clip(segment, 500), "context": clip(prompt, 240)}
            for index, segment in enumerate(segments)
        ]
        record(
            store,
            bucket="train",
            task="claim_extraction",
            dataset="felm",
            split="all",
            source_id=str(row.get("index")),
            license_name=license_name,
            user=response,
            assistant={"claims": claims},
            tags=["felm", domain],
        )
        if domain not in ("wk", "science"):
            continue
        comments = row.get("comment") or []
        labels = row.get("labels") or []
        refs = row.get("ref") or []
        for index, segment in enumerate(segments):
            factual = labels[index] if index < len(labels) else True
            if factual is True or str(factual).lower() == "true":
                continue
            comment = comments[index].strip() if index < len(comments) and isinstance(comments[index], str) else ""
            if len(comment) < 20 or not re.search(r"[A-Za-z]", comment):
                continue
            url = refs[index].strip() if index < len(refs) and isinstance(refs[index], str) else ""
            if url and not url.startswith("http"):
                url = ""
            item = result("s1", "Annotator correction", url, clip(comment, 700), "annotator")
            chosen = selected_from(item, "contradicts", "medium")
            emit_verification(
                store,
                bucket="train",
                dataset="felm",
                split="all",
                source_id=f"{row.get('index')}-seg{index}",
                license_name=license_name,
                claim_text=clip(segment, 500),
                context=clip(prompt, 240),
                results=[item],
                selected=[chosen],
                label="Contradicted",
                explanation=quote_explanation("Contradicted", [chosen]),
                needs_refinement=False,
                refinement_reason=None,
                tags=["felm", domain, "annotator_comment"],
            )


def convert_factscore(store: Store) -> None:
    license_name = "MIT"
    with zipfile.ZipFile(RAW / "factscore" / "data.zip") as archive:
        for name in ("data/labeled/InstructGPT.jsonl", "data/labeled/ChatGPT.jsonl", "data/labeled/PerplexityAI.jsonl"):
            model = Path(name).stem
            with archive.open(name) as handle:
                for line_index, raw_line in enumerate(handle):
                    row = json.loads(raw_line)
                    biography = (row.get("output") or "").strip()
                    topic = (row.get("topic") or "").strip()
                    if not biography:
                        continue
                    claims = []
                    for annotation in row.get("annotations") or []:
                        if annotation.get("is-relevant") is False:
                            continue
                        for fact in annotation.get("human-atomic-facts") or []:
                            text = (fact.get("text") or "").strip()
                            label = (fact.get("label") or "").upper()
                            if not text or label.startswith("I"):
                                continue
                            claims.append(
                                {
                                    "claim_id": f"c{len(claims) + 1}",
                                    "claim_text": text,
                                    "context": f"Biography of {topic}." if topic else "Generated biography.",
                                }
                            )
                    if not claims:
                        continue
                    record(
                        store,
                        bucket="train",
                        task="claim_extraction",
                        dataset="factscore",
                        split="labeled",
                        source_id=f"{model}-{line_index}",
                        license_name=license_name,
                        user=biography,
                        assistant={"claims": claims},
                        tags=["factscore", model],
                    )


def convert_factcheck_bench(store: Store) -> set[str]:
    prompts = set()
    for index, row in enumerate(iter_jsonl(RAW / "factcheck_bench" / "Factbench.jsonl")):
        prompt = (row.get("prompt") or "").strip()
        response = (row.get("response") or "").strip()
        if prompt:
            prompts.add(prompt.casefold())
        claims = []
        for claim_text in row.get("claims") or []:
            claim_text = (claim_text or "").strip()
            if not claim_text:
                continue
            claims.append(
                {
                    "claim_id": f"c{len(claims) + 1}",
                    "claim_text": claim_text,
                    "context": clip(prompt, 240),
                }
            )
        if not response:
            continue
        user = f"Question: {prompt}\n\nAnswer: {response}" if prompt else response
        record(
            store,
            bucket="eval",
            task="claim_extraction",
            dataset="factcheck_bench",
            split="eval",
            source_id=str(index),
            license_name="see Factcheck-GPT repository",
            user=user,
            assistant={"claims": claims},
            tags=["factcheck_bench", row.get("source") or "response"],
        )
    return prompts


def convert_freshqa(store: Store) -> None:
    with (RAW / "freshqa" / "freshqa.csv").open(encoding="utf-8", newline="") as handle:
        lines = handle.read().splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("id,split,question"))
    reader = csv.DictReader(lines[header_index:])
    for row in reader:
        question = (row.get("question") or "").strip()
        if not question:
            continue
        answers = [row.get(f"answer_{i}") for i in range(10)]
        answer = next((item.strip() for item in answers if item and item.strip()), "")
        year = (row.get("effective_year") or "").strip()
        false_premise = (row.get("false_premise") or "").strip().lower() in {"1", "true", "yes"}
        if false_premise:
            claim_text = question
            context = f"The question may contain a false premise. Annotated note as of {year or 'the review date'}: {answer}".strip()
        elif answer:
            claim_text = f"As of {year or 'the review date'}, {answer}"
            context = f"Fast-changing or dated fact. Question: {question}"
        else:
            continue
        record(
            store,
            bucket="eval",
            task="search_planning",
            dataset="freshqa",
            split=row.get("split") or "eval",
            source_id=str(row.get("id") or question[:40]),
            license_name="see FreshQA repository",
            user={"claim_id": "c1", "claim_text": claim_text, "context": context},
            assistant={"claim_id": "c1", "tool": "web_search", "arguments": {"query": question}},
            tags=["freshqa", "false_premise" if false_premise else "dated_fact", "freshqa_2026-04-21"],
        )


def convert_claimdecomp(store: Store) -> None:
    with (RAW / "claimdecomp" / "claimdecomp.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows):
        claim = (row.get("claim") or "").strip().strip('"')
        if not claim:
            continue
        literal = row.get("questions-literal") or ""
        questions = []
        for question in re.split(r"[\r\n]+", literal):
            question = question.strip()
            if question and question.lower() != "n/a":
                questions.append(question)
        context = " ".join(
            bit for bit in ((row.get("who") or "").strip(), (row.get("when_where") or "").strip()) if bit
        )
        for q_index, question in enumerate(dict.fromkeys(questions)):
            record(
                store,
                bucket="train",
                task="search_planning",
                dataset="claimdecomp",
                split="public_sheet",
                source_id=f"{index}-q{q_index}",
                license_name="see ClaimDecomp spreadsheet",
                user={"claim_id": "c1", "claim_text": claim, "context": context or "Complex public claim."},
                assistant={"claim_id": "c1", "tool": "web_search", "arguments": {"query": question}},
                tags=["claimdecomp", "literal_question"],
            )
    NOTES.append(
        "ClaimDecomp's public spreadsheet exposed 100 claims. The authors ask to be emailed for the rest, so the remaining annotated claims were not included."
    )


def convert_feverous(store: Store) -> None:
    license_name = "CC BY-SA 3.0"
    files = {
        "train": (RAW / "feverous" / "feverous_train_challenges.jsonl", "train"),
        "dev": (RAW / "feverous" / "feverous_dev_challenges.jsonl", "eval"),
    }
    for split, (path, bucket) in files.items():
        for row in iter_jsonl(path):
            claim = (row.get("claim") or "").strip()
            if not claim:
                continue
            queries = []
            for operation in row.get("annotator_operations") or []:
                if (operation.get("operation") or "").lower() != "search":
                    continue
                query = (operation.get("value") or "").strip()
                if len(query) < 3 or query.startswith("?"):
                    continue
                queries.append(query)
            for q_index, query in enumerate(dict.fromkeys(queries)):
                record(
                    store,
                    bucket=bucket,
                    task="search_planning",
                    dataset="feverous",
                    split=split,
                    source_id=f"{row.get('id')}-q{q_index}",
                    license_name=license_name,
                    user={
                        "claim_id": "c1",
                        "claim_text": claim,
                        "context": "Search query typed by a FEVEROUS annotator while checking the claim against Wikipedia.",
                    },
                    assistant={"claim_id": "c1", "tool": "web_search", "arguments": {"query": query}},
                    tags=["feverous", "annotator_query"],
                )


def write_report(store: Store) -> None:
    stats = {
        "counts": {f"{bucket}/{task}": count for (bucket, task), count in sorted(store.counts.items())},
        "verdicts": {
            f"{bucket}/{dataset}/{label}": count
            for (bucket, dataset, label), count in sorted(store.verdicts.items())
        },
        "notes": NOTES,
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    lines = [
        "BACKGROUND CHECKERS - PREPARED DATA",
        "====================================",
        "",
        "This file is generated by scripts/build_factcheck_data.py. It describes",
        "the JSONL files in data/processed and the places where the gold schema",
        "was adapted to the public data.",
        "",
        "Files",
        "-----",
        "data/processed/train/{claim_extraction,search_planning,evidence_selection,claim_evaluation}.jsonl",
        "data/processed/eval/{claim_extraction,search_planning,evidence_selection,claim_evaluation}.jsonl",
        "",
        "Each line is one chat-style example. Fields outside messages record the",
        "source dataset, the original split, the original id, and the license.",
        "Training files use source training splits. Eval files use development",
        "and test splits, plus FEVER 2.0, Factcheck-Bench, and FreshQA.",
        "",
        "Counts",
        "------",
    ]
    for key, count in stats["counts"].items():
        lines.append(f"{key:<40} {count}")
    lines.extend(["", "Verdicts in claim_evaluation", "---------------------------"])
    for key, count in stats["verdicts"].items():
        lines.append(f"{key:<48} {count}")
    lines.extend(
        [
            "",
            "Schema adaptations",
            "-------------------",
            "The four tasks and the three verdicts match GOLD_SET_SCHEMA.txt.",
            "Search planning stays one web_search tool call. When a source has",
            "several real questions, each question becomes its own example.",
            "Evidence excerpts are copied from the source. They are shortened",
            "only when the source text is longer than the stored excerpt, and the",
            "shortened text is what both the input and the output contain.",
            "Stance is supports, contradicts, or context. Context is used when",
            "the source did not label that individual excerpt. AVeriTeC answers",
            "are context because only the whole claim has a verdict.",
            "Quality is high for Wikipedia, scientific abstracts, government and",
            "education domains, and RAGTruth news summaries. Other sources are",
            "medium. The source datasets do not label quality, so this is a",
            "domain rule rather than an annotator judgment.",
            "A URL is left blank when the dataset has no URL. No URL was invented.",
            "needs_refinement is true when the evidence is empty, partial,",
            "conflicting, or otherwise insufficient. A supported claim with fewer",
            "than three sources is not marked for refinement: the project asks",
            "for three sources when the results contain them, and these datasets",
            "often do not.",
            "VitaminC pairs that give both supporting and refuting text for the",
            "same claim produce an extra Insufficient Evidence example. The",
            "individual pairs are kept as well.",
            "WiCE not_supported and partially_supported map to Insufficient",
            "Evidence. Missing support is not labeled Contradicted.",
            "RAGTruth Evident Conflict maps to Contradicted. Baseless info and",
            "subtle conflicts map to Insufficient Evidence. Data-to-text",
            "responses are left out.",
            "FELM math and reasoning are left out. Writing segments are used only",
            "for claim extraction. False world-knowledge and science segments",
            "are verdict examples only when the annotator wrote a correction.",
            "FActScore human atomic facts are claim-extraction examples. Their",
            "Supported and Not-supported labels are not turned into verdicts,",
            "because the Wikipedia sentences are not in this download.",
            "FEVEROUS is used for the annotators' search queries. Its evidence",
            "fields are Wikipedia element ids without the sentence or cell text,",
            "so they are not used as verdicts.",
            "Factcheck-Bench prompts are removed from the FELM training rows so",
            "the same response is not in both splits.",
            "",
            "Not included",
            "------------",
            "LLM-AggreFact is gated on Hugging Face and was not downloaded.",
            "ClaimBuster's Zenodo files returned HTTP 403, and no public mirror",
            "was available from this machine.",
            "ClaimDecomp's full release is not public. The public spreadsheet",
            "was converted in full.",
            "WildHallucinations was not converted. Its verdicts are automatic,",
            "and the release does not include gold claim splits.",
            "LIAR, MultiFC, SelfCheckGPT, TruthfulQA, and HaluEval were not",
            "converted. Their labels do not match this project's evidence rules.",
            "",
            "Notes",
            "-----",
        ]
    )
    lines.extend(NOTES or ["None."])
    lines.extend(
        [
            "",
            "Rebuild",
            "-------",
            "python scripts/download_raw.py",
            "python scripts/build_factcheck_data.py",
            "",
        ]
    )
    text = "\n".join(lines)
    (ROOT / "Documentation" / "DATA_BUILD.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    NOTES.append(
        "LLM-AggreFact was not downloaded because the Hugging Face dataset is gated."
    )
    NOTES.append(
        "ClaimBuster was not downloaded because Zenodo returned HTTP 403 and the checked GitHub paths do not host the CSV."
    )
    store = Store()
    print("Factcheck-Bench", flush=True)
    held_out = convert_factcheck_bench(store)
    print("FELM", flush=True)
    convert_felm(store, held_out)
    print("FActScore", flush=True)
    convert_factscore(store)
    print("AVeriTeC", flush=True)
    convert_averitec(store)
    print("SciFact", flush=True)
    convert_scifact(store)
    print("WiCE", flush=True)
    convert_wice(store)
    print("RAGTruth", flush=True)
    convert_ragtruth(store)
    print("VitaminC", flush=True)
    convert_vitaminc(store)
    print("FEVEROUS queries", flush=True)
    convert_feverous(store)
    print("FreshQA", flush=True)
    convert_freshqa(store)
    print("ClaimDecomp", flush=True)
    convert_claimdecomp(store)
    print("FEVER", flush=True)
    convert_fever(store)
    store.close()
    write_report(store)


if __name__ == "__main__":
    main()
