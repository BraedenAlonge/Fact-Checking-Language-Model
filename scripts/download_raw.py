"""Download the public datasets used to build the fact-checking training data.

Raw archives stay in data/raw and are not the training files. Re-running this
script skips a file when it is already present and non-trivial in size.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

URLS = {
    "fever/train.jsonl": "https://fever.ai/download/fever/train.jsonl",
    "fever/shared_task_dev.jsonl": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "fever/wiki-pages.zip": "https://fever.ai/download/fever/wiki-pages.zip",
    "fever/fever2-fixers-dev.jsonl": "https://fever.ai/download/fever2.0/fever2-fixers-dev.jsonl",
    "feverous/feverous_train_challenges.jsonl": "https://fever.ai/download/feverous/feverous_train_challenges.jsonl",
    "feverous/feverous_dev_challenges.jsonl": "https://fever.ai/download/feverous/feverous_dev_challenges.jsonl",
    "vitaminc/vitaminc.zip": "https://github.com/TalSchuster/talschuster.github.io/raw/master/static/vitaminc.zip",
    "scifact/data.tar.gz": "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
    "felm/all.jsonl": "https://huggingface.co/datasets/hkust-nlp/felm/resolve/main/all.jsonl",
    "averitec/train.json": "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/train.json",
    "averitec/dev.json": "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/dev.json",
    "averitec/test.json": "https://huggingface.co/chenxwh/AVeriTeC/resolve/main/data/test.json",
    "factscore/data.zip": "https://drive.google.com/uc?export=download&id=155exEdKs7R21gZF4G-x54-XN3qswBcPo",
    # Zenodo's public file host returns HTTP 403 or times out from some clients.
    # This is the 23,533-row ClaimBuster conversion published on Hugging Face.
    "claimbuster/train.parquet": "https://huggingface.co/datasets/Nithiwat/claimbuster/resolve/main/data/train-00000-of-00001-5f01fa457a7cb014.parquet",
    "ragtruth/response.jsonl": "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/response.jsonl",
    "ragtruth/source_info.jsonl": "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/source_info.jsonl",
    "wice/claim/train.jsonl": "https://raw.githubusercontent.com/ryokamoi/wice/main/data/entailment_retrieval/claim/train.jsonl",
    "wice/claim/dev.jsonl": "https://raw.githubusercontent.com/ryokamoi/wice/main/data/entailment_retrieval/claim/dev.jsonl",
    "wice/claim/test.jsonl": "https://raw.githubusercontent.com/ryokamoi/wice/main/data/entailment_retrieval/claim/test.jsonl",
    "wice/subclaim/train.jsonl": "https://raw.githubusercontent.com/ryokamoi/wice/main/data/entailment_retrieval/subclaim/train.jsonl",
    "wice/subclaim/dev.jsonl": "https://raw.githubusercontent.com/ryokamoi/wice/main/data/entailment_retrieval/subclaim/dev.jsonl",
    "wice/subclaim/test.jsonl": "https://raw.githubusercontent.com/ryokamoi/wice/main/data/entailment_retrieval/subclaim/test.jsonl",
    "factcheck_bench/Factbench.jsonl": "https://raw.githubusercontent.com/yuxiaw/Factcheck-GPT/main/Factbench.jsonl",
    "freshqa/freshqa.csv": "https://docs.google.com/spreadsheets/d/1_8mi-yuK30mvoDJu1KQXD6ODem7MKMcIgVAwDSzJkjM/export?format=csv",
    "claimdecomp/claimdecomp.csv": "https://docs.google.com/spreadsheets/d/1g6bmuc1D5jbLE0U9Y68MZWH_VCr6ZXfeKA2muA1uJD0/export?format=csv",
    "llm_aggre_fact/dev.parquet": "https://huggingface.co/datasets/lytang/LLM-AggreFact/resolve/main/data/dev-00000-of-00001.parquet",
    "llm_aggre_fact/test.parquet": "https://huggingface.co/datasets/lytang/LLM-AggreFact/resolve/main/data/test-00000-of-00001.parquet",
}


def download(rel: str, url: str) -> None:
    dest = RAW / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"skip {rel} ({dest.stat().st_size} bytes)", flush=True)
        return
    print(f"get  {rel}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)
    print(f"done {rel} ({dest.stat().st_size} bytes)", flush=True)


def main() -> None:
    only = set(sys.argv[1:])
    for rel, url in URLS.items():
        if only and rel not in only and not any(rel.startswith(item) for item in only):
            continue
        try:
            download(rel, url)
        except Exception as exc:
            print(f"FAIL {rel}: {exc}", flush=True)


if __name__ == "__main__":
    main()
