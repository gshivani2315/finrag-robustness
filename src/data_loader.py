"""
data_loader.py
--------------
Download / load the official TAT-QA JSON files and flatten them into one
record per QUESTION, keeping the link question -> gold answer -> gold evidence.

OBSERVED SCHEMA (inspected on the real files, not assumed)
----------------------------------------------------------
File = list of documents. Each document:
    {"table": {"uid": str, "table": list[list[str]]},
     "paragraphs": [{"uid": str, "order": int (1-based), "text": str}, ...],
     "questions": [{"uid", "order", "question", "answer", "derivation",
                    "answer_type", "answer_from", "rel_paragraphs",
                    "req_comparison", "scale"}, ...]}
 * `answer` can be a list, float, int or str.
 * `rel_paragraphs` = list of paragraph `order` values AS STRINGS (e.g. ["2"]).
 * `answer_from` in {"text", "table", "table-text"}.
 * The TEST file has NO answers and NO evidence (only uid/order/question).
   It can be retrieved against but not scored.

EVIDENCE STATUS (what we can legitimately evaluate retrieval on)
----------------------------------------------------------------
 paragraph_gold_complete : answer_from == "text" and rel_paragraphs non-empty.
                           Paragraph labels are the full gold set.
 paragraph_gold_partial  : rel_paragraphs non-empty but answer also uses the
                           table (answer_from "table-text"). Paragraph side is
                           labelled; table side is NOT.
 no_gold                 : no paragraph labels (table-only questions, and the
                           whole test split). Table rows are never labelled.
"""

from __future__ import annotations

import json
import os
import urllib.request

from .passage_builder import paragraph_passage_id

SPLITS = ("train", "dev", "test")
RAW_URL = ("https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/"
           "dataset_raw/tatqa_dataset_{split}.json")

COMPLETE = "paragraph_gold_complete"
PARTIAL = "paragraph_gold_partial"
NO_GOLD = "no_gold"


def split_path(split: str, data_dir: str = "data") -> str:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    return os.path.join(data_dir, f"tatqa_dataset_{split}.json")


def download_split(split: str, data_dir: str = "data") -> str:
    """Download one split from the official GitHub repo (skips if present)."""
    os.makedirs(data_dir, exist_ok=True)
    path = split_path(split, data_dir)
    if not os.path.exists(path):
        urllib.request.urlretrieve(RAW_URL.format(split=split), path)
    return path


def load_raw_split(split: str, data_dir: str = "data") -> list[dict]:
    """Load one raw split file and check its top-level structure."""
    path = split_path(split, data_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.data_loader --download")
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    if not isinstance(docs, list) or not all(
            isinstance(d, dict) and {"table", "paragraphs", "questions"} <= d.keys()
            for d in docs):
        raise ValueError(f"{path}: unexpected structure (expected list of "
                         "documents with table/paragraphs/questions)")
    return docs


def answer_to_text(answer) -> str:
    """Readable string for any answer type (list / number / str / missing)."""
    if answer is None:
        return ""
    if isinstance(answer, list):
        return " | ".join(str(a) for a in answer)
    return str(answer)


def build_qa_records(documents: list[dict], split: str) -> list[dict]:
    """One flat record per question; never invents gold evidence."""
    records = []
    for doc in documents:
        doc_id = doc["table"]["uid"]
        valid_orders = {str(p.get("order", i + 1))
                        for i, p in enumerate(doc.get("paragraphs") or [])}
        for q in doc.get("questions") or []:
            raw_rel = q.get("rel_paragraphs") or []
            # keep only references that point at a paragraph that really exists
            gold_orders = [int(r) for r in raw_rel if str(r) in valid_orders]
            invalid_refs = [r for r in raw_rel if str(r) not in valid_orders]
            gold_ids = [paragraph_passage_id(doc_id, o) for o in gold_orders]

            if not gold_ids:
                status = NO_GOLD
            elif q.get("answer_from") == "text":
                status = COMPLETE
            else:
                status = PARTIAL

            records.append({
                "question_id": q["uid"],
                "split": split,
                "doc_id": doc_id,
                "question_order": q.get("order"),
                "question": q.get("question", ""),
                "answer": q.get("answer"),                 # raw, as in dataset
                "answer_text": answer_to_text(q.get("answer")),
                "answer_type": q.get("answer_type"),       # span/multi-span/arithmetic/count
                "answer_from": q.get("answer_from"),       # text/table/table-text
                "scale": q.get("scale"),
                "derivation": q.get("derivation"),
                "req_comparison": q.get("req_comparison"),
                "gold_paragraph_orders": gold_orders,
                "gold_evidence_ids": gold_ids,
                "invalid_rel_paragraphs": invalid_refs,
                "evidence_status": status,
            })
    return records


def load_split(split: str, data_dir: str = "data"):
    """Convenience: returns (documents, qa_records)."""
    docs = load_raw_split(split, data_dir)
    return docs, build_qa_records(docs, split)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Download / summarise TAT-QA")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--download", action="store_true", help="download train/dev/test")
    args = ap.parse_args()
    for s in SPLITS:
        if args.download:
            download_split(s, args.data_dir)
        docs, recs = load_split(s, args.data_dir)
        by = {}
        for r in recs:
            by[r["evidence_status"]] = by.get(r["evidence_status"], 0) + 1
        print(f"{s}: {len(docs)} docs, {len(recs)} questions, evidence_status={by}")

