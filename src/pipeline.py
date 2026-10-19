"""
pipeline.py - the interface Track B calls, plus the batch runner.

(1) SINGLE QUESTION
    from src.pipeline import retrieve
    out = retrieve("What was net sales in 2019?", top_k=5, method="bm25")
    out == {"question": str, "method": "bm25" | "bm25+contriever", "split": "dev",
            "doc_id": None | str, "top_k": 5,
            "results": [ {"passage_id","text","rank","source_type","doc_id",
                          "bm25_score","bm25_rank","contriever_score"?}, ... ]}
    Searches the corpus of `split` (default: dev). Pass doc_id= to restrict to
    one document's passages (oracle-document setting).

(2) BATCH  (writes JSONL, one line per question x retriever)
    python -m src.pipeline --split dev --n 200 --seed 42 --methods bm25 --out outputs/run_bm25

This module knows nothing about noise injection or reasoning (Track B).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from functools import lru_cache

from .data_loader import COMPLETE, NO_GOLD, PARTIAL, load_split
from .evaluate import mean_metrics, retrieval_metrics
from .passage_builder import TABLE_ROW, build_corpus
from .retriever import BM25ContrieverRetriever, BM25Retriever, ContrieverEncoder, reserve_row_slots

METHODS = ("bm25", "bm25+contriever")


# ------------------------------------------------------------ jsonl helpers
def save_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------ retriever setup
def build_retriever(corpus: list[dict], method: str = "bm25", candidate_n: int = 100,
                    similarity: str = "dot", cache_dir: str = "data/cache", **encoder_kw):
    bm25 = BM25Retriever(corpus)
    if method == "bm25":
        return bm25
    if method == "bm25+contriever":
        enc = ContrieverEncoder(cache_dir=cache_dir, **encoder_kw)
        return BM25ContrieverRetriever(bm25, enc, candidate_n=candidate_n, similarity=similarity)
    raise ValueError(f"method must be one of {METHODS}")


SLOT_SEARCH_DEPTH = 100   # candidates examined when reserving slots (same depth as the experiments)


@lru_cache(maxsize=4)
def _cached_retriever(split: str, method: str, data_dir: str, paragraph_context: str = "none"):
    docs, _ = load_split(split, data_dir)
    return build_retriever(build_corpus(docs, split, paragraph_context), method)


def retrieve(question: str, top_k: int = 5, method: str = "bm25", split: str = "dev",
             doc_id: str | None = None, data_dir: str = "data",
             paragraph_context: str = "none", reserve_paragraph_slots: int = 0) -> dict:
    """
    Public single-question interface (see module docstring for the format).
    Defaults reproduce the ORIGINAL baseline. The configuration recommended by the experiments is
        retrieve(q, top_k=5, method="bm25", paragraph_context="table_labels", reserve_paragraph_slots=4)
    (table-aware paragraph indexing; of the top 5, 4 paragraph slots + 1 table-row slot).
    Result passages always carry the ORIGINAL paragraph text, never the enriched search text.
    """
    r = _cached_retriever(split, method, data_dir, paragraph_context)
    if reserve_paragraph_slots:
        res = r.retrieve(question, top_k=max(top_k, SLOT_SEARCH_DEPTH), doc_id=doc_id)
        res = reserve_row_slots(res, reserve_paragraph_slots, k=5)[:top_k]
    else:
        res = r.retrieve(question, top_k=top_k, doc_id=doc_id)
    return {"question": question, "method": r.name, "split": split, "doc_id": doc_id, "top_k": top_k,
            "paragraph_context": paragraph_context, "reserve_paragraph_slots": reserve_paragraph_slots,
            "results": res}


# ------------------------------------------------------------ sampling
def sample_questions(records: list[dict], n: int | None, seed: int = 42,
                     only_evaluable: bool = False) -> list[dict]:
    """Deterministic: sort by question_id first (file order can't matter), then seeded sample."""
    pool = sorted(records, key=lambda r: r["question_id"])
    if only_evaluable:
        pool = [r for r in pool if r["evidence_status"] != NO_GOLD]
    if n is None or n >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, n), key=lambda r: r["question_id"])


# ------------------------------------------------------------ batch run
def run_batch(retriever, questions: list[dict], unjudged_ids: set[str], store_k: int = 30,
              doc_scope: bool = False, reserve_paragraph_slots: int = 0) -> list[dict]:
    """
    Retrieve for each question. Only q["question"] (and optionally q["doc_id"])
    is passed to the retriever - never the answer or gold evidence.
    Gold is attached to the output row AFTER retrieval, only for scoring.
    """
    rows = []
    for q in questions:
        res = retriever.retrieve(q["question"], top_k=store_k,
                                 doc_id=q["doc_id"] if doc_scope else None)
        if reserve_paragraph_slots:
            res = reserve_row_slots(res, reserve_paragraph_slots, k=5)
        ranked_ids = [x["passage_id"] for x in res]
        metrics = retrieval_metrics(ranked_ids, q["gold_evidence_ids"], unjudged_ids)
        rows.append({
            "question_id": q["question_id"], "split": q["split"], "doc_id": q["doc_id"],
            "question": q["question"],
            "gold_answer": q["answer"], "gold_answer_text": q["answer_text"],
            "answer_type": q["answer_type"], "answer_from": q["answer_from"],
            "gold_evidence_ids": q["gold_evidence_ids"],       # paragraph ids only (see README)
            "evidence_status": q["evidence_status"],
            "retriever": retriever.name, "doc_scope": doc_scope,
            "reserve_paragraph_slots": reserve_paragraph_slots,
            "retrieved": [{k: x[k] for k in x if k != "doc_id"} for x in res],
            "retrieval_metrics": metrics,                      # None when no gold labels
        })
    return rows


# ------------------------------------------------------------ summaries
def _group(rows, key_fn):
    groups = {}
    for r in rows:
        groups.setdefault(key_fn(r), []).append(r["retrieval_metrics"])
    return {str(k): mean_metrics(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def summarize(rows: list[dict]) -> dict:
    """Metrics for ONE retriever's rows."""
    evaluable = [r for r in rows if r["retrieval_metrics"]]
    complete = [r for r in rows if r["evidence_status"] == COMPLETE]
    return {
        "n_questions": len(rows),
        "n_evaluated": len(evaluable),
        "n_excluded": len(rows) - len(evaluable),
        "excluded_reason_counts": {
            "no paragraph-level gold evidence (table-only question / unlabelled)":
                sum(1 for r in rows if r["evidence_status"] == NO_GOLD)},
        "overall_evaluable": mean_metrics([r["retrieval_metrics"] for r in evaluable]),
        "text_only_complete_gold": mean_metrics([r["retrieval_metrics"] for r in complete]),
        "by_answer_from": _group(evaluable, lambda r: r["answer_from"]),
        "by_answer_type": _group(evaluable, lambda r: r["answer_type"]),
        "by_evidence_status": _group(evaluable, lambda r: r["evidence_status"]),
    }


# ------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Run Track A retrieval on a TAT-QA split")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=200, help="questions to sample (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-evaluable", action="store_true",
                    help="sample only from questions that have paragraph gold labels")
    ap.add_argument("--methods", nargs="+", default=["bm25"], choices=METHODS)
    ap.add_argument("--store-k", type=int, default=30, help="passages saved per question")
    ap.add_argument("--candidate-n", type=int, default=100, help="BM25 candidates to rerank")
    ap.add_argument("--similarity", default="dot", choices=["dot", "cosine"])
    ap.add_argument("--doc-scope", action="store_true",
                    help="search only inside the question's own document (oracle-document)")
    ap.add_argument("--cache-dir", default="data/cache", help="Contriever embedding cache")
    ap.add_argument("--paragraph-context", default="none", choices=["none", "table_labels"],
                    help="experimental: index paragraphs together with their document's table labels (BM25 and Contriever)")
    ap.add_argument("--reserve-paragraph-slots", type=int, default=0, metavar="N",
                    help="of the top 5, give N slots to paragraphs and 5-N to table rows (0 = off; experiments used 4)")
    ap.add_argument("--exclude-docs-from", nargs="+", default=[], metavar="JSONL",
                    help="held-out sample: drop every question whose document appears in these run files")
    ap.add_argument("--out", default="outputs/run")
    a = ap.parse_args(argv)

    docs, records = load_split(a.split, a.data_dir)
    corpus = build_corpus(docs, a.split, a.paragraph_context)
    unjudged = {p["passage_id"] for p in corpus if p["source_type"] == TABLE_ROW}
    if a.exclude_docs_from:
        excluded = {r["doc_id"] for p in a.exclude_docs_from for r in load_jsonl(p)}
        records = [r for r in records if r["doc_id"] not in excluded]
        print(f"held-out mode: excluded {len(excluded)} documents; {len(records)} questions remain")
    questions = sample_questions(records, a.n or None, a.seed, a.only_evaluable)
    status_counts = {s: sum(1 for q in questions if q["evidence_status"] == s)
                     for s in (COMPLETE, PARTIAL, NO_GOLD)}
    print(f"split={a.split} corpus={len(corpus)} passages | sampled {len(questions)} questions {status_counts}")

    summary = {"config": {**vars(a), "corpus_size": len(corpus),
                          "n_paragraph_passages": len(corpus) - len(unjudged),
                          "n_table_row_passages": len(unjudged),
                          "sample_evidence_status_counts": status_counts,
                          "question_ids_sha_note": "sample is deterministic: sorted by question_id, random.Random(seed)"},
               "results": {}}
    for method in a.methods:
        retr = build_retriever(corpus, method, a.candidate_n, a.similarity, a.cache_dir)
        rows = run_batch(retr, questions, unjudged, a.store_k, a.doc_scope, a.reserve_paragraph_slots)
        if hasattr(getattr(retr, "encoder", None), "save_cache"):
            retr.encoder.save_cache()
        path = os.path.join(a.out, f"retrieval_{retr.name.replace('+', '_')}.jsonl")
        save_jsonl(rows, path)
        summary["results"][retr.name] = summarize(rows)
        print(f"[{retr.name}] wrote {path}")
    spath = os.path.join(a.out, "metrics_summary.json")
    os.makedirs(a.out, exist_ok=True)
    with open(spath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {spath}")
    return summary


if __name__ == "__main__":
    main()