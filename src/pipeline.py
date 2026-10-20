"""
pipeline.py - Fully integrated RAG pipeline combining:
  (1) Person A: Lexical & dense retrieval over TAT-QA financial corpus
  (2) Person B: Evidence tagging, program synthesis, verification, execution, and noise injection

Usage:
  # Retrieval only (Person A):
  from src.pipeline import retrieve
  out = retrieve("What was net sales in 2019?", top_k=5, method="bm25")

  # Single passage answer (Person B standalone):
  from src.pipeline import answer_question
  res = answer_question("What was the change in revenue?", passage_text)

  # Full End-to-End RAG with Noise Injection (Person A + Person B):
  from src.pipeline import answer_with_retrieval
  res = answer_with_retrieval("What was net sales in 2019?", noise_type="conflict", severity=2)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

# Person A components
from .data_loader import COMPLETE, NO_GOLD, PARTIAL, load_split
from .evaluate import mean_metrics, retrieval_metrics
from .passage_builder import TABLE_ROW, build_corpus
from .retriever import BM25ContrieverRetriever, BM25Retriever, ContrieverEncoder, reserve_row_slots

# Person B components
from .evidence_tagger import tag_operands
from .inject_noise import inject_noise
from .program_synthesizer import get_program_citations, synthesize_program
from .verifier import _to_float, verify_program

METHODS = ("bm25", "bm25+contriever")
PROGRAM_RE = re.compile(r"^(\w+)\((.*)\)$")


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


SLOT_SEARCH_DEPTH = 100


@lru_cache(maxsize=4)
def _cached_retriever(split: str, method: str, data_dir: str, paragraph_context: str = "none"):
    docs, _ = load_split(split, data_dir)
    return build_retriever(build_corpus(docs, split, paragraph_context), method)


def retrieve(question: str, top_k: int = 5, method: str = "bm25", split: str = "dev",
             doc_id: str | None = None, data_dir: str = "data",
             paragraph_context: str = "table_labels", reserve_paragraph_slots: int = 4) -> dict:
    """Public single-question retrieval interface (Person A)."""
    r = _cached_retriever(split, method, data_dir, paragraph_context)
    if reserve_paragraph_slots:
        res = r.retrieve(question, top_k=max(top_k, SLOT_SEARCH_DEPTH), doc_id=doc_id)
        res = reserve_row_slots(res, reserve_paragraph_slots, k=5)[:top_k]
    else:
        res = r.retrieve(question, top_k=top_k, doc_id=doc_id)
    return {"question": question, "method": r.name, "split": split, "doc_id": doc_id, "top_k": top_k,
            "paragraph_context": paragraph_context, "reserve_paragraph_slots": reserve_paragraph_slots,
            "results": res}


def sample_questions(records: list[dict], n: int | None, seed: int = 42,
                     only_evaluable: bool = False) -> list[dict]:
    """Deterministic question sampling."""
    pool = sorted(records, key=lambda r: r["question_id"])
    if only_evaluable:
        pool = [r for r in pool if r["evidence_status"] != NO_GOLD]
    if n is None or n >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, n), key=lambda r: r["question_id"])


def run_batch(retriever, questions: list[dict], unjudged_ids: set[str], store_k: int = 30,
              doc_scope: bool = False, reserve_paragraph_slots: int = 0) -> list[dict]:
    """Batch retrieval runner across questions."""
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
            "gold_evidence_ids": q["gold_evidence_ids"],
            "evidence_status": q["evidence_status"],
            "retriever": retriever.name, "doc_scope": doc_scope,
            "reserve_paragraph_slots": reserve_paragraph_slots,
            "retrieved": [{k: x[k] for k in x if k != "doc_id"} for x in res],
            "retrieval_metrics": metrics,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    """Metrics summary for retrieval rows."""
    evaluable = [r for r in rows if r["retrieval_metrics"]]
    complete = [r for r in rows if r["evidence_status"] == COMPLETE]
    return {
        "n_questions": len(rows),
        "n_evaluated": len(evaluable),
        "n_excluded": len(rows) - len(evaluable),
        "overall_evaluable": mean_metrics([r["retrieval_metrics"] for r in evaluable]),
        "text_only_complete_gold": mean_metrics([r["retrieval_metrics"] for r in complete]),
    }


# ------------------------------------------------------------ Person B: Execution Engine
def _compute(program: str) -> Optional[float]:
    """Evaluate a verified program string. Returns None if it can't be evaluated."""
    m = PROGRAM_RE.match(program.strip())
    if not m:
        return None
    op, arg_str = m.group(1), m.group(2)
    args = [a.strip() for a in arg_str.split(", ") if a.strip()]

    if op == "count":
        if args and args[0].isdigit():
            return float(args[0])
        return float(len(args))

    try:
        vals = [_to_float(a) for a in args]
    except (ValueError, TypeError):
        return None

    if not vals:
        return None

    if op == "lookup":
        return vals[0]
    if op == "add":
        return sum(vals)
    if op == "sub":
        return vals[0] - vals[1] if len(vals) >= 2 else vals[0]
    if op == "ratio":
        return vals[0] / vals[1] if len(vals) >= 2 and vals[1] != 0 else None
    if op == "yoy":
        return ((vals[0] - vals[1]) / vals[1] * 100) if len(vals) >= 2 and vals[1] != 0 else None
    if op == "average":
        return sum(vals) / len(vals) if vals else None
    return None


def _infer_scale(program: str, operands: List[Dict[str, Any]]) -> str:
    """Infer the answer scale (e.g. million, billion, percent) from the program and operands."""
    m = PROGRAM_RE.match(program.strip())
    if not m:
        return ""
    op = m.group(1)
    if op == "yoy" or op == "ratio":
        return "percent"

    # For lookup/add/sub, find scale of the participating operands
    for opnd in operands:
        if opnd.get("span") in program:
            if opnd.get("scale"):
                return opnd.get("scale")
            if opnd.get("type") == "percent":
                return "percent"
    return ""


def answer_question(question: str, passage: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]) -> Dict[str, Any]:
    """Run Person B downstream pipeline on one question and provided passage(s)."""
    operands = tag_operands(passage)
    program = synthesize_program(question, operands)
    is_valid, error = verify_program(program, operands, passage)

    result = {
        "program": program,
        "valid": is_valid,
        "error": error,
        "answer": None,
        "scale": "",
        "cited_passages": [],
    }

    if is_valid:
        value = _compute(program)
        if value is not None:
            result["answer"] = round(value, 4) if not math.isclose(value, round(value)) else int(round(value))
            result["scale"] = _infer_scale(program, operands)
            result["cited_passages"] = get_program_citations(program, operands)
    return result


def answer_with_retrieval(
    question: str,
    top_k: int = 5,
    method: str = "bm25",
    split: str = "dev",
    doc_id: Optional[str] = None,
    paragraph_context: str = "table_labels",
    reserve_paragraph_slots: int = 4,
    noise_type: str = "clean",
    severity: int = 0,
    seed: int = 42,
    corpus: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """End-to-End RAG: Retrieve evidence -> Inject Noise -> Tag Operands -> Synthesize -> Verify -> Execute."""
    ret_out = retrieve(
        question,
        top_k=top_k,
        method=method,
        split=split,
        doc_id=doc_id,
        paragraph_context=paragraph_context,
        reserve_paragraph_slots=reserve_paragraph_slots,
    )
    clean_passages = ret_out.get("results", [])

    # Step: Noise injection
    passages_with_noise = inject_noise(
        clean_passages,
        noise_type=noise_type,
        severity=severity,
        corpus=corpus,
        seed=seed,
    )

    # Step: Answer synthesis and verification
    qa_result = answer_question(question, passages_with_noise)

    return {
        "question": question,
        "retrieval": ret_out,
        "noise_type": noise_type,
        "severity": severity,
        "passages_used": passages_with_noise,
        **qa_result,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run Track A retrieval / Track B integrated pipeline")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only-evaluable", action="store_true")
    ap.add_argument("--methods", nargs="+", default=["bm25"], choices=METHODS)
    ap.add_argument("--store-k", type=int, default=30)
    ap.add_argument("--paragraph-context", default="table_labels")
    ap.add_argument("--reserve-paragraph-slots", type=int, default=4)
    ap.add_argument("--out", default="outputs/run")
    a = ap.parse_args(argv)

    docs, records = load_split(a.split, a.data_dir)
    corpus = build_corpus(docs, a.split, a.paragraph_context)
    unjudged = {p["passage_id"] for p in corpus if p["source_type"] == TABLE_ROW}
    questions = sample_questions(records, a.n or None, a.seed, a.only_evaluable)

    summary = {"config": vars(a), "results": {}}
    for method in a.methods:
        retr = build_retriever(corpus, method)
        rows = run_batch(retr, questions, unjudged, a.store_k, False, a.reserve_paragraph_slots)
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
