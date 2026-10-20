"""
evaluate.py  -  RETRIEVAL metrics (Recall@k, nDCG@k) and RAG QA metrics.

Binary relevance: a retrieved passage is relevant iff its id is in the gold set.
If a question has NO gold ids, metrics return None (= "not evaluable"), never 0.

Recall@k = |relevant in top-k| / |gold|            (handles several gold passages)
nDCG@k   = DCG@k / IDCG@k,  DCG = sum rel_i / log2(i+1),  i = 1-based rank,
           IDCG = best possible DCG with min(|gold|, k) relevant at the top.

TWO VIEWS (needed because TAT-QA has no row-level table labels):
  strict    : the ranked list is scored as-is. Table-row passages count as
              NOT relevant. Valid as-is only when the gold set is complete
              (answer_from == "text"); for "table-text" it is a LOWER BOUND,
              because a retrieved table row may be genuinely useful.
  condensed : "unjudged" passages (table rows) are removed from the ranked list
              before scoring, i.e. ranks are computed among paragraphs only.
              Unlabelled rows are neither rewarded nor punished. This is the
              honest primary view for text / table-text questions.
"""

from __future__ import annotations

import math
from typing import Iterable


def _dedupe(ids: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def recall_at_k(ranked_ids: list[str], gold_ids: Iterable[str], k: int = 5):
    gold = set(gold_ids)
    if not gold:
        return None
    top = _dedupe(ranked_ids)[:k]
    return len(gold.intersection(top)) / len(gold)


def ndcg_at_k(ranked_ids: list[str], gold_ids: Iterable[str], k: int = 10):
    gold = set(gold_ids)
    if not gold:
        return None
    top = _dedupe(ranked_ids)[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, pid in enumerate(top) if pid in gold)  # i is 0-based
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg


def condense(ranked_ids: list[str], unjudged_ids: Iterable[str]) -> list[str]:
    drop = set(unjudged_ids)
    return [i for i in ranked_ids if i not in drop]


def retrieval_metrics(ranked_ids: list[str], gold_ids: list[str],
                      unjudged_ids: Iterable[str] = (), k_recall: int = 5, k_ndcg: int = 10):
    """
    Returns None if no gold; else a dict with strict and condensed metrics. Key names follow the k values
    ("recall@5_strict", "ndcg@10_condensed", ...). A gold passage is never treated as unjudged.
    """
    if not gold_ids:
        return None
    unjudged = set(unjudged_ids) - set(gold_ids)
    cond = condense(ranked_ids, unjudged)
    return {
        f"recall@{k_recall}_strict": recall_at_k(ranked_ids, gold_ids, k_recall),
        f"ndcg@{k_ndcg}_strict": ndcg_at_k(ranked_ids, gold_ids, k_ndcg),
        f"recall@{k_recall}_condensed": recall_at_k(cond, gold_ids, k_recall),
        f"ndcg@{k_ndcg}_condensed": ndcg_at_k(cond, gold_ids, k_ndcg),
    }


def mean_metrics(per_question: list[dict]) -> dict:
    """Average each metric over questions whose metrics are not None."""
    scored = [m for m in per_question if m]
    out = {"n_evaluated": len(scored)}
    if scored:
        for key in scored[0]:
            out[key] = sum(m[key] for m in scored) / len(scored)
    return out


def citation_accuracy(cited_passages: list[str], gold_evidence_ids: list[str]) -> float:
    """Check if the cited passage(s) overlap with gold evidence passages."""
    if not gold_evidence_ids:
        return 1.0  # no gold evidence specified
    if not cited_passages:
        return 0.0
    gold_set = set(gold_evidence_ids)
    cited_set = set(cited_passages)
    return float(len(gold_set & cited_set) > 0)

