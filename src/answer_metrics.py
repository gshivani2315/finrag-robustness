"""
answer_metrics.py - SIMPLIFIED answer metrics (Exact Match, token-level F1).

These compare a GENERATED answer to the reference answer.
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal, InvalidOperation

_TOKEN_RE = re.compile(r"(?<!\w)-\d+(?:\.\d+)?|\d+(?:\.\d+)?|[a-z]+")
_ARTICLES = {"a", "an", "the"}


def _to_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        try:
            return format(Decimal(repr(x)), "f")
        except InvalidOperation:
            return str(x)
    return str(x)


def _canon_number(tok: str) -> str:
    try:
        return format(Decimal(tok).normalize(), "f")
    except InvalidOperation:
        return tok


def normalize_tokens(x) -> list[str]:
    s = _to_text(x).lower().replace("$", " ").replace("%", " ")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)          # 1,496.5 -> 1496.5
    toks = [_canon_number(t) if t[-1].isdigit() else t for t in _TOKEN_RE.findall(s)]
    return [t for t in toks if t not in _ARTICLES]


def _is_multi(x) -> bool:
    return isinstance(x, (list, tuple))


def exact_match(prediction, gold) -> float:
    pt = [normalize_tokens(p) for p in prediction] if _is_multi(prediction) else None
    gt = [normalize_tokens(g) for g in gold] if _is_multi(gold) else None
    if pt is not None or gt is not None:
        pt = pt if pt is not None else [normalize_tokens(prediction)]
        gt = gt if gt is not None else [normalize_tokens(gold)]
        key = lambda items: sorted(" ".join(i) for i in items if i)
        return float(key(pt) == key(gt)) if key(gt) else float(not key(pt))
    return float(normalize_tokens(prediction) == normalize_tokens(gold))


def _flat(x) -> list[str]:
    if _is_multi(x):
        return [t for item in x for t in normalize_tokens(item)]
    return normalize_tokens(x)


def f1_score(prediction, gold) -> float:
    p, g = _flat(prediction), _flat(gold)
    if not p or not g:                      # empty prediction (or gold) -> 1 only if both empty
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def evaluate_answers(predictions: list, golds: list) -> dict:
    """Mean EM and F1 over aligned lists. Raises if lengths differ."""
    if len(predictions) != len(golds):
        raise ValueError("predictions and golds must have the same length")
    n = len(golds)
    if n == 0:
        return {"n": 0, "em": None, "f1": None}
    return {"n": n,
            "em": sum(exact_match(p, g) for p, g in zip(predictions, golds)) / n,
            "f1": sum(f1_score(p, g) for p, g in zip(predictions, golds)) / n}

