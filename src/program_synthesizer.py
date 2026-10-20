"""Map a question's keywords and tagged operands to an arithmetic or lookup program."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

RULES = [
    (("increase", "growth", "grow", "yoy", "percentage change", "rate of change"), "yoy"),
    (("subtract", "difference", "minus", "decrease", "less", "change", "from", "decline"), "sub"),
    (("ratio", "percent", "percentage", "divide", "share", "proportion"), "ratio"),
    (("add", "total", "combined", "sum", "together", "aggregate", "overall"), "add"),
    (("average", "mean"), "average"),
    (("count", "how many"), "count"),
]


def _detect_operation(query: str) -> str:
    words = re.findall(r"[a-z]+", query.lower())
    for keywords, op in RULES:
        if any(k in words or k in query.lower() for k in keywords):
            return op
    return "lookup"


def _filter_usable_operands(query: str, operands: List[Dict[str, Any]], op: str) -> List[Dict[str, Any]]:
    """Filter operands to prefer numbers/currencies over metadata years for arithmetic/lookup."""
    if not operands:
        return []

    # If query asks about a year specifically (e.g. "what year", "which year")
    q_lower = query.lower()
    if "which year" in q_lower or "what year" in q_lower:
        years = [op for op in operands if op.get("type") == "year"]
        if years:
            return years

    # For arithmetic and monetary lookup, filter out year tokens if other numbers exist
    non_years = [op for op in operands if op.get("type") != "year"]
    if non_years:
        return non_years
    return operands


def synthesize_program(query: str, operands: List[Dict[str, Any]]) -> str:
    """Return a program string such as 'sub($120M, $100M)' or 'lookup($100M)'.

    Operands are chosen based on query operation and relevance.
    """
    op = _detect_operation(query)
    usable = _filter_usable_operands(query, operands, op)
    spans = [d["span"] for d in usable]

    if op == "count":
        return f"count({len(spans)})" if spans else "count(0)"

    if op == "lookup":
        if not spans:
            return "lookup()"
        return f"lookup({spans[0]})"

    if len(spans) < 2:
        return f"{op}({', '.join(spans)})" if spans else f"{op}()"

    if op == "add":
        # Can sum up to 4 operands if relevant, default first 2
        return f"add({spans[0]}, {spans[1]})"

    return f"{op}({spans[0]}, {spans[1]})"


def get_program_citations(program: str, operands: List[Dict[str, Any]]) -> List[str]:
    """Retrieve passage IDs cited by the arguments of the program."""
    citations = []
    seen = set()
    span_to_pid = {op.get("span"): op.get("passage_id") for op in operands if op.get("passage_id")}
    for span, pid in span_to_pid.items():
        if span in program and pid not in seen:
            citations.append(pid)
            seen.add(pid)
    return citations
