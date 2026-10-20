"""Map a question's keywords to a simple arithmetic program over operands."""
from typing import Dict, List

# Checked in order, so more specific patterns come first.
RULES = [
    (("increase", "growth", "grow", "yoy"), "yoy"),
    (("subtract", "difference", "minus", "decrease", "less", "change", "from"), "sub"),
    (("ratio", "percent", "percentage", "divide", "share"), "ratio"),
    (("add", "total", "combined", "sum", "together"), "add"),
]


def _detect_operation(query: str) -> str:
    words = query.lower().replace("?", " ").replace(",", " ").split()
    for keywords, op in RULES:
        if any(k in words for k in keywords):
            return op
    return "lookup"  # no arithmetic keyword: return a single value


def synthesize_program(query: str, operands: List[Dict]) -> str:
    """Return a program string such as 'sub($120M, $100M)'.

    Operands are used in the order the tagger found them.
    Returns 'lookup(op1)' when the query needs no arithmetic.
    """
    op = _detect_operation(query)
    spans = [d["span"] for d in operands]

    if op == "lookup":
        if not spans:
            return "lookup()"
        return f"lookup({spans[0]})"

    if len(spans) < 2:
        return f"{op}({', '.join(spans)})"

    return f"{op}({spans[0]}, {spans[1]})"
