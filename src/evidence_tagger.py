"""Find numeric operands in a passage or table row."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Union

# Order matters: percentages and years are matched before plain numbers,
# so a span is never tagged twice.
PERCENT_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?\s*%")
YEAR_RE = re.compile(r"(?<![\w.,])(?:19|20)\d{2}(?![\w.,]|\d)")
NUMBER_RE = re.compile(
    r"(?P<open>\()?\s*(?P<cur>\$)?\s*"
    r"(?P<num>-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
    r"\s*(?P<scale>million|billion|thousand|[MBK])?\b"
    r"(?P<close>\))?",
    re.IGNORECASE,
)
SCALE_WORDS = {"million": "million", "billion": "billion", "thousand": "thousand",
               "m": "million", "b": "billion", "k": "thousand"}


def _overlaps(start: int, end: int, taken: List[tuple]) -> bool:
    return any(start < t_end and end > t_start for t_start, t_end in taken)


def tag_operands(passage: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    """Return tagged numeric operands, in the order they appear in the text.

    Supports:
      - A single text string
      - A passage dict with 'text' and optional 'passage_id'
      - A list of text strings or passage dicts
    """
    if isinstance(passage, list):
        results = []
        for p in passage:
            results.extend(tag_operands(p))
        return results

    passage_id = None
    if isinstance(passage, dict):
        text = str(passage.get("text", ""))
        passage_id = passage.get("passage_id")
    else:
        text = str(passage)

    found = []
    taken = []

    for m in PERCENT_RE.finditer(text):
        raw_val = m.group(0).replace("%", "").strip()
        try:
            value = float(raw_val)
        except ValueError:
            continue
        op = {"span": m.group(0).strip(), "type": "percent", "value": value,
              "scale": "percent", "start": m.start()}
        if passage_id:
            op["passage_id"] = passage_id
        found.append(op)
        taken.append((m.start(), m.end()))

    for m in YEAR_RE.finditer(text):
        if _overlaps(m.start(), m.end(), taken):
            continue
        op = {"span": m.group(0).strip(), "type": "year",
              "value": int(m.group(0)), "scale": "", "start": m.start()}
        if passage_id:
            op["passage_id"] = passage_id
        found.append(op)
        taken.append((m.start(), m.end()))

    for m in NUMBER_RE.finditer(text):
        if _overlaps(m.start(), m.end(), taken):
            continue
        raw = m.group("num").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if m.group("open") and m.group("close"):
            value = -value  # accounting negative, e.g. (3.2)
        scale_match = m.group("scale")
        scale = SCALE_WORDS.get(scale_match.lower(), "") if scale_match else ""
        kind = "currency" if m.group("cur") else "number"
        span = m.group(0).strip()
        op = {"span": span, "type": kind, "value": value,
              "scale": scale, "start": m.start()}
        if passage_id:
            op["passage_id"] = passage_id
        found.append(op)
        taken.append((m.start(), m.end()))

    found.sort(key=lambda d: d["start"])
    for d in found:
        d.pop("start", None)
    return found
