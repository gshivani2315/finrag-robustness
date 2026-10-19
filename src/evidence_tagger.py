"""Find numeric operands in a passage or table row."""
import re
from typing import Dict, List

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
SCALE_WORDS = {"million": "M", "billion": "B", "thousand": "K"}


def _overlaps(start: int, end: int, taken: List[tuple]) -> bool:
    return any(start < t_end and end > t_start for t_start, t_end in taken)


def tag_operands(passage: str) -> List[Dict]:
    """Return tagged numeric operands, in the order they appear in the text."""
    found = []
    taken = []

    for m in PERCENT_RE.finditer(passage):
        value = float(m.group(0).replace("%", "").strip())
        found.append({"span": m.group(0), "type": "percent", "value": value,
                      "start": m.start()})
        taken.append((m.start(), m.end()))

    for m in YEAR_RE.finditer(passage):
        if _overlaps(m.start(), m.end(), taken):
            continue
        found.append({"span": m.group(0), "type": "year",
                      "value": int(m.group(0)), "start": m.start()})
        taken.append((m.start(), m.end()))

    for m in NUMBER_RE.finditer(passage):
        if _overlaps(m.start(), m.end(), taken):
            continue
        raw = m.group("num").replace(",", "")
        value = float(raw)
        if m.group("open") and m.group("close"):
            value = -value  # accounting negative, e.g. (3.2)
        scale = m.group("scale")
        if scale:
            scale = SCALE_WORDS.get(scale.lower(), scale.upper())
        kind = "currency" if m.group("cur") else "number"
        found.append({"span": m.group(0).strip(), "type": kind, "value": value,
                      "scale": scale, "start": m.start()})
        taken.append((m.start(), m.end()))

    found.sort(key=lambda d: d["start"])
    for d in found:
        d.pop("start")
    return found
