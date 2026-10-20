"""Check that a synthesized program is grounded in the passage and sane."""
import math
import re
from typing import Dict, List, Tuple

PROGRAM_RE = re.compile(r"^(\w+)\((.*)\)$")
KNOWN_OPS = {"add", "sub", "ratio", "yoy", "lookup"}


def _split_args(arg_str: str) -> List[str]:
    return [a.strip() for a in arg_str.split(", ") if a.strip()]


def _unit(span: str) -> str:
    if "%" in span:
        return "percent"
    if "$" in span:
        return "currency"
    return "number"


def _to_float(span: str) -> float:
    return float(span.replace("$", "").replace("%", "").replace(",", "").strip())


def verify_program(program: str, operands: List[Dict], context: str) -> Tuple[bool, str]:
    """Return (is_valid, error_message). Checks run in order and stop at the first failure."""
    m = PROGRAM_RE.match(program.strip())
    if not m:
        return False, f"malformed program: {program}"
    op, arg_str = m.group(1), m.group(2)
    if op not in KNOWN_OPS:
        return False, f"unknown operation: {op}"
    args = _split_args(arg_str)

    # Check 1: pointer validity. Every argument must appear in the passage.
    for a in args:
        if a not in context:
            return False, f"operand not found in context: {a}"

    if op == "lookup":
        return True, ""

    # Check 2: unit consistency. All operands must share one unit.
    known_units = {d["span"]: _unit(d["span"]) for d in operands}
    units = {known_units.get(a, _unit(a)) for a in args}
    if len(units) > 1:
        return False, f"mixed units in {op}: {sorted(units)}"

    # Check 3: reasonableness. The result must be a finite number.
    try:
        vals = [_to_float(a) for a in args]
    except ValueError:
        return False, f"could not read numbers from: {args}"

    if op == "ratio":
        if len(vals) < 2 or vals[1] == 0:
            return False, "ratio with zero or missing denominator"
        result = vals[0] / vals[1]
    elif op == "sub":
        result = vals[0] - vals[1]
    elif op == "add":
        result = sum(vals)
    else:  # yoy: percent change from the second value to the first
        if len(vals) < 2 or vals[1] == 0:
            return False, "yoy with zero or missing base"
        result = (vals[0] - vals[1]) / vals[1] * 100

    if not math.isfinite(result):
        return False, f"non-finite result from {op}"
    return True, ""
