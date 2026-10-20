"""Check that a synthesized program is grounded in the passage and sane."""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple, Union

PROGRAM_RE = re.compile(r"^(\w+)\((.*)\)$")
KNOWN_OPS = {"add", "sub", "ratio", "yoy", "lookup", "average", "count"}


def _split_args(arg_str: str) -> List[str]:
    return [a.strip() for a in arg_str.split(", ") if a.strip()]


def _unit(span: str) -> str:
    if "%" in span:
        return "percent"
    if "$" in span:
        return "currency"
    return "number"


def _to_float(span: str) -> float:
    cleaned = span.replace("$", "").replace("%", "").replace(",", "").strip()
    # Handle scale words inside span
    scale_mult = 1.0
    s_lower = cleaned.lower()
    if s_lower.endswith("million") or s_lower.endswith("m"):
        scale_mult = 1.0
        cleaned = re.sub(r"(?i)million|m", "", cleaned).strip()
    elif s_lower.endswith("billion") or s_lower.endswith("b"):
        scale_mult = 1.0
        cleaned = re.sub(r"(?i)billion|b", "", cleaned).strip()
    elif s_lower.endswith("thousand") or s_lower.endswith("k"):
        scale_mult = 1.0
        cleaned = re.sub(r"(?i)thousand|k", "", cleaned).strip()
    return float(cleaned) * scale_mult


def _in_context(arg: str, context: str) -> bool:
    # Match the argument as a whole token, so "1" does not match "1452.4".
    pattern = r"(?<![\w.])" + re.escape(arg) + r"(?![\w.]|\d)"
    return re.search(pattern, context) is not None


def verify_program(program: str, operands: List[Dict[str, Any]],
                   context: Union[str, List[Union[str, Dict[str, Any]]]]) -> Tuple[bool, str]:
    """Return (is_valid, error_message). Checks run in order and stop at first failure."""
    if isinstance(context, list):
        full_context = " \n ".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in context
        )
    else:
        full_context = str(context)

    m = PROGRAM_RE.match(program.strip())
    if not m:
        return False, f"malformed program: {program}"
    op, arg_str = m.group(1), m.group(2)
    if op not in KNOWN_OPS:
        return False, f"unknown operation: {op}"
    args = _split_args(arg_str)

    if op == "count":
        return True, ""

    # Check 1: Pointer validity. Every argument must appear in the context.
    for a in args:
        if not _in_context(a, full_context):
            return False, f"operand not found in context: {a}"

    if op == "lookup":
        if not args:
            return False, "lookup has no operand"
        return True, ""

    # Check 2: Unit consistency. All operands must share one unit.
    known_units = {d.get("span", ""): _unit(d.get("span", "")) for d in operands}
    units = {known_units.get(a, _unit(a)) for a in args}
    if len(units) > 1:
        return False, f"mixed units in {op}: {sorted(units)}"

    # Check 3: Reasonableness. The result must be a finite number.
    try:
        vals = [_to_float(a) for a in args]
    except ValueError:
        return False, f"could not read numbers from: {args}"

    if op == "ratio":
        if len(vals) < 2 or vals[1] == 0:
            return False, "ratio with zero or missing denominator"
        result = vals[0] / vals[1]
    elif op == "sub":
        if len(vals) < 2:
            return False, "sub requires 2 operands"
        result = vals[0] - vals[1]
    elif op == "add":
        result = sum(vals)
    elif op == "average":
        if not vals:
            return False, "average requires at least 1 operand"
        result = sum(vals) / len(vals)
    else:  # yoy: percent change from second value to first
        if len(vals) < 2 or vals[1] == 0:
            return False, "yoy with zero or missing base"
        result = (vals[0] - vals[1]) / vals[1] * 100

    if not math.isfinite(result):
        return False, f"non-finite result from {op}"
    return True, ""
