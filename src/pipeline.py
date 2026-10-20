"""End-to-end: tag operands, synthesize a program, verify it, and compute the answer."""
import re
from typing import Dict, Optional

from src.evidence_tagger import tag_operands
from src.program_synthesizer import synthesize_program
from src.verifier import verify_program, _to_float

PROGRAM_RE = re.compile(r"^(\w+)\((.*)\)$")


def _compute(program: str) -> Optional[float]:
    """Evaluate a verified program string. Returns None if it can't be evaluated."""
    m = PROGRAM_RE.match(program.strip())
    if not m:
        return None
    op, arg_str = m.group(1), m.group(2)
    args = [a.strip() for a in arg_str.split(", ") if a.strip()]
    try:
        vals = [_to_float(a) for a in args]
    except ValueError:
        return None

    if op == "lookup":
        return vals[0] if vals else None
    if op == "add":
        return sum(vals)
    if op == "sub":
        return vals[0] - vals[1]
    if op == "ratio":
        return vals[0] / vals[1]
    if op == "yoy":
        return (vals[0] - vals[1]) / vals[1] * 100
    return None


def answer_question(question: str, passage: str) -> Dict:
    """Run the full pipeline on one question and one passage (or table row)."""
    operands = tag_operands(passage)
    program = synthesize_program(question, operands)
    is_valid, error = verify_program(program, operands, passage)

    result = {"program": program, "valid": is_valid, "error": error, "answer": None}
    if is_valid:
        value = _compute(program)
        if value is not None:
            result["answer"] = round(value, 4)
    return result
