"""Unit tests for Verifier (src/verifier.py)."""
import pytest
from src.evidence_tagger import tag_operands
from src.verifier import verify_program


def test_verify_pointer_validity():
    context = "Sales were $100M in 2021."
    ops = tag_operands(context)

    # Valid pointer
    valid, msg = verify_program("lookup($100M)", ops, context)
    assert valid

    # Hallucinated pointer
    valid, msg = verify_program("lookup($999M)", ops, context)
    assert not valid
    assert "not found in context" in msg


def test_verify_unit_consistency():
    context = "Sales were $100M while profit margin was 15%."
    ops = tag_operands(context)

    # Mixing currency and percentage in subtraction
    valid, msg = verify_program("sub($100M, 15%)", ops, context)
    assert not valid
    assert "mixed units" in msg


def test_verify_reasonableness():
    context = "Numerator was 50 and denominator was 0."
    ops = tag_operands(context)

    valid, msg = verify_program("ratio(50, 0)", ops, context)
    assert not valid
    assert "zero" in msg or "non-finite" in msg

