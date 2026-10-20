"""Unit tests for Program Synthesizer (src/program_synthesizer.py)."""
import pytest
from src.evidence_tagger import tag_operands
from src.program_synthesizer import get_program_citations, synthesize_program


def test_synthesize_subtraction():
    text = "Revenues were $120M in 2019 and $100M in 2018."
    ops = tag_operands(text)
    prog = synthesize_program("What was the difference in revenue?", ops)
    assert prog.startswith("sub(")
    assert "$120M" in prog or "$120" in prog


def test_synthesize_lookup():
    text = "The effective tax rate was 21%."
    ops = tag_operands(text)
    prog = synthesize_program("What was the tax rate?", ops)
    assert prog.startswith("lookup(")
    assert "21%" in prog


def test_synthesize_addition():
    text = "Segment A contributed $50M and Segment B contributed $70M."
    ops = tag_operands(text)
    prog = synthesize_program("What was the total combined revenue?", ops)
    assert prog.startswith("add(")


def test_program_citations():
    passages = [
        {"passage_id": "doc1::p1", "text": "Revenues were $100M."},
        {"passage_id": "doc1::p2", "text": "Operating profit was $20M."},
    ]
    ops = tag_operands(passages)
    prog = synthesize_program("What was the total?", ops)
    citations = get_program_citations(prog, ops)
    assert len(citations) > 0
    assert "doc1::p1" in citations

