"""Unit tests for Evidence Tagger (src/evidence_tagger.py)."""
import pytest
from src.evidence_tagger import tag_operands


def test_tag_currency_and_scale():
    text = "Net sales increased to $1,452.4 million in fiscal 2019 compared to $1,146.2 million in 2018."
    operands = tag_operands(text)
    spans = [op["span"] for op in operands]
    assert "$1,452.4 million" in spans or "$1,452.4" in spans
    types = {op["type"] for op in operands}
    assert "currency" in types
    assert "year" in types


def test_tag_percentages_and_negatives():
    text = "Operating margin was (3.2)% and net loss was $(45.0) million."
    operands = tag_operands(text)
    vals = [op["value"] for op in operands]
    assert any(v < 0 for v in vals)


def test_tag_passage_dict_with_id():
    passage = {
        "passage_id": "doc1::p1",
        "text": "Total revenues were $100M in 2021.",
        "source_type": "paragraph",
    }
    operands = tag_operands(passage)
    assert len(operands) >= 2
    assert all(op["passage_id"] == "doc1::p1" for op in operands)


def test_tag_list_of_passages():
    passages = [
        {"passage_id": "doc1::p1", "text": "Revenues were $100 million."},
        {"passage_id": "doc1::p2", "text": "Operating margin was 15%."},
    ]
    operands = tag_operands(passages)
    assert len(operands) >= 2
    pids = {op["passage_id"] for op in operands}
    assert pids == {"doc1::p1", "doc1::p2"}

