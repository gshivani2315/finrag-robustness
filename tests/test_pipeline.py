"""Integration and unit tests for pipeline (src/pipeline.py)."""
import pytest
from src.pipeline import answer_question, answer_with_retrieval, sample_questions, save_jsonl, load_jsonl, _compute


def test_answer_question_lookup():
    passage = "Total revenues in fiscal 2019 were $1,452.4 million."
    res = answer_question("What were the revenues in 2019?", passage)
    assert res["valid"] is True
    assert res["answer"] == 1452.4
    assert res["program"].startswith("lookup(")


def test_answer_question_arithmetic_subtraction():
    passage = "Revenues increased to $150M in 2019 from $100M in 2018."
    res = answer_question("What was the difference in revenues?", passage)
    assert res["valid"] is True
    assert res["answer"] == 50.0
    assert res["program"].startswith("sub(")


def test_compute_program():
    assert _compute("lookup($100)") == 100.0
    assert _compute("add(10, 20)") == 30.0
    assert _compute("sub(50, 20)") == 30.0
    assert _compute("ratio(100, 200)") == 0.5
    assert _compute("yoy(120, 100)") == 20.0
    assert _compute("count(5)") == 5.0
    assert _compute("invalid_program") is None


def test_sample_questions_deterministic():
    records = [{"question_id": f"q_{i}", "evidence_status": "complete"} for i in range(10)]
    s1 = sample_questions(records, n=5, seed=12)
    s2 = sample_questions(records, n=5, seed=12)
    assert [r["question_id"] for r in s1] == [r["question_id"] for r in s2]


def test_jsonl_roundtrip(tmp_path):
    rows = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
    file_path = str(tmp_path / "test.jsonl")
    save_jsonl(rows, file_path)
    loaded = load_jsonl(file_path)
    assert loaded == rows

