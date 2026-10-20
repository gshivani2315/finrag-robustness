"""Passage builder / loader tests: synthetic docs always; real data if present."""
import os
import pytest
from src.passage_builder import (build_corpus, build_document_passages, detect_header_rows,
                                 paragraph_passage_id, table_row_passage_id, PARAGRAPH, TABLE_ROW)
from src.data_loader import build_qa_records, COMPLETE, PARTIAL, NO_GOLD

DOC = {
    "table": {"uid": "T1", "table": [["", "", "Years Ended", ""],
                                     ["", "2019", "2018", "2017"],
                                     ["Fixed Price", "$  1,452.4", "1,146.2", "1,036.9"],
                                     ["", "", "", ""],
                                     ["Other", "44.1", "56.7", "70.8"]]},
    "paragraphs": [{"uid": "p-a", "order": 1, "text": "First paragraph."},
                   {"uid": "p-b", "order": 2, "text": "Second   paragraph."}],
    "questions": [
        {"uid": "q1", "order": 1, "question": "text q", "answer": ["x"], "answer_type": "span",
         "answer_from": "text", "rel_paragraphs": ["2"], "scale": ""},
        {"uid": "q2", "order": 2, "question": "mixed q", "answer": 3.5, "answer_type": "arithmetic",
         "answer_from": "table-text", "rel_paragraphs": ["1"], "scale": ""},
        {"uid": "q3", "order": 3, "question": "table q", "answer": "7", "answer_type": "span",
         "answer_from": "table", "rel_paragraphs": [], "scale": ""},
        {"uid": "q4", "order": 4, "question": "bad ref", "answer": 1, "answer_from": "text",
         "rel_paragraphs": ["9"]},
    ],
}


def test_header_detection():
    assert detect_header_rows(DOC["table"]["table"]) == 2
    assert detect_header_rows([["Label", "1"], ["A", "2"]]) == 1
    assert detect_header_rows([["", "a"]]) == 0


def test_passage_ids_and_types():
    ps = build_document_passages(DOC, "dev")
    ids = [p["passage_id"] for p in ps]
    assert ids == ["T1::p1", "T1::p2", "T1::r2", "T1::r4"]
    assert [p["source_type"] for p in ps] == [PARAGRAPH, PARAGRAPH, TABLE_ROW, TABLE_ROW]
    assert ps[1]["text"] == "Second paragraph."


def test_corpus_unique_ids_and_dup_detection():
    assert len(build_corpus([DOC], "dev")) == 4
    with pytest.raises(ValueError):
        build_corpus([DOC, DOC], "dev")


def test_gold_mapping_and_status():
    recs = {r["question_id"]: r for r in build_qa_records([DOC], "dev")}
    assert recs["q1"]["gold_evidence_ids"] == [paragraph_passage_id("T1", 2)]
    assert recs["q1"]["evidence_status"] == COMPLETE
    assert recs["q2"]["evidence_status"] == PARTIAL
    assert recs["q3"]["gold_evidence_ids"] == [] and recs["q3"]["evidence_status"] == NO_GOLD

