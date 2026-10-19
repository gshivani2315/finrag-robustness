"""Passage builder / loader tests: synthetic docs always; real data if present."""
import os
import pytest
from src.passage_builder import (build_corpus, build_document_passages, detect_header_rows,
                                 paragraph_passage_id, table_row_passage_id, PARAGRAPH, TABLE_ROW)
from src.data_loader import build_qa_records, load_split, COMPLETE, PARTIAL, NO_GOLD

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
    assert detect_header_rows([["Label", "1"], ["A", "2"]]) == 1   # no empty-first rows -> row 0
    assert detect_header_rows([["", "a"]]) == 0                    # would leave no data rows


def test_passage_ids_and_types():
    ps = build_document_passages(DOC, "dev")
    ids = [p["passage_id"] for p in ps]
    assert ids == ["T1::p1", "T1::p2", "T1::r2", "T1::r4"]         # empty row 3 skipped
    assert [p["source_type"] for p in ps] == [PARAGRAPH, PARAGRAPH, TABLE_ROW, TABLE_ROW]
    assert ps[1]["text"] == "Second paragraph."                    # whitespace cleaned
    assert "Fixed Price" in ps[2]["text"] and "2019" in ps[2]["text"]
    assert ps == build_document_passages(DOC, "dev")               # deterministic / stable


def test_corpus_unique_ids_and_dup_detection():
    assert len(build_corpus([DOC], "dev")) == 4
    with pytest.raises(ValueError):
        build_corpus([DOC, DOC], "dev")                            # same table uid twice


def test_gold_mapping_and_status():
    recs = {r["question_id"]: r for r in build_qa_records([DOC], "dev")}
    assert recs["q1"]["gold_evidence_ids"] == [paragraph_passage_id("T1", 2)]
    assert recs["q1"]["evidence_status"] == COMPLETE
    assert recs["q2"]["evidence_status"] == PARTIAL and recs["q2"]["answer_text"] == "3.5"
    assert recs["q3"]["gold_evidence_ids"] == [] and recs["q3"]["evidence_status"] == NO_GOLD
    assert recs["q4"]["gold_evidence_ids"] == [] and recs["q4"]["invalid_rel_paragraphs"] == ["9"]
    assert recs["q1"]["answer_text"] == "x"


@pytest.mark.skipif(not os.path.exists("data/tatqa_dataset_dev.json"), reason="dev data not downloaded")
def test_real_dev_data():
    docs, recs = load_split("dev")
    corpus = build_corpus(docs, "dev")
    ids = {p["passage_id"] for p in corpus}
    assert len(ids) == len(corpus)                                  # unique
    assert len(recs) == 1668 and len(docs) == 278
    # every gold id must exist in the corpus (gold mapping is consistent)
    for r in recs:
        assert set(r["gold_evidence_ids"]) <= ids
    # no 'text'/'table-text' question lacks gold in dev; table-only mostly lacks it
    assert all(r["evidence_status"] != NO_GOLD for r in recs if r["answer_from"] in ("text", "table-text"))
    # rebuilding gives identical ids (stability)
    assert [p["passage_id"] for p in build_corpus(docs, "dev")] == [p["passage_id"] for p in corpus]


@pytest.mark.skipif(not all(os.path.exists(f"data/tatqa_dataset_{s}.json") for s in ("train", "dev", "test")),
                    reason="data not downloaded")
def test_no_cross_split_leakage():
    sets = {s: {d["table"]["uid"] for d in load_split(s)[0]} for s in ("train", "dev", "test")}
    assert not (sets["train"] & sets["dev"]) and not (sets["train"] & sets["test"]) \
        and not (sets["dev"] & sets["test"])


def test_table_label_context_is_search_only():
    base = build_document_passages(DOC, "dev")
    ctx = build_document_passages(DOC, "dev", context="table_labels")
    assert all("search_text" not in p for p in base)                       # baseline untouched
    assert [p["passage_id"] for p in ctx] == [p["passage_id"] for p in base]  # ids unchanged
    paras = [p for p in ctx if p["source_type"] == PARAGRAPH]
    assert all("Fixed Price" in p["search_text"] and "2019" in p["search_text"] for p in paras)
    assert [p["text"] for p in ctx] == [p["text"] for p in base]           # display text unchanged
    assert all("search_text" not in p for p in ctx if p["source_type"] == TABLE_ROW)
    with pytest.raises(ValueError):
        build_document_passages(DOC, "dev", context="bogus")