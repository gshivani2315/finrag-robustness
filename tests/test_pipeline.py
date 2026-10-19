import json, os, subprocess, sys
import pytest
from src.pipeline import sample_questions, run_batch, save_jsonl, load_jsonl, summarize, retrieve
from src.retriever import BM25Retriever
from src.passage_builder import build_corpus
from src.data_loader import build_qa_records
from tests.test_passages_and_data import DOC

DOC2 = json.loads(json.dumps(DOC).replace('"T1"', '"T2"'))     # second doc, unique ids


def _setup():
    docs = [DOC, DOC2]
    corpus = build_corpus(docs, "dev")
    recs = build_qa_records(docs, "dev")
    unj = {p["passage_id"] for p in corpus if p["source_type"] == "table_row"}
    return corpus, recs, unj


def test_sampling_is_deterministic_and_order_independent():
    _, recs, _ = _setup()
    a = sample_questions(recs, 3, seed=7)
    b = sample_questions(list(reversed(recs)), 3, seed=7)
    assert [r["question_id"] for r in a] == [r["question_id"] for r in b]
    ev = sample_questions(recs, None, only_evaluable=True)
    assert all(r["evidence_status"] != "no_gold" for r in ev)


def test_batch_roundtrip_and_no_leakage_of_gold_into_ranking(tmp_path):
    corpus, recs, unj = _setup()
    ret = BM25Retriever(corpus)
    rows = run_batch(ret, recs, unj, store_k=5)
    p = tmp_path / "r.jsonl"; save_jsonl(rows, str(p))
    back = load_jsonl(str(p))                                   # read by separate code path
    assert back == json.loads(json.dumps(rows))
    # retrieval for a question must equal retrieval from the question text alone
    r0 = back[0]
    assert [x["passage_id"] for x in r0["retrieved"]] == \
           [x["passage_id"] for x in ret.retrieve(r0["question"], top_k=5)]
    assert r0["retrieval_metrics"] is not None                  # q1 has gold
    assert any(r["retrieval_metrics"] is None for r in back)    # table-only q has none
    s = summarize(back)
    assert s["n_questions"] == 8 and s["n_evaluated"] + s["n_excluded"] == 8


@pytest.mark.skipif(not os.path.exists("data/tatqa_dataset_dev.json"), reason="dev data not downloaded")
def test_public_retrieve_interface_real_dev():
    out = retrieve("What is the amount of total sales in 2019?", top_k=5, method="bm25", split="dev")
    assert out["method"] == "bm25" and len(out["results"]) == 5
    assert [r["rank"] for r in out["results"]] == [1, 2, 3, 4, 5]


@pytest.mark.skipif(not os.path.exists("data/tatqa_dataset_dev.json"), reason="dev data not downloaded")
def test_retrieve_with_slots_matches_offline_arrangement():
    from src.retriever import reserve_row_slots
    q = "What is the amount of total sales in 2019?"
    base = retrieve(q, top_k=100, method="bm25", split="dev", paragraph_context="table_labels")["results"]
    want = [x["passage_id"] for x in reserve_row_slots(base, 4, 5)[:5]]
    got = retrieve(q, top_k=5, method="bm25", split="dev", paragraph_context="table_labels", reserve_paragraph_slots=4)
    assert [x["passage_id"] for x in got["results"]] == want
    assert got["reserve_paragraph_slots"] == 4 and all("Table context" not in x["text"] for x in got["results"])
    plain = retrieve(q, top_k=5)                      # defaults still reproduce the original baseline
    assert plain["paragraph_context"] == "none" and plain["reserve_paragraph_slots"] == 0