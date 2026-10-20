"""BM25 ranking + reranking logic. Tested with a FakeEncoder to verify logic."""
import numpy as np
import pytest
from src.retriever import BM25Retriever, BM25ContrieverRetriever, tokenize, reserve_row_slots


def P(pid, text, doc="d1"):
    return {"passage_id": pid, "text": text, "source_type": "paragraph", "doc_id": doc}


CORPUS = [
    P("a", "Net sales increased to 1,452.4 million in fiscal 2019 driven by fixed price contracts"),
    P("b", "The effective tax rate was 21 percent"),
    P("c", "Revenue from cloud services grew in 2018", doc="d2"),
    P("d", "Employees received stock options vesting over four years", doc="d2"),
]


def test_tokenize_keeps_numbers_whole():
    assert tokenize("Sales were $1,452.4 in 2019.") == ["sales", "were", "1452.4", "in", "2019"]


def test_bm25_ranks_and_format():
    r = BM25Retriever(CORPUS).retrieve("what were net sales in fiscal 2019", top_k=3)
    assert [x["rank"] for x in r] == [1, 2, 3]
    assert r[0]["passage_id"] == "a"
    assert all(r[i]["bm25_score"] >= r[i + 1]["bm25_score"] for i in range(2))


def test_bm25_doc_scope_restricts_results():
    r = BM25Retriever(CORPUS).retrieve("net sales 2019", top_k=10, doc_id="d2")
    assert {x["passage_id"] for x in r} == {"c", "d"}


def test_reserve_row_slots():
    results = [
        {"passage_id": "r1", "source_type": "table_row", "rank": 1},
        {"passage_id": "p1", "source_type": "paragraph", "rank": 2},
        {"passage_id": "p2", "source_type": "paragraph", "rank": 3},
        {"passage_id": "p3", "source_type": "paragraph", "rank": 4},
        {"passage_id": "p4", "source_type": "paragraph", "rank": 5},
        {"passage_id": "r2", "source_type": "table_row", "rank": 6},
    ]
    slotted = reserve_row_slots(results, n_paragraphs=4, k=5)
    top5 = slotted[:5]
    assert sum(p["source_type"] == "paragraph" for p in top5) == 4
    assert sum(p["source_type"] == "table_row" for p in top5) == 1

