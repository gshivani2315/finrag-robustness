"""BM25 ranking + reranking logic. The reranker is tested with a FAKE encoder,
so this verifies the rerank logic, NOT the real facebook/contriever model."""
import numpy as np
from src.retriever import BM25Retriever, BM25ContrieverRetriever, tokenize


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
    assert {"passage_id", "text", "rank", "bm25_score"} <= r[0].keys()


def test_bm25_deterministic_and_top_n():
    ret = BM25Retriever(CORPUS)
    assert ret.retrieve("tax rate", 2) == ret.retrieve("tax rate", 2)
    assert len(ret.retrieve("tax rate", 10)) == len(CORPUS)         # capped at corpus size


def test_bm25_doc_scope_restricts_results():
    r = BM25Retriever(CORPUS).retrieve("net sales 2019", top_k=10, doc_id="d2")
    assert {x["passage_id"] for x in r} == {"c", "d"}


class FakeEncoder:
    """Bag-of-words hashing 'embedding' - only for testing rerank mechanics."""
    def _vec(self, t):
        v = np.zeros(64, dtype="float32")
        for tok in tokenize(t):
            v[hash(tok) % 64] += 1.0
        return v
    def encode_queries(self, texts): return np.vstack([self._vec(t) for t in texts])
    def encode_passages(self, ps): return np.vstack([self._vec(p["text"]) for p in ps])


class ReverseEncoder(FakeEncoder):
    """Gives the BM25-worst candidate the highest score, to prove reordering happens."""
    def encode_passages(self, ps):
        n = len(ps)
        M = np.zeros((n, 2), dtype="float32"); M[:, 0] = np.arange(n)   # later candidate = bigger
        return M
    def encode_queries(self, texts): return np.array([[1.0, 0.0]], dtype="float32")


def test_rerank_reorders_and_keeps_scores():
    bm25 = BM25Retriever(CORPUS)
    base = bm25.retrieve("net sales fiscal 2019", top_k=4)
    rr = BM25ContrieverRetriever(bm25, ReverseEncoder(), candidate_n=4).retrieve("net sales fiscal 2019", top_k=3)
    assert [x["passage_id"] for x in rr] == [b["passage_id"] for b in base[::-1][:3]]
    assert [x["rank"] for x in rr] == [1, 2, 3]
    assert rr[0]["bm25_rank"] == 4 and "contriever_score" in rr[0] and "bm25_score" in rr[0]


def test_rerank_candidate_pool_limits_recall():
    bm25 = BM25Retriever(CORPUS)
    rr = BM25ContrieverRetriever(bm25, FakeEncoder(), candidate_n=2, similarity="cosine")
    assert len(rr.retrieve("net sales fiscal 2019", top_k=10)) == 2   # can only rerank candidates


def test_cache_key_never_mixes_original_and_enriched_text():
    from src.retriever import passage_cache_key, passage_text_for_search
    plain = P("x", "original text")
    enriched = dict(plain, search_text="original text Table context: A; B.")
    other = dict(plain, search_text="original text Table context: C.")
    assert passage_cache_key(plain) == "x"                       # legacy key: old caches stay valid
    assert passage_cache_key(enriched).startswith("x#") and passage_cache_key(enriched) != passage_cache_key(other)
    assert passage_cache_key(enriched) == passage_cache_key(dict(enriched))   # stable
    assert passage_text_for_search(plain) == "original text" and "Table context" in passage_text_for_search(enriched)


def test_reranker_encodes_the_search_text_not_the_output_text():
    seen = []
    class Recorder(FakeEncoder):
        def encode_passages(self, ps):
            seen.extend(p.get("search_text", p["text"]) for p in ps); return super().encode_passages(ps)
    corpus = [dict(p, search_text=p["text"] + " Table context: Fixed Price.") for p in CORPUS]
    bm25 = BM25Retriever(corpus)
    out = BM25ContrieverRetriever(bm25, Recorder(), candidate_n=3).retrieve("net sales fiscal 2019", top_k=3)
    assert seen and all("Table context" in s for s in seen)       # encoder saw the enriched text
    assert all("Table context" not in x["text"] for x in out)     # output rows keep the original text


def test_reserve_row_slots_composition_and_order():
    from src.retriever import reserve_row_slots
    def R(i, kind): return {"passage_id": f"{kind[0]}{i}", "text": "t", "source_type": kind, "doc_id": "d", "rank": i}
    res = [R(1, "table_row"), R(2, "paragraph"), R(3, "paragraph"), R(4, "table_row"), R(5, "paragraph"),
           R(6, "paragraph"), R(7, "paragraph"), R(8, "table_row")]
    out = reserve_row_slots(res, n_paragraphs=4, k=5)
    top5 = out[:5]
    assert [x["passage_id"] for x in top5] == ["t1", "p2", "p3", "p5", "p6"]
    assert sum(x["source_type"] == "table_row" for x in top5) == 1 and sum(x["source_type"] == "paragraph" for x in top5) == 4
    assert [x["rank"] for x in out] == list(range(1, 9)) and len({x["passage_id"] for x in out}) == 8
    assert [x["passage_id"] for x in out[5:]] == ["t4", "p7", "t8"]       # the rest keeps its original order
    assert out[1]["rank_before_slots"] == 2
    assert [x["passage_id"] for x in reserve_row_slots(res, 5, 5)[:5]] == ["p2", "p3", "p5", "p6", "p7"]   # 5 paragraph slots, 0 rows
    only_par = [R(i, "paragraph") for i in range(1, 8)]
    assert [x["passage_id"] for x in reserve_row_slots(only_par, 4, 5)] == [x["passage_id"] for x in only_par]  # no rows: nothing to reserve
    import pytest
    with pytest.raises(ValueError):
        reserve_row_slots(res, 6, 5)