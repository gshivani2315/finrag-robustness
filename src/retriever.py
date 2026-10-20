"""
retriever.py
------------
Two retrievers with the SAME output format:

  BM25Retriever            - lexical first-stage retrieval (rank-bm25, BM25Okapi)
  BM25ContrieverRetriever  - BM25 top-N candidates, reranked by Contriever

Neither retriever ever sees the gold answer or gold evidence: they only get
the question text (and, optionally, a doc_id - see below).

Result format (a list of dicts, best first):
  {"passage_id", "text", "rank", "source_type", "doc_id",
   "bm25_score", "bm25_rank",                 # always present
   "contriever_score"}                        # only for bm25+contriever

SEARCH SCOPE
  doc_id=None  -> search the whole split corpus (realistic, harder; DEFAULT)
  doc_id="..." -> only search that document's own passages ("oracle document"
                  setting: uses the knowledge of WHICH document the question
                  belongs to, but never the answer or the gold evidence).
"""

from __future__ import annotations

import os
import re

import numpy as np
from rank_bm25 import BM25Okapi

# Keep numbers whole ("1,452.4" -> "1452.4") and split everything else into
# lowercase words. Financial questions depend heavily on numbers/years.
_TOKEN_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?|[a-z]+")


def tokenize(text: str) -> list[str]:
    return [t.replace(",", "") for t in _TOKEN_RE.findall(text.lower())]


def passage_text_for_search(p: dict) -> str:
    """Text that is indexed / encoded: `search_text` if the passage has one, else the original text."""
    return p.get("search_text", p["text"])


def passage_cache_key(p: dict) -> str:
    """
    Embedding-cache key. Passages WITHOUT search_text keep the old key (the plain passage id), so existing
    caches stay valid. Passages WITH search_text get id + a hash of that text, so enriched and original
    embeddings can never be mixed up.
    """
    if "search_text" not in p:
        return p["passage_id"]
    import hashlib
    return p["passage_id"] + "#" + hashlib.md5(p["search_text"].encode("utf-8")).hexdigest()[:10]


def _public(p: dict) -> dict:
    return {"passage_id": p["passage_id"], "text": p["text"],
            "source_type": p["source_type"], "doc_id": p["doc_id"]}


def reserve_row_slots(results: list[dict], n_paragraphs: int = 4, k: int = 5) -> list[dict]:
    """
    Evidence-slot rule: of the first `k` slots, give `n_paragraphs` to the best-ranked paragraphs and the
    remaining `k - n_paragraphs` to the best-ranked table rows. Inside the first k, items keep their
    original relative order; everything after the first k stays in the original order.
    Why: table-aware paragraph text lets paragraphs crowd table rows out of the top 5, and table rows are
    unlabelled in TAT-QA, so no metric would show that loss. Ranks are renumbered 1..N; the old rank is
    kept as `rank_before_slots`.
    """
    if not 0 <= n_paragraphs <= k:
        raise ValueError("need 0 <= n_paragraphs <= k")
    paras = [x for x in results if x["source_type"] == "paragraph"][:n_paragraphs]
    rows = [x for x in results if x["source_type"] == "table_row"][:k - n_paragraphs]
    pos = {x["passage_id"]: i for i, x in enumerate(results)}
    top = sorted(paras + rows, key=lambda x: pos[x["passage_id"]])
    used = {x["passage_id"] for x in top}
    out = top + [x for x in results if x["passage_id"] not in used]
    return [dict(x, rank=i + 1, rank_before_slots=x["rank"]) for i, x in enumerate(out)]


# ---------------------------------------------------------------- BM25
class BM25Retriever:
    name = "bm25"

    def __init__(self, passages: list[dict], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self._bm25 = BM25Okapi([tokenize(passage_text_for_search(p)) for p in passages], k1=k1, b=b)
        self._doc_ids = np.array([p["doc_id"] for p in passages])
        self.by_id = {p["passage_id"]: p for p in passages}

    def retrieve(self, question: str, top_k: int = 5, doc_id: str | None = None) -> list[dict]:
        scores = np.asarray(self._bm25.get_scores(tokenize(question)), dtype=float)
        if doc_id is not None:
            scores = np.where(self._doc_ids == doc_id, scores, -np.inf)
        # stable sort -> ties broken by corpus order -> fully deterministic
        order = np.argsort(-scores, kind="stable")[:top_k]
        out = []
        for j in order:
            if not np.isfinite(scores[j]):
                break
            out.append({**_public(self.passages[j]), "rank": len(out) + 1,
                        "bm25_score": float(scores[j]), "bm25_rank": len(out) + 1})
        return out


# ---------------------------------------------------------------- Contriever
class ContrieverEncoder:
    """
    facebook/contriever via Hugging Face Transformers.
    Why plain Transformers (not Sentence-Transformers): the checkpoint has no
    sentence-transformers config, and Contriever REQUIRES mean pooling over
    tokens (not the [CLS] token). Doing it by hand makes that explicit.

    Passage embeddings are cached (by passage id) in an .npz file so that
    each passage is encoded at most once across questions and across runs.
    """

    def __init__(self, model_name: str = "facebook/contriever", device: str | None = None,
                 batch_size: int = 16, max_length: int = 256,
                 cache_dir: str | None = "data/cache"):
        import torch  # imported lazily so BM25-only users don't need torch
        from transformers import AutoModel, AutoTokenizer
        self._torch = torch
        self.model_name, self.batch_size, self.max_length = model_name, batch_size, max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self._cache: dict[str, np.ndarray] = {}
        self._cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            slug = model_name.replace("/", "_")
            self._cache_path = os.path.join(cache_dir, f"{slug}_len{max_length}.npz")
            if os.path.exists(self._cache_path):
                z = np.load(self._cache_path, allow_pickle=False)
                self._cache = dict(zip(z["ids"].tolist(), z["vecs"]))

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        chunks = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(**batch).last_hidden_state           # (B, T, H)
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)   # mean pooling
            chunks.append(emb.cpu().numpy().astype("float32"))
        return np.vstack(chunks) if chunks else np.zeros((0, 768), dtype="float32")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode_texts(texts)

    def encode_passages(self, passages: list[dict]) -> np.ndarray:
        missing = [p for p in passages if passage_cache_key(p) not in self._cache]
        if missing:
            vecs = self._encode_texts([passage_text_for_search(p) for p in missing])
            for p, v in zip(missing, vecs):
                self._cache[passage_cache_key(p)] = v
        return np.vstack([self._cache[passage_cache_key(p)] for p in passages])

    def save_cache(self) -> None:
        if self._cache_path and self._cache:
            ids = np.array(list(self._cache.keys()))
            np.savez(self._cache_path, ids=ids, vecs=np.vstack(list(self._cache.values())))


class BM25ContrieverRetriever:
    """BM25 -> top `candidate_n` -> rerank by Contriever similarity -> top_k."""
    name = "bm25+contriever"

    def __init__(self, bm25: BM25Retriever, encoder, candidate_n: int = 100,
                 similarity: str = "dot"):
        if similarity not in ("dot", "cosine"):
            raise ValueError("similarity must be 'dot' or 'cosine'")
        self.bm25, self.encoder = bm25, encoder
        self.candidate_n, self.similarity = candidate_n, similarity

    def retrieve(self, question: str, top_k: int = 5, doc_id: str | None = None) -> list[dict]:
        cands = self.bm25.retrieve(question, top_k=self.candidate_n, doc_id=doc_id)
        if not cands:
            return []
        q = self.encoder.encode_queries([question])[0]
        # encode the CORPUS passages (they carry search_text when table context is on), not the output rows
        P = self.encoder.encode_passages([self.bm25.by_id[c["passage_id"]] for c in cands])
        if self.similarity == "cosine":
            q = q / (np.linalg.norm(q) + 1e-12)
            P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        sims = P @ q
        order = np.argsort(-sims, kind="stable")[:top_k]
        out = []
        for new_rank, j in enumerate(order, start=1):
            c = cands[j]
            out.append({**{k: c[k] for k in ("passage_id", "text", "source_type", "doc_id")},
                        "rank": new_rank, "bm25_score": c["bm25_score"],
                        "bm25_rank": c["bm25_rank"], "contriever_score": float(sims[j])})
        return out

