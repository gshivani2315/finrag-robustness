"""
ctx_checks.py - is the table-context gain real, and is it bought at the cost of table rows?
Reads only existing run files. Run from the project folder:   python scripts\\ctx_checks.py
For each sample (your 200 questions, and the 315 held-out questions) it prints:
  1. paired differences (table_context - BM25) per question group, with document-level 95% CIs
  2. how many TABLE ROWS sit in the top 5 / top 10 (rows are unlabelled, so the metrics cannot see them)
  3. a ROUGH PROXY for "does the retrieved evidence still contain the answer?": for span / multi-span questions
     whose answer is at least 4 characters, the share whose answer string appears in the top-5 / top-10 passages
     (rows OR paragraphs). The gold answer is used ONLY here, to evaluate; it never influences retrieval.
"""
import argparse, json, math, random

def load(p):
    with open(p, encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]
def ids(row, cond): return [p["passage_id"] for p in row["retrieved"] if not (cond and p["source_type"] == "table_row")]
def recall(r, g, k): g = set(g); return len(g & set(r[:k])) / len(g)
def ndcg(r, g, k):
    g = set(g); dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(r[:k]) if x in g)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(len(g), k)))
def mean(x): x = list(x); return sum(x) / len(x) if x else float("nan")
def boot(d, docs, n=10000, seed=0):
    by = {}
    for v, doc in zip(d, docs): by.setdefault(doc, []).append(v)
    gs, rng, out = list(by.values()), random.Random(seed), []
    for _ in range(n):
        pick = [gs[rng.randrange(len(gs))] for _ in gs]
        out.append(sum(sum(x) for x in pick) / sum(len(x) for x in pick))
    out.sort(); return out[int(.025 * n)], out[int(.975 * n)]
def norm(s): return " ".join(str(s).lower().replace("$", "").replace(",", "").replace("%", "").split())
def items(row):
    a = row["gold_answer"]; return [norm(x) for x in (a if isinstance(a, list) else [a])]
def eligible(row): return row["answer_type"] in ("span", "multi-span") and all(len(x) >= 4 for x in items(row))
def has_answer(row, k):
    text = " || ".join(norm(p["text"]) for p in row["retrieved"][:k]); return all(x in text for x in items(row))
def rows_in_top(row, k): return sum(p["source_type"] == "table_row" for p in row["retrieved"][:k])
grp = lambda r: "text" if r["answer_from"] == "text" else "text+table"

ap = argparse.ArgumentParser()
ap.add_argument("--eval-base", default="outputs/dev200_k100/retrieval_bm25.jsonl")
ap.add_argument("--eval-ctx", default="outputs/exp4_ctx_eval/retrieval_bm25.jsonl")
ap.add_argument("--tune-base", default="outputs/tune/retrieval_bm25.jsonl")
ap.add_argument("--tune-ctx", default="outputs/exp4_ctx_tune/retrieval_bm25.jsonl")
a = ap.parse_args()
for name, bp, cp in (("YOUR 200 QUESTIONS", a.eval_base, a.eval_ctx), ("HELD-OUT 315 QUESTIONS", a.tune_base, a.tune_ctx)):
    B, C = load(bp), load(cp)
    assert [r["question_id"] for r in B] == [r["question_id"] for r in C], "question order differs"
    print(f"\n================ {name} ================")
    print("1) paired differences, table_context - BM25 (condensed), document-level 95% CI")
    for g, sel in (("all", lambda r: True), ("text", lambda r: grp(r) == "text"), ("text+table", lambda r: grp(r) == "text+table")):
        idx = [i for i, r in enumerate(B) if sel(r)]; docs = [B[i]["doc_id"] for i in idx]
        for m, f, k in (("recall@5", recall, 5), ("ndcg@10", ndcg, 10)):
            d = [f(ids(C[i], True), C[i]["gold_evidence_ids"], k) - f(ids(B[i], True), B[i]["gold_evidence_ids"], k) for i in idx]
            lo, hi = boot(d, docs)
            print(f"   {g:11s} n={len(idx):3d} {m:9s} {mean(d):+.4f}  CI [{lo:+.3f}, {hi:+.3f}]  better={sum(x > 1e-12 for x in d)} worse={sum(x < -1e-12 for x in d)}")
    print("2) table rows among the retrieved passages (average count)")
    for g, sel in (("all", lambda r: True), ("text", lambda r: grp(r) == "text"), ("text+table", lambda r: grp(r) == "text+table")):
        idx = [i for i, r in enumerate(B) if sel(r)]
        print(f"   {g:11s} rows in top 5: BM25 {mean(rows_in_top(B[i], 5) for i in idx):.2f} -> table_context {mean(rows_in_top(C[i], 5) for i in idx):.2f}"
              f" | rows in top 10: BM25 {mean(rows_in_top(B[i], 10) for i in idx):.2f} -> table_context {mean(rows_in_top(C[i], 10) for i in idx):.2f}")
    el = [i for i, r in enumerate(B) if eligible(r)]
    print(f"3) rough proxy: answer string found in retrieved passages (rows or paragraphs); {len(el)} eligible span/multi-span questions")
    for k in (5, 10):
        print(f"   top {k:2d}: BM25 {mean(has_answer(B[i], k) for i in el):.3f} -> table_context {mean(has_answer(C[i], k) for i in el):.3f}"
              f"   (questions: gained {sum(has_answer(C[i], k) and not has_answer(B[i], k) for i in el)}, lost {sum(has_answer(B[i], k) and not has_answer(C[i], k) for i in el)})")