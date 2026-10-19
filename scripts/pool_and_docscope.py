"""
pool_and_docscope.py - two diagnostics on the SAME 200 questions.
  Part 1 (outputs/dev200_k100):      did BM25's top-100 pool contain the gold paragraphs at all?
                                     (a reranker can only reorder what BM25 handed it)
  Part 2 (outputs/dev200_docscope):  results when each question searches ONLY its own document
                                     ("oracle-document": uses which document the question belongs to,
                                     never the answer or gold evidence).
Everything is recomputed from the JSONL files. Run from the project folder:
    python scripts/pool_and_docscope.py
"""
import argparse, json, math, os, random

def load(p):
    with open(p, encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]
def recall(r, g, k): g = set(g); return len(g & set(r[:k])) / len(g)
def ndcg(r, g, k):
    g = set(g); dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(r[:k]) if x in g)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(len(g), k)))
def ids(row, cond): return [p["passage_id"] for p in row["retrieved"] if not (cond and p["source_type"] == "table_row")]
def mean(x): x = list(x); return sum(x) / len(x) if x else float("nan")
def boot(d, docs, n=10000, seed=0):
    by = {}
    for v, doc in zip(d, docs): by.setdefault(doc, []).append(v)
    gs, rng, out = list(by.values()), random.Random(seed), []
    for _ in range(n):
        pick = [gs[rng.randrange(len(gs))] for _ in gs]
        out.append(sum(sum(x) for x in pick) / sum(len(x) for x in pick))
    out.sort(); return out[int(.025 * n)], out[int(.975 * n)]
def group(row): return "text" if row["answer_from"] == "text" else "table-text/table"

ap = argparse.ArgumentParser()
ap.add_argument("--k100", default="outputs/dev200_k100")
ap.add_argument("--doc", default="outputs/dev200_docscope")
ap.add_argument("--ref", default="outputs/dev200", help="earlier top-30 run, used only for a consistency check")
ap.add_argument("--a", default="retrieval_bm25.jsonl")
ap.add_argument("--b", default="retrieval_bm25_contriever.jsonl")
a = ap.parse_args()

# ------------------------------------------------------------------ Part 1
A, B = load(os.path.join(a.k100, a.a)), load(os.path.join(a.k100, a.b))
print("== PART 1: was the gold paragraph inside BM25's top-100 candidate pool? ==")
print("rows:", len(A), len(B), "| same question ids:", [r["question_id"] for r in A] == [r["question_id"] for r in B])
print("stored passages per question: min", min(len(r["retrieved"]) for r in A), "max", max(len(r["retrieved"]) for r in A))
print("CHECK Contriever's list is the same set of passages as BM25's top-100 pool:",
      all(set(ids(x, False)) == set(ids(y, False)) for x, y in zip(A, B)))
pools = [set(ids(r, False)) for r in A]
for name, sel in (("all 200", lambda r: True), ("text", lambda r: group(r) == "text"), ("table-text/table", lambda r: group(r) != "text")):
    idx = [i for i, r in enumerate(A) if sel(r)]
    anyg = mean(bool(set(A[i]["gold_evidence_ids"]) & pools[i]) for i in idx)
    cov = mean(len(set(A[i]["gold_evidence_ids"]) & pools[i]) / len(A[i]["gold_evidence_ids"]) for i in idx)
    print(f"  {name:17s} n={len(idx):3d}  questions with >=1 gold paragraph in pool: {anyg:.3f} | share of all gold paragraphs in pool: {cov:.3f}")
print("\nBM25 recall of gold paragraphs at several depths (raw ranking, table rows counted as non-relevant):")
for k in (5, 10, 30, 100):
    print(f"  recall@{k:<3d} = {mean(recall(ids(r, False), r['gold_evidence_ids'], k) for r in A):.3f}")

def oracle(row):  # gold paragraphs that are in the pool placed first, then other paragraphs
    pool_cond = ids(row, True); g = set(row["gold_evidence_ids"])
    return [x for x in pool_cond if x in g] + [x for x in pool_cond if x not in g]
print("\nCONDENSED metrics from the top-100 files (condensed lists now hold more paragraphs than the earlier top-30 run):")
rows_out = [("BM25", A), ("BM25+Contriever", B)]
for name, R in rows_out:
    print(f"  {name:16s} recall@5 = {mean(recall(ids(r, True), r['gold_evidence_ids'], 5) for r in R):.3f}   ndcg@10 = {mean(ndcg(ids(r, True), r['gold_evidence_ids'], 10) for r in R):.3f}")
print(f"  {'ORACLE reorder':16s} recall@5 = {mean(recall(oracle(r), r['gold_evidence_ids'], 5) for r in A):.3f}   ndcg@10 = {mean(ndcg(oracle(r), r['gold_evidence_ids'], 10) for r in A):.3f}   (ceiling for ANY reranker of this pool)")
print("\nCONSISTENCY with the earlier top-30 run (strict metrics use only the top 5/10 raw ranks, so should be identical):")
for name, fn, R in (("BM25", a.a, A), ("BM25+Contriever", a.b, B)):
    p = os.path.join(a.ref, fn)
    if not os.path.exists(p): print("  SKIP", name, "(no reference file)"); continue
    ref = load(p)
    ok = all(abs(recall(ids(x, False), x["gold_evidence_ids"], 5) - recall(ids(y, False), y["gold_evidence_ids"], 5)) < 1e-12 and
             abs(ndcg(ids(x, False), x["gold_evidence_ids"], 10) - ndcg(ids(y, False), y["gold_evidence_ids"], 10)) < 1e-12 for x, y in zip(R, ref))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: strict recall@5 / ndcg@10 identical per question")

# ------------------------------------------------------------------ Part 2
DA, DB = load(os.path.join(a.doc, a.a)), load(os.path.join(a.doc, a.b))
print("\n\n== PART 2: search restricted to the question's own document (oracle-document) ==")
print("rows:", len(DA), len(DB), "| same question ids:", [r["question_id"] for r in DA] == [r["question_id"] for r in DB])
sizes = [len(r["retrieved"]) for r in DA]; npar = [len(ids(r, True)) for r in DA]
print(f"passages per document: min {min(sizes)}, max {max(sizes)} (store_k was 100, so each list holds the WHOLE document if max < 100)")
print(f"paragraphs per document: min {min(npar)}, median {sorted(npar)[len(npar)//2]}, max {max(npar)}; questions whose document has <=5 paragraphs (recall@5 is then trivially 1): {sum(n <= 5 for n in npar)}/{len(npar)}")
rng = random.Random(0)
def rand_base(row, reps=200):
    p = ids(row, True); r5 = n10 = 0.0
    for _ in range(reps):
        q = p[:]; rng.shuffle(q); r5 += recall(q, row["gold_evidence_ids"], 5); n10 += ndcg(q, row["gold_evidence_ids"], 10)
    return r5 / reps, n10 / reps
rb = [rand_base(r) for r in DA]
docs = [r["doc_id"] for r in DA]
def table(sel, label):
    idx = [i for i, r in enumerate(DA) if sel(r)]
    out = f"  {label:17s} n={len(idx):3d}"
    for name, R in (("BM25", DA), ("BM25+Contr", DB)):
        out += f" | {name}: R@5 {mean(recall(ids(R[i], True), R[i]['gold_evidence_ids'], 5) for i in idx):.3f} nDCG@10 {mean(ndcg(ids(R[i], True), R[i]['gold_evidence_ids'], 10) for i in idx):.3f}"
    out += f" | random order: R@5 {mean(rb[i][0] for i in idx):.3f} nDCG@10 {mean(rb[i][1] for i in idx):.3f}"
    print(out)
print("CONDENSED metrics (paragraphs only):")
table(lambda r: True, "all 200"); table(lambda r: group(r) == "text", "text"); table(lambda r: group(r) != "text", "table-text/table")
print("\nPaired difference Contriever - BM25 (condensed), document-level 95% bootstrap CI:")
for m, f, k in (("recall@5", recall, 5), ("ndcg@10", ndcg, 10)):
    d = [f(ids(y, True), y["gold_evidence_ids"], k) - f(ids(x, True), x["gold_evidence_ids"], k) for x, y in zip(DA, DB)]
    lo, hi = boot(d, docs)
    print(f"  {m}: mean diff {mean(d):+.4f}, CI [{lo:+.3f}, {hi:+.3f}], better={sum(v > 0 for v in d)} worse={sum(v < 0 for v in d)} tie={sum(v == 0 for v in d)}")