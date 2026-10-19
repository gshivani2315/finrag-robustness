"""
compare_runs.py - compare ONE candidate retrieval run against the two baselines, from the JSONL files only.
    python scripts\\compare_runs.py --cand outputs\\exp1_cosine\\retrieval_bm25_contriever.jsonl --name cosine

Defaults: BM25 = outputs/dev200_k100/retrieval_bm25.jsonl ; baseline Contriever = outputs/dev200_k100/retrieval_bm25_contriever.jsonl
Metrics are recomputed here (not read from metrics_summary.json). Condensed = table rows removed (unlabelled).
CIs: paired bootstrap that resamples whole DOCUMENTS (10,000 resamples).
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
def metric(row, name):
    cond = name.endswith("condensed"); k = 5 if name.startswith("recall") else 10
    return (recall if name.startswith("recall") else ndcg)(ids(row, cond), row["gold_evidence_ids"], k)
def boot(d, docs, n=10000, seed=0):
    by = {}
    for v, doc in zip(d, docs): by.setdefault(doc, []).append(v)
    gs, rng, out = list(by.values()), random.Random(seed), []
    for _ in range(n):
        pick = [gs[rng.randrange(len(gs))] for _ in gs]
        out.append(sum(sum(x) for x in pick) / sum(len(x) for x in pick))
    out.sort(); return out[int(.025 * n)], out[int(.975 * n)]

ap = argparse.ArgumentParser()
ap.add_argument("--cand", required=True); ap.add_argument("--name", default="candidate")
ap.add_argument("--bm25", default="outputs/dev200_k100/retrieval_bm25.jsonl")
ap.add_argument("--base", default="outputs/dev200_k100/retrieval_bm25_contriever.jsonl")
ap.add_argument("--vs-bm25-only", action="store_true", help="candidate is a BM25 variant: compare against BM25 only")
a = ap.parse_args()
R = {"BM25": load(a.bm25)}
if not a.vs_bm25_only: R["baseline Contriever"] = load(a.base)
R[a.name] = load(a.cand)
ref = R["BM25"]
print("== sanity ==")
for k, rows in R.items():
    print(f"{k:22s} rows={len(rows)} retriever={sorted({r['retriever'] for r in rows})} passages/question min={min(len(r['retrieved']) for r in rows)} max={max(len(r['retrieved']) for r in rows)}")
for k, rows in R.items():
    ok = [r["question_id"] for r in rows] == [r["question_id"] for r in ref] and all(r["gold_evidence_ids"] == x["gold_evidence_ids"] for r, x in zip(rows, ref))
    print(f"{'PASS' if ok else 'FAIL'}  {k}: same questions, order and gold ids as BM25 file")
if "baseline Contriever" in R:
    same_pool = all(set(ids(x, False)) == set(ids(y, False)) for x, y in zip(R["baseline Contriever"], R[a.name]))
    print(f"candidate pool identical to baseline pool: {same_pool}  (False is expected only if you changed the pool size or the text)")
docs = [r["doc_id"] for r in ref]
grp = lambda r: "text" if r["answer_from"] == "text" else "text+table"
names = ["recall@5_condensed", "ndcg@10_condensed", "recall@5_strict", "ndcg@10_strict"]
print("\n== metrics (recomputed from the files) ==")
print(f"{'':22s}{'group':11s}{'n':>4s}" + "".join(f"{m:>20s}" for m in names))
for gname, sel in (("all", lambda r: True), ("text", lambda r: grp(r) == "text"), ("text+table", lambda r: grp(r) == "text+table")):
    for k, rows in R.items():
        s = [r for r in rows if sel(r)]
        print(f"{k:22s}{gname:11s}{len(s):4d}" + "".join(f"{mean(metric(r, m) for r in s):20.4f}" for m in names))
print("\n== paired differences, document-level 95% bootstrap CI ==")
for other in [k for k in ("baseline Contriever", "BM25") if k in R]:
    for m in names[:2]:
        d = [metric(c, m) - metric(o, m) for c, o in zip(R[a.name], R[other])]
        lo, hi = boot(d, docs)
        verdict = "CI above 0" if lo > 0 else ("CI below 0" if hi < 0 else "CI includes 0")
        print(f"{a.name} - {other:20s} {m:19s} mean diff {mean(d):+.4f}  CI [{lo:+.3f}, {hi:+.3f}]  better={sum(v > 1e-12 for v in d)} worse={sum(v < -1e-12 for v in d)} tie={sum(abs(v) <= 1e-12 for v in d)}  -> {verdict}")