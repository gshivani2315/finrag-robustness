"""
verify_outputs.py - independent check of the retrieval outputs.
It re-computes every metric straight from the JSONL files (it does NOT import
src/evaluate.py), so agreement with metrics_summary.json is a real cross-check.
Usage: python scripts/verify_outputs.py --dir outputs/dev200
"""
import argparse, json, math, os, random

def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def recall(ranked, gold, k):
    g = set(gold); return len(g & set(ranked[:k])) / len(g)

def ndcg(ranked, gold, k):
    g = set(gold)
    dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(ranked[:k]) if x in g)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(g), k)))
    return dcg / idcg

def ids(row, condensed):
    return [p["passage_id"] for p in row["retrieved"] if not (condensed and p["source_type"] == "table_row")]

def metric(row, name):
    cond = name.endswith("condensed")
    f = recall if name.startswith("recall") else ndcg
    k = 5 if name.startswith("recall") else 10
    return f(ids(row, cond), row["gold_evidence_ids"], k)

def boot(diffs, docs, n=10000, seed=0, cluster=False):
    rng = random.Random(seed)
    if not cluster:
        m = len(diffs); means = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    else:
        by = {}
        for d, doc in zip(diffs, docs): by.setdefault(doc, []).append(d)
        groups = list(by.values()); g = len(groups); means = []
        for _ in range(n):
            pick = [groups[rng.randrange(g)] for _ in range(g)]
            means.append(sum(sum(x) for x in pick) / sum(len(x) for x in pick))
        means.sort()
    return means[int(.025 * n)], means[int(.975 * n)]

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="outputs/dev200")
ap.add_argument("--a", default="retrieval_bm25.jsonl")
ap.add_argument("--b", default="retrieval_bm25_contriever.jsonl")
a = ap.parse_args()
A, B = load(os.path.join(a.dir, a.a)), load(os.path.join(a.dir, a.b))
summ = json.load(open(os.path.join(a.dir, "metrics_summary.json"), encoding="utf-8"))
def check(label, ok): print(("PASS  " if ok else "FAIL  ") + label)

print("== 1. structure ==")
for name, R in (("A", A), ("B", B)):
    print(f"{name}: retriever names = {sorted({r['retriever'] for r in R})}, rows = {len(R)}, unique question ids = {len({r['question_id'] for r in R})}")
check("same question ids in same order", [r["question_id"] for r in A] == [r["question_id"] for r in B])
check("same question text and gold ids", all(x["question"] == y["question"] and x["gold_evidence_ids"] == y["gold_evidence_ids"] for x, y in zip(A, B)))
n_nogold = sum(not r["gold_evidence_ids"] for r in A)
print(f"rows without gold evidence: {n_nogold} (these are skipped in sections 2 and 3)")
keep = [i for i, r in enumerate(A) if r["gold_evidence_ids"]]
A, B = [A[i] for i in keep], [B[i] for i in keep]
check("ranks are 1..N with no gaps", all([p["rank"] for p in r["retrieved"]] == list(range(1, len(r["retrieved"]) + 1)) for r in A + B))
check("BM25 scores non-increasing", all(all(r["retrieved"][i]["bm25_score"] >= r["retrieved"][i+1]["bm25_score"] for i in range(len(r["retrieved"]) - 1)) for r in A))
check("Contriever scores present and non-increasing", all("contriever_score" in p for r in B for p in r["retrieved"]) and all(all(r["retrieved"][i]["contriever_score"] >= r["retrieved"][i+1]["contriever_score"] for i in range(len(r["retrieved"]) - 1)) for r in B))
check("no duplicate passages within a ranking", all(len({p["passage_id"] for p in r["retrieved"]}) == len(r["retrieved"]) for r in A + B))
print("answer_from counts:", {k: sum(r["answer_from"] == k for r in A) for k in sorted({r["answer_from"] for r in A})})
print("documents covered:", len({r["doc_id"] for r in A}))
print("sample config:", {k: summ["config"].get(k) for k in ("split", "n", "seed", "only_evaluable", "candidate_n", "similarity", "doc_scope")})

print("\n== 2. recomputed metrics vs metrics_summary.json ==")
names = ["recall@5_strict", "ndcg@10_strict", "recall@5_condensed", "ndcg@10_condensed"]
for label, R in (("bm25", A), ("bm25+contriever", B)):
    rows = [r for r in R]
    for m in names:
        mine = sum(metric(r, m) for r in rows) / len(rows)
        if label not in summ["results"]:
            print(f"SKIP  {label}: not in metrics_summary.json"); break
        saved = summ["results"][label]["overall_evaluable"][m]
        check(f"{label:16s} {m:19s} recomputed={mine:.4f} saved={saved:.4f}", abs(mine - saved) < 1e-9)
    stored = [r["retrieval_metrics"] for r in rows]
    check(f"{label:16s} per-row stored metrics equal recomputed", all(abs(s[m] - metric(r, m)) < 1e-9 for r, s in zip(rows, stored) for m in names))

print("\n== 3. paired bootstrap of (Contriever - BM25), 10,000 resamples ==")
print("question-level treats questions as independent; document-level resamples whole documents")
print("(questions from one document share tables/paragraphs, so the document-level interval is the safer one)")
docs = [r["doc_id"] for r in A]
for m in ["recall@5_condensed", "ndcg@10_condensed"]:
    d = [metric(y, m) - metric(x, m) for x, y in zip(A, B)]
    q, c = boot(d, docs), boot(d, docs, cluster=True)
    print(f"{m}: mean diff {sum(d)/len(d):+.4f} | question-level 95% CI [{q[0]:+.3f}, {q[1]:+.3f}] | document-level 95% CI [{c[0]:+.3f}, {c[1]:+.3f}]")