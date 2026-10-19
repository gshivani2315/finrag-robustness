"""
final_confirmation.py - one pre-declared comparison on FRESH questions.
Fresh = evaluable dev questions that were in neither the 315 held-out questions (used to choose the fusion weight and the
slot split) nor your 200 evaluation questions. Nothing below is tuned on them.

Systems compared (all from saved run files, no models):
  A  original BM25
  B  original BM25 + Contriever (dot, candidate pool 100)          <- the baseline we set out to improve
  C  table-aware BM25 (natural ranking)
  D  table-aware BM25, 4 paragraph slots + 1 table-row slot in the top 5   <- the system we propose
Run from the project folder:   python scripts\\final_confirmation.py
"""
import argparse, json, math, random

def load(p):
    with open(p, encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]
def pid(ranked, cond=False): return [p["passage_id"] for p in ranked if not (cond and p["source_type"] == "table_row")]
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
def answer_at(ranked, row, k): t = " || ".join(norm(p["text"]) for p in ranked[:k]); return all(x in t for x in items(row))
def with_slots(retrieved, n_par=4, k=5):
    paras = [x for x in retrieved if x["source_type"] == "paragraph"][:n_par]
    rows = [x for x in retrieved if x["source_type"] == "table_row"][:k - n_par]
    pos = {x["passage_id"]: i for i, x in enumerate(retrieved)}
    top = sorted(paras + rows, key=lambda x: pos[x["passage_id"]]); used = {x["passage_id"] for x in top}
    return top + [x for x in retrieved if x["passage_id"] not in used]
grp = lambda r: "text" if r["answer_from"] == "text" else "text+table"

ap = argparse.ArgumentParser()
ap.add_argument("--orig-bm25", default="outputs/final_orig/retrieval_bm25.jsonl")
ap.add_argument("--orig-ctr", default="outputs/final_orig/retrieval_bm25_contriever.jsonl")
ap.add_argument("--ctx-bm25", default="outputs/final_ctx/retrieval_bm25.jsonl")
ap.add_argument("--used-eval", default="outputs/dev200_k100/retrieval_bm25.jsonl")
ap.add_argument("--used-tune", default="outputs/tune/retrieval_bm25.jsonl")
a = ap.parse_args()
A, B, C = load(a.orig_bm25), load(a.orig_ctr), load(a.ctx_bm25)
for X in (B, C): assert [r["question_id"] for r in X] == [r["question_id"] for r in A], "question order differs"
eval_ids = {r["question_id"] for r in load(a.used_eval)}; tune = load(a.used_tune)
print(f"run files: {len(A)} questions | tune documents present: {len({r['doc_id'] for r in A} & {r['doc_id'] for r in tune})} (must be 0)")
keep = [i for i, r in enumerate(A) if r["question_id"] not in eval_ids]
A, B, C = ([X[i] for i in keep] for X in (A, B, C))
assert all(r["gold_evidence_ids"] for r in A)
print(f"fresh questions (never used for evaluation or tuning): {len(A)} from {len({r['doc_id'] for r in A})} documents; "
      f"text-only {sum(grp(r) == 'text' for r in A)}, text+table {sum(grp(r) == 'text+table' for r in A)}")
S = {"A original BM25": [r["retrieved"] for r in A], "B original BM25+Contriever": [r["retrieved"] for r in B],
     "C table-aware BM25": [r["retrieved"] for r in C], "D table-aware, 4 par + 1 row": [with_slots(r["retrieved"]) for r in C]}
G = [r["gold_evidence_ids"] for r in A]; el = [i for i, r in enumerate(A) if eligible(r)]
M = {"cond R@5": lambda rk, g: recall(pid(rk, True), g, 5), "cond nDCG@10": lambda rk, g: ndcg(pid(rk, True), g, 10),
     "strict R@5": lambda rk, g: recall(pid(rk), g, 5), "strict nDCG@10": lambda rk, g: ndcg(pid(rk), g, 10)}
print(f"\n{'system':30s}{'group':11s}{'n':>4s}" + "".join(f"{m:>15s}" for m in M) + f"{'rows top5':>11s}{'answer@5':>10s}")
for gname, sel in (("all", lambda r: True), ("text", lambda r: grp(r) == "text"), ("text+table", lambda r: grp(r) == "text+table")):
    idx = [i for i, r in enumerate(A) if sel(r)]; e2 = [i for i in el if i in set(idx)]
    for name, rk in S.items():
        print(f"{name:30s}{gname:11s}{len(idx):4d}" + "".join(f"{mean(f(rk[i], G[i]) for i in idx):15.4f}" for f in M.values())
              + f"{mean(sum(p['source_type'] == 'table_row' for p in rk[i][:5]) for i in idx):11.2f}{mean(answer_at(rk[i], A[i], 5) for i in e2):10.3f}")
print(f"(answer@5 is the rough answer-string proxy, {len(el)} eligible span/multi-span questions overall)")
docs = [r["doc_id"] for r in A]
print("\npaired differences, document-level 95% bootstrap CI")
for cand in ("D table-aware, 4 par + 1 row", "C table-aware BM25"):
    for other in ("B original BM25+Contriever", "A original BM25"):
        for m, f in M.items():
            d = [f(S[cand][i], G[i]) - f(S[other][i], G[i]) for i in range(len(A))]
            lo, hi = boot(d, docs); v = "CI above 0" if lo > 0 else ("CI below 0" if hi < 0 else "CI includes 0")
            print(f"{cand:30s} - {other:28s} {m:15s} {mean(d):+.4f}  CI [{lo:+.3f}, {hi:+.3f}]  better={sum(x > 1e-12 for x in d)} worse={sum(x < -1e-12 for x in d)}  -> {v}")
for other in ("B original BM25+Contriever", "A original BM25"):
    for k in (5, 10):
        g_ = sum(answer_at(S["D table-aware, 4 par + 1 row"][i], A[i], k) and not answer_at(S[other][i], A[i], k) for i in el)
        l_ = sum(answer_at(S[other][i], A[i], k) and not answer_at(S["D table-aware, 4 par + 1 row"][i], A[i], k) for i in el)
        print(f"answer proxy top {k:2d}: D vs {other:28s} {mean(answer_at(S['D table-aware, 4 par + 1 row'][i], A[i], k) for i in el):.3f} vs {mean(answer_at(S[other][i], A[i], k) for i in el):.3f}  (D gained {g_}, lost {l_} questions)")