"""
fusion_analysis.py - hybrid BM25 + Contriever scoring, evaluated WITHOUT re-running any model.
The Contriever run files already hold, for the same 100 candidates, both bm25_score and contriever_score.

PROTOCOL (fixed in advance, so the result is not tuned on the evaluation sample):
  1. Choose the weight w on the HELD-OUT sample (outputs/tune, different documents).   criterion: best condensed nDCG@10
  2. Apply that single w (and parameter-free RRF, k=60) ONCE to the 200 evaluation questions.
Fusion: per question, min-max rescale each score over the 100 candidates, then  fused = w*BM25 + (1-w)*Contriever.
        w=1 is exactly BM25's order, w=0 is exactly Contriever's order (both are checked below).
RRF:    fused = 1/(60+bm25_rank) + 1/(60+contriever_rank)   (no weight to tune)
Run from the project folder:  python scripts\\fusion_analysis.py
"""
import argparse, csv, json, math, os, random

def load(p):
    with open(p, encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]
def ids(ranked, cond): return [p["passage_id"] for p in ranked if not (cond and p["source_type"] == "table_row")]
def recall(r, g, k): g = set(g); return len(g & set(r[:k])) / len(g)
def ndcg(r, g, k):
    g = set(g); dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(r[:k]) if x in g)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(len(g), k)))
def mean(x): x = list(x); return sum(x) / len(x) if x else float("nan")
def metrics(ranked, gold):
    return {"recall@5_condensed": recall(ids(ranked, True), gold, 5), "ndcg@10_condensed": ndcg(ids(ranked, True), gold, 10),
            "recall@5_strict": recall(ids(ranked, False), gold, 5), "ndcg@10_strict": ndcg(ids(ranked, False), gold, 10)}
def boot(d, docs, n=10000, seed=0):
    by = {}
    for v, doc in zip(d, docs): by.setdefault(doc, []).append(v)
    gs, rng, out = list(by.values()), random.Random(seed), []
    for _ in range(n):
        pick = [gs[rng.randrange(len(gs))] for _ in gs]
        out.append(sum(sum(x) for x in pick) / sum(len(x) for x in pick))
    out.sort(); return out[int(.025 * n)], out[int(.975 * n)]

def minmax(x):
    lo, hi = min(x), max(x)
    return [0.5] * len(x) if hi == lo else [(v - lo) / (hi - lo) for v in x]
def fuse(cands, w):
    b = minmax([c["bm25_score"] for c in cands]); s = minmax([c["contriever_score"] for c in cands])
    order = sorted(range(len(cands)), key=lambda j: (-(w * b[j] + (1 - w) * s[j]), cands[j]["bm25_rank"]))
    return [dict(cands[j], fused_score=w * b[j] + (1 - w) * s[j]) for j in order]
def rrf(cands, k=60):
    sc = [1 / (k + c["bm25_rank"]) + 1 / (k + c["rank"]) for c in cands]   # c["rank"] = Contriever rank in the file
    order = sorted(range(len(cands)), key=lambda j: (-sc[j], cands[j]["bm25_rank"]))
    return [dict(cands[j], fused_score=sc[j]) for j in order]

ap = argparse.ArgumentParser()
ap.add_argument("--tune-dir", default="outputs/tune"); ap.add_argument("--eval-dir", default="outputs/dev200_k100")
ap.add_argument("--out", default="outputs/exp3_fusion")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
BM, CT = "retrieval_bm25.jsonl", "retrieval_bm25_contriever.jsonl"

def get(d):
    bm, ct = load(os.path.join(d, BM)), load(os.path.join(d, CT))
    assert [r["question_id"] for r in bm] == [r["question_id"] for r in ct], "question order differs in " + d
    return bm, ct

tbm, tct = get(a.tune_dir); ebm, ect = get(a.eval_dir)
overlap = {r["doc_id"] for r in tbm} & {r["doc_id"] for r in ebm}
print(f"tune questions={len(tbm)}  eval questions={len(ebm)}  document overlap between them: {len(overlap)}  (must be 0)")
assert not overlap, "tune and eval share documents - protocol broken"

def mean_metrics(rows_ranked, gold_rows, key): return mean(metrics(rk, g["gold_evidence_ids"])[key] for rk, g in zip(rows_ranked, gold_rows))
print("\n== sanity: w=1 must equal BM25, w=0 must equal Contriever (condensed nDCG@10) ==")
for name, bm, ct in (("tune", tbm, tct), ("eval", ebm, ect)):
    for w, ref in ((1.0, bm), (0.0, ct)):
        mine = mean_metrics([fuse(r["retrieved"], w) for r in ct], ct, "ndcg@10_condensed")
        theirs = mean_metrics([r["retrieved"] for r in ref], ref, "ndcg@10_condensed")
        print(f"{'PASS' if abs(mine - theirs) < 1e-9 else 'FAIL'}  {name} w={w}: fused {mine:.4f} vs file {theirs:.4f}")

print("\n== TUNE sample: sweep of w (condensed) ==")
grid = []
for i in range(11):
    w = round(i / 10, 1); rk = [fuse(r["retrieved"], w) for r in tct]
    grid.append((w, mean_metrics(rk, tct, "recall@5_condensed"), mean_metrics(rk, tct, "ndcg@10_condensed")))
rk = [rrf(r["retrieved"]) for r in tct]
rrf_t = (mean_metrics(rk, tct, "recall@5_condensed"), mean_metrics(rk, tct, "ndcg@10_condensed"))
print(f"{'w':>5s} {'Recall@5':>9s} {'nDCG@10':>8s}")
for w, r5, n10 in grid:
    print(f"{w:5.1f} {r5:9.4f} {n10:8.4f}" + ("   <- BM25 order" if w == 1.0 else "   <- Contriever order" if w == 0.0 else ""))
print(f"{'RRF':>5s} {rrf_t[0]:9.4f} {rrf_t[1]:8.4f}   (k=60, nothing tuned)")
best = max(grid, key=lambda g: (round(g[2], 12), g[0]))
wstar = best[0]; print(f"\nchosen w* = {wstar} (best tune nDCG@10 = {best[2]:.4f}; ties go to the larger w)")
d = [metrics(fuse(c["retrieved"], wstar), c["gold_evidence_ids"])["ndcg@10_condensed"] - metrics(b["retrieved"], b["gold_evidence_ids"])["ndcg@10_condensed"] for b, c in zip(tbm, tct)]
lo, hi = boot(d, [r["doc_id"] for r in tbm])
print(f"on the TUNE sample, fusion(w*) - BM25 nDCG@10: {mean(d):+.4f}  CI [{lo:+.3f}, {hi:+.3f}]  (w*=1.0 would mean fusion found nothing better than BM25)")

print(f"\n== EVAL sample (200 questions), evaluated once with w*={wstar} and RRF ==")
variants = {"BM25": [r["retrieved"] for r in ebm], "baseline Contriever": [r["retrieved"] for r in ect],
            f"fusion w={wstar}": [fuse(r["retrieved"], wstar) for r in ect], "RRF k=60": [rrf(r["retrieved"]) for r in ect]}
grp = lambda r: "text" if r["answer_from"] == "text" else "text+table"
keys = ["recall@5_condensed", "ndcg@10_condensed", "recall@5_strict", "ndcg@10_strict"]
print(f"{'':22s}{'group':11s}{'n':>4s}" + "".join(f"{k:>20s}" for k in keys))
for g, sel in (("all", lambda r: True), ("text", lambda r: grp(r) == "text"), ("text+table", lambda r: grp(r) == "text+table")):
    idx = [i for i, r in enumerate(ect) if sel(r)]
    for name, rk in variants.items():
        print(f"{name:22s}{g:11s}{len(idx):4d}" + "".join(f"{mean(metrics(rk[i], ect[i]['gold_evidence_ids'])[k] for i in idx):20.4f}" for k in keys))
docs = [r["doc_id"] for r in ect]
print("\npaired differences (document-level 95% bootstrap CI)")
for cand in (f"fusion w={wstar}", "RRF k=60"):
    for other in ("baseline Contriever", "BM25"):
        for k in keys[:2]:
            dd = [metrics(variants[cand][i], ect[i]["gold_evidence_ids"])[k] - metrics(variants[other][i], ect[i]["gold_evidence_ids"])[k] for i in range(len(ect))]
            lo, hi = boot(dd, docs); v = "CI above 0" if lo > 0 else ("CI below 0" if hi < 0 else "CI includes 0")
            print(f"{cand} - {other:20s} {k:19s} {mean(dd):+.4f}  CI [{lo:+.3f}, {hi:+.3f}]  better={sum(x > 1e-12 for x in dd)} worse={sum(x < -1e-12 for x in dd)} tie={sum(abs(x) <= 1e-12 for x in dd)}  -> {v}")

# save eval rankings in the pipeline's JSONL layout so compare_runs.py can re-verify them independently
for tag, rk, name in (("fusion", variants[f"fusion w={wstar}"], f"bm25+contriever_fusion_w{wstar}"), ("rrf", variants["RRF k=60"], "bm25+contriever_rrf60")):
    rows = []
    for r, ranked in zip(ect, rk):
        ranked = [dict(p, rank=i + 1) for i, p in enumerate(ranked)]   # note: 'contriever_rank' info is dropped; rank = fused rank
        m = metrics(ranked, r["gold_evidence_ids"])
        rows.append(dict(r, retriever=name, retrieved=ranked, retrieval_metrics=m))
    with open(os.path.join(a.out, f"retrieval_{tag}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(os.path.join(a.out, "tune_grid.csv"), "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["w", "recall@5_condensed", "ndcg@10_condensed"]); wr.writerows(grid)
print("\nwrote", sorted(os.listdir(a.out)), "to", a.out)