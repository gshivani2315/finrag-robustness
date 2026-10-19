"""
slot_analysis.py - can we keep the table_context gain AND keep table rows in the evidence set?
Offline: re-arranges the SAVED table_context rankings (no model, no new retrieval).
Idea: of the 5 evidence slots, reserve p for paragraphs and 5-p for table rows (each group keeps its own
ranking order); everything after the first 5 stays in the original order.

Metrics shown (rows are unlabelled, so no single metric sees both sides):
  strict Recall@5 / nDCG@10 : rows count as NOT relevant, so it understates when rows are useful (lower bound)
  rows in top 5             : average number of table rows in the first 5 slots
  answer@5 (ROUGH PROXY)    : share of span/multi-span questions (answer >= 4 chars) whose answer string appears in the
                              top-5 passages (rows or paragraphs). Gold answer is used only for evaluation.
Condensed (paragraph-only) metrics are identical for every split by construction, so they are not repeated.

PRE-DECLARED SELECTION RULE (uses the HELD-OUT sample only): among the splits whose answer@5 is within 0.02 of
BM25's answer@5, choose the one with the highest strict Recall@5. If none qualifies, report that none does.
Run from the project folder:   python scripts\\slot_analysis.py
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
def answer_at(ranked, row, k):
    text = " || ".join(norm(p["text"]) for p in ranked[:k]); return all(x in text for x in items(row))

def with_slots(retrieved, n_par, k=5):
    paras = [x for x in retrieved if x["source_type"] == "paragraph"][:n_par]
    rows = [x for x in retrieved if x["source_type"] == "table_row"][:k - n_par]
    pos = {x["passage_id"]: i for i, x in enumerate(retrieved)}
    top = sorted(paras + rows, key=lambda x: pos[x["passage_id"]])
    used = {x["passage_id"] for x in top}
    return top + [x for x in retrieved if x["passage_id"] not in used]

ap = argparse.ArgumentParser()
ap.add_argument("--eval-base", default="outputs/dev200_k100/retrieval_bm25.jsonl")
ap.add_argument("--eval-ctx", default="outputs/exp4_ctx_eval/retrieval_bm25.jsonl")
ap.add_argument("--tune-base", default="outputs/tune/retrieval_bm25.jsonl")
ap.add_argument("--tune-ctx", default="outputs/exp4_ctx_tune/retrieval_bm25.jsonl")
a = ap.parse_args()

def variants(B, C):
    v = {"BM25 (natural mix)": [r["retrieved"] for r in B], "table_context (natural mix)": [r["retrieved"] for r in C]}
    for p in (4, 3, 2): v[f"table_context, {p} paragraphs + {5 - p} rows"] = [with_slots(r["retrieved"], p) for r in C]
    return v

def evaluate(B, C, v):
    el = [i for i, r in enumerate(B) if eligible(r)]; out = {}
    for name, rk in v.items():
        out[name] = {"sr5": [recall(pid(rk[i]), C[i]["gold_evidence_ids"], 5) for i in range(len(C))],
                     "sn10": [ndcg(pid(rk[i]), C[i]["gold_evidence_ids"], 10) for i in range(len(C))],
                     "cr5": mean(recall(pid(rk[i], True), C[i]["gold_evidence_ids"], 5) for i in range(len(C))),
                     "rows5": mean(sum(p["source_type"] == "table_row" for p in rk[i][:5]) for i in range(len(C))),
                     "ans5": mean(answer_at(rk[i], B[i], 5) for i in el), "n_el": len(el)}
    return out

data = {}
for name, bp, cp in (("HELD-OUT 315 (used for the selection)", a.tune_base, a.tune_ctx), ("YOUR 200 (check only)", a.eval_base, a.eval_ctx)):
    B, C = load(bp), load(cp)
    assert [r["question_id"] for r in B] == [r["question_id"] for r in C]
    v = variants(B, C); res = evaluate(B, C, v); data[name] = (B, C, res)
    print(f"\n================ {name}: {len(B)} questions, {res['BM25 (natural mix)']['n_el']} eligible for the answer proxy ================")
    print(f"{'variant':44s}{'strict R@5':>11s}{'strict nDCG@10':>16s}{'rows in top5':>14s}{'answer@5':>10s}")
    for k, m in res.items():
        print(f"{k:44s}{mean(m['sr5']):11.3f}{mean(m['sn10']):16.3f}{m['rows5']:14.2f}{m['ans5']:10.3f}")
    print(f"condensed Recall@5 (same for all table_context variants): {res['table_context (natural mix)']['cr5']:.3f}")

held = list(data)[0]; Bh, Ch, rh = data[held]
base_ans = rh["BM25 (natural mix)"]["ans5"]
ok = {k: m for k, m in rh.items() if k.startswith("table_context,") and m["ans5"] >= base_ans - 0.02}
print(f"\n== selection rule on the held-out sample: answer@5 must be >= {base_ans - 0.02:.3f} (BM25's {base_ans:.3f} minus 0.02) ==")
if not ok:
    print("RESULT: no slot split keeps the answer proxy within 0.02 of BM25 - none selected.")
else:
    best = max(ok, key=lambda k: mean(ok[k]["sr5"])); print("qualifying splits:", ", ".join(ok)); print("SELECTED:", best)
    for name, (B, C, res) in data.items():
        d = [x - y for x, y in zip(res[best]["sr5"], res["BM25 (natural mix)"]["sr5"])]
        lo, hi = boot(d, [r["doc_id"] for r in B])
        print(f"{name}: strict Recall@5 {best} - BM25 = {mean(d):+.4f}  document-level CI [{lo:+.3f}, {hi:+.3f}]  better={sum(x > 1e-12 for x in d)} worse={sum(x < -1e-12 for x in d)}")