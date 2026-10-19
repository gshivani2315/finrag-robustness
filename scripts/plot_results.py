"""
plot_results.py - tables, CSVs and figures from the saved retrieval outputs.
Reads ONLY the JSONL files in outputs/ (nothing is typed in by hand). If a run folder is
missing, that setting is printed as UNAVAILABLE and skipped.

    python src\\plot_results.py                 (from the project folder)

Settings (same 200 questions in both):
  global   : outputs/dev200_k100      search the whole dev corpus, 100 passages stored per question
  docscope : outputs/dev200_docscope  search only the question's own document (oracle-document)
Metrics use the CONDENSED view (table rows removed; they are unlabelled). Strict values are in the CSV.
Error bars = 95% bootstrap CI that resamples whole DOCUMENTS (questions of one document are correlated).
Group "text+table" = answer_from 'table-text' (117) plus 2 'table' questions that carry a paragraph label.
"""
import argparse, csv, json, math, os, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = [("bm25", "BM25", "retrieval_bm25.jsonl"),
           ("bm25+contriever", "BM25 + Contriever", "retrieval_bm25_contriever.jsonl")]
SETTINGS = [("global", "Whole dev corpus", "outputs/dev200_k100"),
            ("docscope", "Question's own document only (oracle-document)", "outputs/dev200_docscope")]
GROUPS = [("all", "All"), ("text", "Text-only"), ("table-text", "Text+table")]
COLOR = {"bm25": "#4C72B0", "bm25+contriever": "#DD8452"}
DEPTHS = [1, 3, 5, 10, 20, 30, 50, 100]

def load(p):
    with open(p, encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]
def ids(row, cond): return [p["passage_id"] for p in row["retrieved"] if not (cond and p["source_type"] == "table_row")]
def recall(r, g, k): g = set(g); return len(g & set(r[:k])) / len(g)
def ndcg(r, g, k):
    g = set(g); dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(r[:k]) if x in g)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(len(g), k)))
def grp(row): return "text" if row["answer_from"] == "text" else "table-text"
def mean(x): x = list(x); return sum(x) / len(x) if x else float("nan")
def boot_ci(vals, docs, n=5000, seed=0):
    by = {}
    for v, d in zip(vals, docs): by.setdefault(d, []).append(v)
    gs, rng, out = list(by.values()), random.Random(seed), []
    for _ in range(n):
        pick = [gs[rng.randrange(len(gs))] for _ in gs]
        out.append(sum(sum(x) for x in pick) / sum(len(x) for x in pick))
    out.sort(); return out[int(.025 * n)], out[int(.975 * n)]

def per_q(row):
    g = row["gold_evidence_ids"]
    return {"recall@5": recall(ids(row, True), g, 5), "ndcg@10": ndcg(ids(row, True), g, 10),
            "recall@5_strict": recall(ids(row, False), g, 5), "ndcg@10_strict": ndcg(ids(row, False), g, 10)}

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/figures"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})

data, res, table_rows = {}, {}, []
for sk, slabel, folder in SETTINGS:
    paths = [os.path.join(folder, f) for _, _, f in METHODS]
    if not all(os.path.exists(p) for p in paths):
        print(f"UNAVAILABLE: setting '{sk}' ({folder} missing or incomplete) - skipped"); continue
    rows = {m: load(p) for (m, _, _), p in zip(METHODS, paths)}
    assert [r["question_id"] for r in rows["bm25"]] == [r["question_id"] for r in rows["bm25+contriever"]], "question order differs"
    data[sk] = rows
    for m, _, _ in METHODS:
        for gk, _ in GROUPS:
            sel = [r for r in rows[m] if gk == "all" or grp(r) == gk]
            pq = [per_q(r) for r in sel]; docs = [r["doc_id"] for r in sel]
            rec = {"setting": sk, "method": m, "group": gk, "n": len(sel)}
            for met in ("recall@5", "ndcg@10", "recall@5_strict", "ndcg@10_strict"):
                v = [q[met] for q in pq]; lo, hi = boot_ci(v, docs)
                rec[met], rec[met + "_ci_low"], rec[met + "_ci_high"] = mean(v), lo, hi
            res[(sk, m, gk)] = rec; table_rows.append(rec)
if not data: raise SystemExit("No run folders found - run the pipeline first.")

# ---------------- CSV + printed table
cols = list(table_rows[0].keys())
with open(os.path.join(a.out, "results_table.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(table_rows)
print(f"\n{'setting':9s} {'group':11s} {'n':>4s} {'method':17s} {'Recall@5':>9s} {'nDCG@10':>8s}   (condensed; 95% CI by document)")
for r in table_rows:
    print(f"{r['setting']:9s} {r['group']:11s} {r['n']:4d} {r['method']:17s} {r['recall@5']:9.3f} {r['ndcg@10']:8.3f}   "
          f"R@5 [{r['recall@5_ci_low']:.3f}, {r['recall@5_ci_high']:.3f}]  nDCG [{r['ndcg@10_ci_low']:.3f}, {r['ndcg@10_ci_high']:.3f}]")

def bars(ax, sk, metric, groups):
    w = 0.38
    for j, (m, mlabel, _) in enumerate(METHODS):
        xs = [i + (j - 0.5) * w for i in range(len(groups))]
        vals = [res[(sk, m, g)][metric] for g, _ in groups]
        lo = [res[(sk, m, g)][metric] - res[(sk, m, g)][metric + "_ci_low"] for g, _ in groups]
        hi = [res[(sk, m, g)][metric + "_ci_high"] - res[(sk, m, g)][metric] for g, _ in groups]
        ax.bar(xs, vals, w, yerr=[lo, hi], capsize=3, color=COLOR[m], label=mlabel)
        for x, v in zip(xs, vals): ax.text(x, v / 2, f"{v:.3f}", ha="center", color="white", fontsize=9)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{lab}\n(n={res[(sk, 'bm25', g)]['n']})" for g, lab in groups])
    ax.set_ylim(0, 1.18); ax.set_yticks([0, .2, .4, .6, .8, 1.0]); ax.set_ylabel(metric.replace("recall", "Recall").replace("ndcg", "nDCG")); ax.grid(axis="y", alpha=.3)

def panel_figure(groups, fname, title):
    sks = [s for s, _, _ in SETTINGS if s in data]
    fig, axes = plt.subplots(len(sks), 2, figsize=(10, 3.8 * len(sks)), squeeze=False)
    for i, sk in enumerate(sks):
        for j, met in enumerate(("recall@5", "ndcg@10")):
            bars(axes[i][j], sk, met, groups)
            axes[i][j].set_title(f"{dict((s, l) for s, l, _ in SETTINGS)[sk]}", fontsize=9)
    axes[0][0].legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    fig.suptitle(title, fontsize=11); fig.text(0.5, 0.005, "Error bars: 95% bootstrap CI (resampling documents). Condensed view: unlabelled table rows removed.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.out, f"{fname}.{ext}"), dpi=300)
    plt.close(fig)

panel_figure(GROUPS[:1], "fig1_overall_comparison", "BM25 vs BM25 + Contriever (200 dev questions with paragraph gold labels)")
panel_figure(GROUPS[1:], "fig2_by_question_type", "Breakdown by TAT-QA answer source")

# ---------------- Fig 3: paired per-question differences (nDCG@10)
sks = [s for s, _, _ in SETTINGS if s in data]
fig, axes = plt.subplots(1, len(sks), figsize=(5.2 * len(sks), 3.8), squeeze=False)
diff_rows = []
for ax, sk in zip(axes[0], sks):
    A, B = data[sk]["bm25"], data[sk]["bm25+contriever"]
    d = [per_q(y)["ndcg@10"] - per_q(x)["ndcg@10"] for x, y in zip(A, B)]
    for x, y, v in zip(A, B, d):
        diff_rows.append({"setting": sk, "question_id": x["question_id"], "group": grp(x), "ndcg@10_diff_contriever_minus_bm25": v,
                          "recall@5_diff_contriever_minus_bm25": per_q(y)["recall@5"] - per_q(x)["recall@5"]})
    nz = [v for v in d if abs(v) > 1e-12]
    ax.hist(nz, bins=[-1 + 0.1 * i for i in range(21)], color="#777777", edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_title(f"{dict((s, l) for s, l, _ in SETTINGS)[sk]}\nbetter {sum(v > 1e-12 for v in d)} | worse {sum(v < -1e-12 for v in d)} | unchanged {len(d) - len(nz)}", fontsize=9)
    ax.set_xlabel("nDCG@10 change (Contriever - BM25)"); ax.set_ylabel("questions")
fig.text(0.5, 0.005, "One question = one count. Unchanged questions are counted in the panel titles but not drawn. Condensed view.", ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("png", "pdf"): fig.savefig(os.path.join(a.out, f"fig3_per_question_changes.{ext}"), dpi=300)
plt.close(fig)
with open(os.path.join(a.out, "paired_differences.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, list(diff_rows[0].keys())); w.writeheader(); w.writerows(diff_rows)

# ---------------- Fig 4: recall vs depth (global setting only)
if "global" in data:
    depth_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (gk, glab) in zip(axes, GROUPS):
        for m, mlabel, _ in METHODS:
            sel = [r for r in data["global"][m] if gk == "all" or grp(r) == gk]
            ys = [mean(recall(ids(r, False), r["gold_evidence_ids"], k) for r in sel) for k in DEPTHS]
            ax.plot(DEPTHS, ys, marker="o", color=COLOR[m], label=mlabel, ls="-" if m == "bm25" else "--")
            depth_rows += [{"group": gk, "method": m, "k": k, "recall_at_k_strict": y} for k, y in zip(DEPTHS, ys)]
        ax.set_xscale("log"); ax.set_xticks(DEPTHS); ax.set_xticklabels(DEPTHS, fontsize=8)
        ax.set_title(f"{glab} (n={len(sel)})", fontsize=9); ax.set_xlabel("depth k"); ax.grid(alpha=.3)
    axes[0].set_ylabel("Recall@k (table rows count as non-relevant)"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Gold-paragraph recall vs depth. At k=100 both methods coincide: Contriever only reorders BM25's top 100.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"): fig.savefig(os.path.join(a.out, f"fig4_recall_vs_depth.{ext}"), dpi=300)
    plt.close(fig)
    with open(os.path.join(a.out, "depth_curve.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(depth_rows[0].keys())); w.writeheader(); w.writerows(depth_rows)

print("\nwrote to", a.out, ":", ", ".join(sorted(os.listdir(a.out))))
print("NOT AVAILABLE yet (needs Track B perturbation runs): degradation curves and heatmaps for noise / redundancy / conflict.")