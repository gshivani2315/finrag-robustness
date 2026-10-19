"""
failure_analysis.py - which questions did BOTH retrievers miss, and what do the misses look like?
Reads outputs/<run>/retrieval_*.jsonl (no model, no network). Run from the project folder:
    python scripts/failure_analysis.py --dir outputs/dev200

"Condensed" = table-row passages removed from the stored ranking (same primary view as the metrics).
The stored ranking holds at most 30 passages, so "miss in top 30" means "not in the stored list".
The FLAGS printed at the end are simple rule-based observations about each missed question.
They describe the question/evidence; they are NOT proven causes of the failure.
"""
import argparse, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_split
from src.passage_builder import build_corpus

STOP = set("what was were is are the a an of in to for and or by from at on as with that this which how much many did does do has have had than between".split())
def toks(t): return [x.replace(",", "") for x in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?|[a-z]+", t.lower())]
def load(p):
    with open(p, encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]

def gold_ranks(row):
    """(rank among paragraphs only, rank in raw list) of the best gold passage; None if absent."""
    gold, para, raw = set(row["gold_evidence_ids"]), 0, 0
    cond = rawr = None
    for p in row["retrieved"]:
        raw += 1
        if p["source_type"] != "table_row": para += 1
        if p["passage_id"] in gold:
            return para, raw
    return None, None

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="outputs/dev200")
ap.add_argument("--a", default="retrieval_bm25.jsonl")
ap.add_argument("--b", default="retrieval_bm25_contriever.jsonl")
ap.add_argument("--n-examples", type=int, default=10)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--split", default="dev")
a = ap.parse_args()

A, B = load(os.path.join(a.dir, a.a)), load(os.path.join(a.dir, a.b))
pairs = [(x, y) for x, y in zip(A, B) if x["gold_evidence_ids"]]
docs, _ = load_split(a.split)
text = {p["passage_id"]: p["text"] for p in build_corpus(docs, a.split)}

info = []
for x, y in pairs:
    ra, rb = gold_ranks(x)[0], gold_ranks(y)[0]
    info.append({"x": x, "y": y, "ra": ra, "rb": rb})
n = len(info)
miss5 = lambda r: r is None or r > 5
both5 = [i for i in info if miss5(i["ra"]) and miss5(i["rb"])]
both30 = [i for i in info if i["ra"] is None and i["rb"] is None]
print(f"questions with gold: {n}")
print(f"gold paragraph NOT in top 5 (paragraphs only) for BM25: {sum(miss5(i['ra']) for i in info)}, Contriever: {sum(miss5(i['rb']) for i in info)}, BOTH: {len(both5)}")
print(f"gold paragraph NOT in stored top 30 for BM25: {sum(i['ra'] is None for i in info)}, Contriever: {sum(i['rb'] is None for i in info)}, BOTH: {len(both30)}")
print(f"found by exactly one method in top 30: BM25 only {sum(i['ra'] is not None and i['rb'] is None for i in info)}, Contriever only {sum(i['rb'] is not None and i['ra'] is None for i in info)}")
print("both-miss-top-5 by answer_from:", {k: sum(i['x']['answer_from'] == k for i in both5) for k in sorted({i['x']['answer_from'] for i in info})})

rng = random.Random(a.seed)
only5 = [i for i in both5 if i not in both30]
k30 = min(len(both30), (a.n_examples * 6 + 5) // 10)
chosen = rng.sample(both30, k30) + rng.sample(only5, min(len(only5), a.n_examples - k30))
print(f"\n=== {len(chosen)} examples (seeded sample; first those missed in top 30 by both, then those missed in top 5 only) ===")
def show(tag, row):
    print(f"  {tag}:")
    for p in row["retrieved"][:3]:
        sc = f"bm25={p['bm25_score']:.1f}" + (f" contr={p['contriever_score']:.1f}" if "contriever_score" in p else "")
        print(f"    #{p['rank']} [{p['source_type']}] {sc} | {p['text'][:130]}")
for n_, i in enumerate(chosen, 1):
    x = i["x"]; g = text[x["gold_evidence_ids"][0]]
    where = "missed in stored top 30 by BOTH" if i["ra"] is None and i["rb"] is None else f"gold paragraph rank: BM25={i['ra']}, Contriever={i['rb']} (paragraph-only ranks)"
    print(f"\n[{n_}] {x['answer_from']} | {x['answer_type']} | {where}")
    print(f"  Q: {x['question']}")
    print(f"  gold answer: {x['gold_answer_text']}")
    print(f"  GOLD paragraph ({len(x['gold_evidence_ids'])} gold): {g[:260]}")
    show("BM25 top 3", x); show("Contriever top 3", i["y"])

print("\n=== rule-based FLAGS over all questions missed in top 5 by both (observations, not causes) ===")
flags = {"answer string not found in gold paragraph (table may be needed)": 0,
         "<30% of question content words appear in gold paragraph (wording mismatch?)": 0,
         "question contains a number or year": 0,
         "BM25 top-1 is a table row": 0,
         "gold paragraph longer than 120 words": 0,
         "question longer than 25 words": 0}
for i in both5:
    x = i["x"]; g = text[x["gold_evidence_ids"][0]]; gt = set(toks(g))
    q = [t for t in toks(x["question"]) if t not in STOP]
    ans = re.sub(r"[,$%\s]", "", str(x["gold_answer_text"]).lower())
    if not ans or ans not in re.sub(r"[,$%\s]", "", g.lower()): flags["answer string not found in gold paragraph (table may be needed)"] += 1
    if q and sum(t in gt for t in q) / len(q) < 0.3: flags["<30% of question content words appear in gold paragraph (wording mismatch?)"] += 1
    if any(re.fullmatch(r"\d[\d.]*", t) for t in toks(x["question"])): flags["question contains a number or year"] += 1
    if x["retrieved"][0]["source_type"] == "table_row": flags["BM25 top-1 is a table row"] += 1
    if len(g.split()) > 120: flags["gold paragraph longer than 120 words"] += 1
    if len(x["question"].split()) > 25: flags["question longer than 25 words"] += 1
for k, v in flags.items(): print(f"  {v:3d} / {len(both5)}  {k}")