"""
compare_scorers.py - how do the OFFICIAL TAT-QA scorer and the SIMPLIFIED scorer behave on the same answers?
No model is involved: every "prediction" is the dev-set GOLD answer, left as is or deliberately altered in one way.
This validates the scorers (and shows why the official one is primary); it says nothing about model quality.
Needs: data\\tatqa_dataset_dev.json and the official scorer (python -m src.official_tatqa --download).
Run from the project folder:   python scripts\\compare_scorers.py
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.answer_metrics import exact_match, f1_score
from src.data_loader import load_split
from src.official_tatqa import OfficialScorer

_, recs = load_split("dev")
sc = OfficialScorer()
gold = lambda r: {"uid": r["question_id"], "answer": r["answer"], "answer_type": r["answer_type"],
                  "scale": r["scale"], "answer_from": r["answer_from"]}

def run(name, rows, make):
    n = len(rows); oe = of = se = sf = 0.0
    for r in rows:
        pred, scale = make(r)
        o = sc.score(gold(r), pred, scale)
        oe += o["em"]; of += o["f1"]; se += exact_match(pred, r["answer"]); sf += f1_score(pred, r["answer"])
    print(f"{name:58s} n={n:4d} | EM official {oe/n:.3f} simplified {se/n:.3f} | F1 official {of/n:.3f} simplified {sf/n:.3f}")

arith = [r for r in recs if r["answer_type"] == "arithmetic"]
scaled = [r for r in recs if r["scale"]]
pct = [r for r in arith if r["scale"] == "percent"]
multi = [r for r in recs if r["answer_type"] == "multi-span" and len(r["answer"]) > 1]
paren = [r for r in recs if isinstance(r["answer"], list) and any(re.search(r"\(\s*[\d.,]+\s*\)", str(a)) for a in r["answer"])]
print("Scorer comparison on GOLD-DERIVED answers (dev split). Not model results.\n")
run("gold answer + gold scale (identity)", recs, lambda r: (r["answer"], r["scale"]))
run("gold answer, scale OMITTED (gold has a scale)", scaled, lambda r: (r["answer"], ""))
run("arithmetic, number + 1 (wrong)", arith, lambda r: (float(r["answer"]) + 1, r["scale"]))
run("percent questions: '33.69%' with scale omitted", pct, lambda r: (f"{r['answer']}%", ""))
run("percent questions: fraction x/100, no scale", pct, lambda r: (round(float(r["answer"]) / 100, 6), ""))
run("multi-span: last span dropped", multi, lambda r: (r["answer"][:-1], r["scale"]))
run("multi-span: reversed order", multi, lambda r: (r["answer"][::-1], r["scale"]))
run("answers with (parenthesised) negatives written as -x", paren,
    lambda r: ([re.sub(r"\(\s*([\d.,]+)\s*\)", r"-\1", str(a)) for a in r["answer"]], r["scale"]))
run("empty prediction", recs, lambda r: ("", r["scale"]))