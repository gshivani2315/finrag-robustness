"""
answer_eval.py - score a PREDICTION FILE (EM / F1) against the labelled TAT-QA questions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict

from .answer_metrics import exact_match, f1_score
from .data_loader import load_split
from .official_tatqa import (DEFAULT_DIR, VALID_SCALES, OfficialScorer, OfficialScorerUnavailable,
                             normalise_prediction, normalise_scale)

ID_KEYS = ("question_id", "uid")
ANSWER_KEYS = ("predicted_answer", "prediction", "pred_answer")
SCALE_KEYS = ("scale", "predicted_scale", "pred_scale")
DEFAULT_EVIDENCE_SETUP = "retrieved_top5_table_aware"


class PredictionFileError(ValueError):
    """The prediction file is unusable (the message says why)."""


def read_ids(path: str) -> list[str]:
    with open(path, encoding="utf-8-sig") as f:
        ids = [line.strip() for line in f if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: contains duplicate question ids")
    return ids


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _first(d: dict, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def gold_view(rec: dict) -> dict:
    """The TAT-QA question fields the official scorer needs."""
    return {"uid": rec["question_id"], "answer": rec["answer"], "answer_type": rec["answer_type"],
            "scale": rec["scale"], "answer_from": rec["answer_from"]}


def load_prediction_file(path: str) -> tuple[list[dict], list[str]]:
    """Returns (rows, warnings). Raises PredictionFileError for unusable files."""
    if not os.path.exists(path):
        raise PredictionFileError(f"prediction file not found: {path}")
    rows, warnings = [], []
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    if not text.strip():
        raise PredictionFileError(f"{path} is empty")
    if path.lower().endswith(".json"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise PredictionFileError(f"{path}: invalid JSON ({e})") from e
        if not isinstance(data, dict):
            raise PredictionFileError(f"{path}: expected a dict {{question_id: [answer, scale]}}")
        for qid, val in data.items():
            if not (isinstance(val, (list, tuple)) and len(val) == 2):
                raise PredictionFileError(f"{path}: value for {qid} must be [answer, scale]")
            rows.append({"question_id": qid, "answer": val[0], "scale": val[1], "degradation_type": None,
                         "severity": None, "seed": None})
    else:
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise PredictionFileError(f"{path} line {n}: invalid JSON ({e})") from e
            if not isinstance(obj, dict):
                raise PredictionFileError(f"{path} line {n}: expected a JSON object")
            qid = _first(obj, ID_KEYS)
            if not qid:
                raise PredictionFileError(f"{path} line {n}: missing question_id")
            rows.append({"question_id": qid, "answer": _first(obj, ANSWER_KEYS), "scale": _first(obj, SCALE_KEYS),
                         "degradation_type": obj.get("degradation_type"), "severity": obj.get("severity"),
                         "seed": obj.get("seed"), "cited_passages": obj.get("cited_passages", [])})
    seen = Counter((r["question_id"], r["degradation_type"], r["severity"], r["seed"]) for r in rows)
    dups = [k for k, c in seen.items() if c > 1]
    if dups:
        raise PredictionFileError(f"{path}: {len(dups)} duplicated (question_id, degradation_type, severity, seed), "
                                  f"first: {dups[0]}")
    bad_scale = Counter(normalise_scale(r["scale"]) for r in rows if normalise_scale(r["scale"]) not in VALID_SCALES)
    if bad_scale:
        warnings.append(f"unrecognised scale values {dict(bad_scale)}; allowed: {list(VALID_SCALES)} "
                        "(they are kept as written and will not match the gold scale)")
    n_empty = sum(normalise_prediction(r["answer"]) is None for r in rows)
    if n_empty:
        warnings.append(f"{n_empty} rows have an empty predicted_answer (scored 0)")
    return rows, warnings


def score_rows(rows: list[dict], gold_by_id: dict, scorer: OfficialScorer | None,
               expected_ids: list[str] | None = None, missing_policy: str = "zero") -> list[dict]:
    if missing_policy not in ("zero", "skip"):
        raise ValueError("missing_policy must be 'zero' or 'skip'")
    unknown = sorted({r["question_id"] for r in rows if r["question_id"] not in gold_by_id})
    if unknown:
        raise KeyError(f"{len(unknown)} question ids are not in the gold split, e.g. {unknown[:3]}")
    work = [dict(r, missing=False) for r in rows]
    if expected_ids is not None:
        groups = {(r["degradation_type"], r["severity"], r["seed"]) for r in rows}
        present = {(r["question_id"], r["degradation_type"], r["severity"], r["seed"]) for r in rows}
        for dt, sev, seed in sorted(groups, key=str):
            for qid in expected_ids:
                if (qid, dt, sev, seed) not in present:
                    work.append({"question_id": qid, "answer": None, "scale": None, "degradation_type": dt,
                                 "severity": sev, "seed": seed, "missing": True, "cited_passages": []})
    out = []
    for r in work:
        if r["missing"] and missing_policy == "skip":
            continue
        g = gold_by_id[r["question_id"]]
        ans = None if r["missing"] else r["answer"]
        simple_em = exact_match(ans, g["answer"]) if normalise_prediction(ans) is not None else 0.0
        simple_f1 = f1_score(ans, g["answer"]) if normalise_prediction(ans) is not None else 0.0
        o = scorer.score(gold_view(g), ans, r["scale"]) if scorer else {"em": None, "f1": None, "scale_ok": None}

        # Citation accuracy check
        cited = r.get("cited_passages") or []
        gold_e = g.get("gold_evidence_ids") or []
        cit_acc = float(len(set(cited) & set(gold_e)) > 0) if gold_e else 1.0

        out.append({"question_id": r["question_id"], "degradation_type": r["degradation_type"],
                    "severity": r["severity"], "seed": r["seed"], "missing": r["missing"],
                    "answer_type": g["answer_type"], "answer_from": g["answer_from"],
                    "evidence_status": g["evidence_status"], "gold_scale": g["scale"],
                    "gold_answer": json.dumps(g["answer"], ensure_ascii=False),
                    "predicted_answer": json.dumps(ans, ensure_ascii=False), "predicted_scale": r["scale"],
                    "official_em": o["em"], "official_f1": o["f1"], "scale_ok": o["scale_ok"],
                    "simplified_em": simple_em, "simplified_f1": simple_f1,
                    "citation_accuracy": cit_acc})
    return out


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _block(rs: list[dict]) -> dict:
    return {"n": len(rs), "n_missing": sum(r["missing"] for r in rs),
            "official_em": _mean(r["official_em"] for r in rs), "official_f1": _mean(r["official_f1"] for r in rs),
            "scale_accuracy": _mean(r["scale_ok"] for r in rs),
            "simplified_em": _mean(r["simplified_em"] for r in rs), "simplified_f1": _mean(r["simplified_f1"] for r in rs),
            "citation_accuracy": _mean(r["citation_accuracy"] for r in rs)}


def summarize(results: list[dict]) -> dict:
    def grouped(key):
        g = defaultdict(list)
        for r in results:
            g[str(key(r))].append(r)
        return {k: _block(v) for k, v in sorted(g.items())}
    return {"overall": _block(results),
            "by_condition": grouped(lambda r: f"{r['degradation_type'] or 'unspecified'}|severity={r['severity']}"),
            "by_answer_type": grouped(lambda r: r["answer_type"]),
            "by_answer_from": grouped(lambda r: r["answer_from"]),
            "by_evidence_status": grouped(lambda r: r["evidence_status"])}


def _gold(split: str, data_dir: str) -> dict:
    if split == "test":
        raise SystemExit("The test split has no gold answers, so EM/F1 cannot be computed on it.")
    _, recs = load_split(split, data_dir)
    return {r["question_id"]: r for r in recs}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score a prediction file (EM/F1) against labelled TAT-QA questions")
    ap.add_argument("--predictions")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--question-ids", help="text file, one question id per line")
    ap.add_argument("--out", help="folder for answer_eval_summary.json and answer_eval_per_question.csv")
    ap.add_argument("--missing-policy", default="zero", choices=["zero", "skip"])
    ap.add_argument("--evidence-setup", default=DEFAULT_EVIDENCE_SETUP)
    ap.add_argument("--official-dir", default=DEFAULT_DIR)
    ap.add_argument("--no-official", action="store_true", help="skip the official scorer")
    ap.add_argument("--write-template", metavar="PATH", help="write an empty prediction file for --question-ids")
    ap.add_argument("--check-ids", action="store_true", help="check --question-ids against the data")
    ap.add_argument("--retrieval-run", help="with --check-ids: a retrieval JSONL whose question ids must match")
    a = ap.parse_args(argv)

    if a.write_template or a.check_ids:
        if not a.question_ids:
            raise SystemExit("--question-ids is required for --write-template / --check-ids")
        gold = _gold(a.split, a.data_dir)
        ids = read_ids(a.question_ids)
        missing = [i for i in ids if i not in gold]
        if a.check_ids:
            print(f"{a.question_ids}: {len(ids)} ids, {len(missing)} not found in the {a.split} split")
            have = [gold[i] for i in ids if i in gold]
            print("answer_type:", dict(Counter(r["answer_type"] for r in have)))
            print("answer_from:", dict(Counter(r["answer_from"] for r in have)))
            print("evidence_status:", dict(Counter(r["evidence_status"] for r in have)))
            return 0 if not missing else 2
        if missing:
            raise SystemExit(f"{len(missing)} ids are not in the {a.split} split, e.g. {missing[:3]}")
        os.makedirs(os.path.dirname(a.write_template) or ".", exist_ok=True)
        with open(a.write_template, "w", encoding="utf-8") as f:
            for i in ids:
                f.write(json.dumps({"question_id": i, "question": gold[i]["question"], "predicted_answer": None,
                                    "scale": None}, ensure_ascii=False) + "\n")
        print(f"wrote {len(ids)} empty rows to {a.write_template}")
        return 0

    if not a.predictions:
        raise SystemExit("--predictions is required (or use --write-template / --check-ids)")
    gold = _gold(a.split, a.data_dir)
    try:
        rows, warnings = load_prediction_file(a.predictions)
    except PredictionFileError as e:
        raise SystemExit(f"cannot score: {e}")
    expected = read_ids(a.question_ids) if a.question_ids else None
    scorer, official_status = None, {"used": False}
    if not a.no_official:
        try:
            scorer = OfficialScorer(a.official_dir)
            official_status = {"used": True, "files_sha256": scorer.files}
        except OfficialScorerUnavailable as e:
            official_status = {"used": False, "reason": str(e)}
            warnings.append(f"OFFICIAL SCORER UNAVAILABLE ({e}); only simplified scorer was computed")
    try:
        results = score_rows(rows, gold, scorer, expected, a.missing_policy)
    except KeyError as e:
        raise SystemExit(f"cannot score: {e.args[0]}")
    summary = {"status": "computed",
               "config": {"predictions": a.predictions, "predictions_sha256": sha256_file(a.predictions),
                          "split": a.split, "question_ids_file": a.question_ids,
                          "n_expected": len(expected) if expected is not None else None,
                          "n_prediction_rows": len(rows), "missing_policy": a.missing_policy,
                          "evidence_setup": a.evidence_setup, "official_scorer": official_status},
               "warnings": warnings, **summarize(results)}
    o = summary["overall"]
    print(f"scored {o['n']} rows; EM {o['official_em'] or o['simplified_em']} F1 {o['official_f1'] or o['simplified_f1']}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "answer_eval_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(os.path.join(a.out, "answer_eval_per_question.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

