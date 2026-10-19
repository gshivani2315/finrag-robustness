"""Prediction-file loading, validation and scoring. Predictions here are TEST DATA built in memory; they are not results."""
import json
import pytest
from src.answer_eval import (PredictionFileError, load_prediction_file, read_ids, score_rows, summarize, main)
from src.official_tatqa import DEFAULT_DIR, OfficialScorer, OfficialScorerUnavailable

try:
    SC = OfficialScorer(DEFAULT_DIR)
except OfficialScorerUnavailable:
    SC = None
need_official = pytest.mark.skipif(SC is None, reason="official scorer not ready")


def G(qid, answer, answer_type, scale="", answer_from="table", status="no_gold"):
    return {"question_id": qid, "answer": answer, "answer_type": answer_type, "scale": scale,
            "answer_from": answer_from, "evidence_status": status, "question": "q?"}


GOLD = {"a": G("a", [" $5 "], "span", "million"), "b": G("b", 12.5, "arithmetic", "percent", "table-text", "paragraph_gold_partial"),
        "c": G("c", ["x", "y"], "multi-span", "", "text", "paragraph_gold_complete")}


def write(tmp_path, rows, name="p.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def test_loads_jsonl_with_aliases_and_official_json(tmp_path):
    rows, w = load_prediction_file(write(tmp_path, [{"question_id": "a", "prediction": "5", "predicted_scale": "million"},
                                                    {"uid": "b", "predicted_answer": 12.5, "scale": "percent", "severity": 1}]))
    assert [(r["question_id"], r["answer"], r["scale"]) for r in rows] == [("a", "5", "million"), ("b", 12.5, "percent")]
    assert rows[1]["severity"] == 1 and w == []
    p = tmp_path / "official.json"
    p.write_text(json.dumps({"a": ["5", "million"], "c": [["x", "y"], ""]}), encoding="utf-8")
    rows, _ = load_prediction_file(str(p))
    assert {r["question_id"] for r in rows} == {"a", "c"}


def test_bad_files_are_rejected_with_clear_errors(tmp_path):
    with pytest.raises(PredictionFileError, match="not found"):
        load_prediction_file(str(tmp_path / "nope.jsonl"))
    with pytest.raises(PredictionFileError, match="empty"):
        load_prediction_file(write(tmp_path, [], "e.jsonl"))
    with pytest.raises(PredictionFileError, match="missing question_id"):
        load_prediction_file(write(tmp_path, [{"predicted_answer": "1"}], "m.jsonl"))
    with pytest.raises(PredictionFileError, match="duplicated"):
        load_prediction_file(write(tmp_path, [{"question_id": "a", "predicted_answer": "1"}] * 2, "d.jsonl"))
    (tmp_path / "bad.jsonl").write_text("{not json}", encoding="utf-8")
    with pytest.raises(PredictionFileError, match="invalid JSON"):
        load_prediction_file(str(tmp_path / "bad.jsonl"))


def test_same_question_in_different_conditions_is_allowed(tmp_path):
    rows, _ = load_prediction_file(write(tmp_path, [
        {"question_id": "a", "predicted_answer": "5", "degradation_type": "clean", "severity": 0},
        {"question_id": "a", "predicted_answer": "6", "degradation_type": "conflict", "severity": 1}]))
    assert len(rows) == 2


def test_warnings_for_empty_answers_and_unknown_scales(tmp_path):
    _, w = load_prediction_file(write(tmp_path, [{"question_id": "a", "predicted_answer": "", "scale": "millions"}]))
    assert any("empty predicted_answer" in x for x in w) and any("unrecognised scale" in x for x in w)


def test_unknown_ids_and_missing_policy():
    rows = [{"question_id": "a", "answer": "5", "scale": "million", "degradation_type": None, "severity": None, "seed": None}]
    with pytest.raises(KeyError, match="not in the gold split"):
        score_rows([dict(rows[0], question_id="zzz")], GOLD, None)
    zero = score_rows(rows, GOLD, None, expected_ids=["a", "b"], missing_policy="zero")
    assert {r["question_id"]: r["missing"] for r in zero} == {"a": False, "b": True}
    assert [r for r in zero if r["missing"]][0]["simplified_em"] == 0.0
    skip = score_rows(rows, GOLD, None, expected_ids=["a", "b"], missing_policy="skip")
    assert [r["question_id"] for r in skip] == ["a"]
    with pytest.raises(ValueError):
        score_rows(rows, GOLD, None, missing_policy="drop")


@need_official
def test_scoring_summary_and_grouping():
    base = {"degradation_type": "clean", "severity": 0, "seed": 1}
    rows = [dict(base, question_id="a", answer="$5", scale="million"),        # correct
            dict(base, question_id="b", answer="12.5%", scale=""),            # correct
            dict(base, question_id="c", answer=["x"], scale="")]               # partial
    res = score_rows(rows, GOLD, SC, expected_ids=["a", "b", "c"])
    s = summarize(res)
    assert s["overall"]["n"] == 3 and s["overall"]["n_missing"] == 0
    assert s["overall"]["official_em"] == pytest.approx(2 / 3)
    # 'c': gold tokens {x, y}, predicted {x}: P=1, R=1/2 -> F1 = 0.67 (the official scorer rounds F1 to 2 decimals)
    assert s["overall"]["official_f1"] == pytest.approx((1 + 1 + 0.67) / 3)
    assert s["by_answer_type"]["multi-span"]["official_f1"] == pytest.approx(0.67)
    assert set(s["by_condition"]) == {"clean|severity=0"}
    assert s["by_evidence_status"]["paragraph_gold_complete"]["n"] == 1


def test_cli_refuses_test_split_and_missing_inputs(tmp_path, capsys):
    with pytest.raises(SystemExit, match="test split has no gold"):
        main(["--predictions", "x.jsonl", "--split", "test"])
    with pytest.raises(SystemExit, match="--predictions is required"):
        main([])
    with pytest.raises(SystemExit, match="--question-ids is required"):
        main(["--write-template", str(tmp_path / "t.jsonl")])