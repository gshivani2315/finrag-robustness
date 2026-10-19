"""
Adapter around the OFFICIAL TAT-QA scorer. Skipped until the official files are downloaded
(python -m src.official_tatqa --download) and scipy/pandas are installed.
Every expected value below follows from the official rules (scale applied to numbers, multi-span order ignored,
arithmetic/count F1 = EM, empty prediction = 0), not from copying the scorer's output.
"""
import os
import pytest
from src.official_tatqa import (DEFAULT_DIR, OfficialScorer, OfficialScorerUnavailable, normalise_prediction,
                                normalise_scale, verify_files)

try:
    SC = OfficialScorer(DEFAULT_DIR)
    SKIP = None
except OfficialScorerUnavailable as e:
    SC, SKIP = None, str(e)
need_official = pytest.mark.skipif(SC is None, reason=f"official scorer not ready: {SKIP}")


def gold(answer, answer_type, scale="", answer_from="table"):
    return {"uid": "q", "answer": answer, "answer_type": answer_type, "scale": scale, "answer_from": answer_from}


def test_normalisers_do_not_need_the_official_files():
    assert normalise_prediction(0) == ["0"]                  # a numeric zero must not look like "no answer"
    assert normalise_prediction(12.50) == ["12.5"]
    assert normalise_prediction("") is None and normalise_prediction(None) is None
    assert normalise_prediction([" a ", "", None, "b"]) == ["a", "b"]
    assert normalise_scale(None) == "" and normalise_scale(" Million ") == "million"


def test_pinned_hashes_reported_as_expected_keys():
    assert set(verify_files(os.path.join("no", "such", "dir"))) == {"tatqa_metric.py", "tatqa_utils.py"}


@need_official
def test_arithmetic_percent_and_scale_rules():
    g = gold(33.69, "arithmetic", "percent")
    assert SC.score(g, "33.69%", "")["em"] == 1.0              # '%' in the answer itself
    assert SC.score(g, "33.69", "percent")["em"] == 1.0        # number + the right scale
    assert SC.score(g, "0.3369", "")["em"] == 1.0              # fraction form, no scale
    assert SC.score(g, "33.69", "")["em"] == 0.0               # percent number with the scale omitted
    assert SC.score(g, "34", "percent")["em"] == 0.0
    assert SC.score(g, "33.69%", "")["f1"] == 1.0 and SC.score(g, "34", "percent")["f1"] == 0.0   # F1 = EM here


@need_official
def test_scale_must_be_supplied_for_scaled_answers():
    g = gold(["$1,496.5"], "span", "million", "table")
    assert SC.score(g, "$1,496.5", "million")["em"] == 1.0
    assert SC.score(g, "1496.5", "million")["em"] == 1.0       # formatting differences are ignored
    assert SC.score(g, "$1,496.5", "")["em"] == 0.0            # omitting the scale changes the value
    assert SC.score(g, "$1,496.5", "")["scale_ok"] == 0.0 and SC.score(g, "x", "million")["scale_ok"] == 1.0


@need_official
def test_multi_span_order_partial_credit_and_empty():
    g = gold(["alpha beta", "gamma"], "multi-span", "", "text")
    assert SC.score(g, ["gamma", "alpha beta"], "")["em"] == 1.0   # order ignored
    # the official scorer SORTS AND JOINS all spans into one string, so partial credit is token-level:
    # gold tokens {alpha, beta, gamma}, predicted {alpha, beta} -> precision 1, recall 2/3, F1 = 0.8
    part = SC.score(g, ["alpha beta"], "")
    assert part["em"] == 0.0 and part["f1"] == pytest.approx(0.8)
    assert SC.score(g, None, "") == {"em": 0.0, "f1": 0.0, "scale_ok": 1.0}
    assert SC.score(g, "", "")["em"] == 0.0


@need_official
def test_count_and_negative_numbers_and_numeric_zero():
    assert SC.score(gold(3, "count"), 3, "")["em"] == 1.0 and SC.score(gold(3, "count"), "4", "")["em"] == 0.0
    neg = gold(["(114)"], "span", "million", "table")
    assert SC.score(neg, "-114", "million")["em"] == 1.0       # parentheses mean negative
    assert SC.score(neg, "114", "million")["em"] == 0.0
    assert SC.score(gold(0, "arithmetic"), 0, "")["em"] == 1.0  # adapter: numeric 0 is not "empty"


@need_official
@pytest.mark.skipif(not os.path.exists("data/tatqa_dataset_dev.json"), reason="dev data not downloaded")
def test_gold_answers_score_perfectly_on_all_dev_questions():
    """Plumbing check on real data (NOT a model result): every dev gold answer scored as a prediction is 1.0."""
    from src.data_loader import load_split
    _, recs = load_split("dev")
    for r in recs:
        g = {"uid": r["question_id"], "answer": r["answer"], "answer_type": r["answer_type"],
             "scale": r["scale"], "answer_from": r["answer_from"]}
        s = SC.score(g, r["answer"], r["scale"])
        assert s["em"] == 1.0 and s["f1"] == 1.0 and s["scale_ok"] == 1.0, r["question_id"]