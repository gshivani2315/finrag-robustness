import pytest
from src.answer_metrics import exact_match, f1_score, evaluate_answers


def test_number_formats_match():
    assert exact_match("$1,496.5", ["1496.5"]) == 1.0
    assert exact_match("5.0", 5) == 1.0
    assert exact_match("33.69%", 33.69) == 1.0
    assert exact_match("-5", "5") == 0.0


def test_f1_hand_example():
    assert f1_score("the cat sat", "cat sat down") == pytest.approx(0.8)


def test_empty_prediction():
    assert exact_match("", ["x"]) == 0.0
    assert f1_score("", ["x"]) == 0.0


def test_multi_span_order_insensitive():
    assert exact_match(["b", "a"], ["a", "b"]) == 1.0
    assert exact_match(["a"], ["a", "b"]) == 0.0


def test_evaluate_answers_and_length_check():
    r = evaluate_answers(["1", "x"], ["1", "y"])
    assert r == {"n": 2, "em": 0.5, "f1": 0.5}
    with pytest.raises(ValueError):
        evaluate_answers(["1"], [])

