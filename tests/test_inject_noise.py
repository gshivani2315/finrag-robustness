"""Unit tests for Noise Injector (src/inject_noise.py)."""
import pytest
from src.inject_noise import inject_noise


PASSAGES = [
    {"passage_id": "p1", "text": "Net sales were $1,452.4 million in 2019.", "source_type": "paragraph"},
    {"passage_id": "p2", "text": "Operating profit was $312.5 million.", "source_type": "paragraph"},
]


def test_clean_baseline_unaltered():
    out = inject_noise(PASSAGES, "clean", 0)
    assert len(out) == 2
    assert out[0]["text"] == PASSAGES[0]["text"]


def test_conflict_noise_injection():
    out = inject_noise(PASSAGES, "conflict", severity=2, seed=42)
    assert len(out) > len(PASSAGES)
    injected = [p for p in out if p.get("is_noise")]
    assert len(injected) > 0
    assert all(p["noise_type"] == "conflict" for p in injected)


def test_redundancy_noise_injection():
    out = inject_noise(PASSAGES, "redundancy", severity=1, seed=42)
    assert len(out) > len(PASSAGES)
    injected = [p for p in out if p.get("is_noise")]
    assert any("p1" in p["passage_id"] for p in injected)


def test_semantic_distraction_injection():
    out = inject_noise(PASSAGES, "semantic", severity=2, seed=42)
    injected = [p for p in out if p.get("is_noise")]
    assert len(injected) > 0
    assert all(p["noise_type"] == "semantic" for p in injected)


def test_noise_determinism():
    out1 = inject_noise(PASSAGES, "conflict", severity=2, seed=99)
    out2 = inject_noise(PASSAGES, "conflict", severity=2, seed=99)
    assert [p["text"] for p in out1] == [p["text"] for p in out2]

