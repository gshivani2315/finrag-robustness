"""
official_tatqa.py - adapter around the OFFICIAL TAT-QA scorer (github.com/NExTplusplus/TAT-QA, MIT licence).
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
import urllib.request
import warnings

from .answer_metrics import _to_text

RAW_URL = "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/{name}"
PINNED_SHA256 = {
    "tatqa_metric.py": "2aeeac479f89f8c76300af1cc0e8d098eb86af84bc386b38b6ab4af484a6dea8",
    "tatqa_utils.py": "a84bb2f960737cf0a53733637a674cc4b20ef030a2be6a4b21dc2c4356f415ec",
}
DEFAULT_DIR = os.path.join("third_party", "tatqa")
VALID_SCALES = ("", "thousand", "million", "billion", "percent")


class OfficialScorerUnavailable(RuntimeError):
    """Raised when the official scorer cannot be used (files missing/changed, scipy/pandas missing)."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def verify_files(dest_dir: str = DEFAULT_DIR) -> dict[str, str]:
    """name -> 'ok' | 'missing' | 'hash mismatch'."""
    out = {}
    for name, want in PINNED_SHA256.items():
        p = os.path.join(dest_dir, name)
        out[name] = "missing" if not os.path.exists(p) else ("ok" if _sha256(p) == want else "hash mismatch")
    return out


def download(dest_dir: str = DEFAULT_DIR) -> dict[str, str]:
    """Download missing files, verify every file against the pinned hash. Never overwrites a file silently."""
    os.makedirs(dest_dir, exist_ok=True)
    for name, want in PINNED_SHA256.items():
        p = os.path.join(dest_dir, name)
        if os.path.exists(p):
            continue
        tmp = p + ".part"
        urllib.request.urlretrieve(RAW_URL.format(name=name), tmp)
        if _sha256(tmp) != want:
            os.replace(tmp, p + ".bad")
            raise OfficialScorerUnavailable(
                f"{name}: downloaded file does not match the pinned SHA-256 (upstream may have changed). "
                f"Saved as {p}.bad for inspection; nothing was installed.")
        os.replace(tmp, p)
    status = verify_files(dest_dir)
    if any(v != "ok" for v in status.values()):
        raise OfficialScorerUnavailable(f"official files not usable: {status}")
    return status


def normalise_prediction(pred) -> list[str] | None:
    """Prediction (str / number / list / None) -> list of non-empty strings, or None if empty."""
    if pred is None:
        return None
    items = pred if isinstance(pred, (list, tuple)) else [pred]
    out = [s for s in (_to_text(x).strip() for x in items if x is not None) if s]
    return out or None


def normalise_scale(scale) -> str:
    return "" if scale is None else str(scale).strip().lower()


class OfficialScorer:
    """Scores ONE question at a time with the official TaTQAEmAndF1 class."""

    def __init__(self, dest_dir: str = DEFAULT_DIR):
        status = verify_files(dest_dir)
        if any(v != "ok" for v in status.values()):
            raise OfficialScorerUnavailable(
                f"official TAT-QA files not ready {status}. Run: python -m src.official_tatqa --download")
        self.files = {n: _sha256(os.path.join(dest_dir, n)) for n in PINNED_SHA256}
        path = os.path.abspath(dest_dir)
        sys.path.insert(0, path)
        try:
            with warnings.catch_warnings():     # upstream code has harmless invalid-escape SyntaxWarnings
                warnings.simplefilter("ignore", SyntaxWarning)
                self._mod = importlib.import_module("tatqa_metric")
        except ImportError as e:        # pandas / scipy missing
            raise OfficialScorerUnavailable(f"cannot import the official scorer ({e}). "
                                            "Try: pip install scipy pandas") from e
        finally:
            sys.path.remove(path)

    def score(self, gold: dict, prediction, pred_scale) -> dict:
        """
        gold: {"uid", "answer", "answer_type", "scale", "answer_from"} (the TAT-QA question fields).
        Returns {"em", "f1", "scale_ok"}.
        """
        g = dict(gold, scale=normalise_scale(gold.get("scale")))
        meter = self._mod.TaTQAEmAndF1()
        scale = normalise_scale(pred_scale)
        meter(ground_truth=g, prediction=normalise_prediction(prediction), pred_scale=scale)
        rec = meter.get_raw()[0]
        return {"em": float(rec["em"]), "f1": float(rec["f1"]), "scale_ok": float(scale == g["scale"])}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fetch / verify the official TAT-QA scorer files")
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.download:
        print("download:", download(a.dir))
    print("verify:", verify_files(a.dir))
    try:
        OfficialScorer(a.dir)
        print("official scorer imports correctly")
    except OfficialScorerUnavailable as e:
        print("NOT usable:", e)

