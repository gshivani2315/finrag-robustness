"""
inject_noise.py
---------------
Systematic evidence noise injection across three distinct noise modalities:
  1. Conflicting Information (conflict)
  2. Redundancy (redundancy)
  3. Semantic Distraction (semantic)

Severities:
  - Level 0: Clean baseline (no noise)
  - Level 1: Light noise (1-2 instances)
  - Level 2: Moderate noise (3-4 instances)
  - Level 3: Heavy noise (5+ instances)
"""

from __future__ import annotations

import copy
import random
import re
from typing import Any, Dict, List, Optional


SEVERITY_COUNTS = {
    0: 0,
    1: 1,
    2: 3,
    3: 5,
}

_NUM_RE = re.compile(r"(\$?\s*)(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(\s*%?)")


def _perturb_numbers(text: str, rng: random.Random) -> str:
    """Perturb numbers in text by +/- 10% to 50% to create realistic financial conflicts."""
    def repl(match: re.Match) -> str:
        prefix = match.group(1)
        raw_num = match.group(2).replace(",", "")
        suffix = match.group(3)
        try:
            val = float(raw_num)
            # Avoid perturbing obvious 4-digit years (1900-2099)
            if 1900 <= val <= 2099 and "." not in raw_num and not prefix and not suffix:
                return match.group(0)
            # Perturb value by 10% to 40%
            factor = rng.choice([0.75, 0.85, 1.15, 1.25, 1.35])
            new_val = val * factor
            if "." in raw_num:
                new_str = f"{new_val:.2f}"
            else:
                new_str = f"{int(round(new_val)):,}"
            return f"{prefix}{new_str}{suffix}"
        except (ValueError, OverflowError):
            return match.group(0)

    perturbed = _NUM_RE.sub(repl, text)
    # If no number was changed, append a conflicting financial disclosure line
    if perturbed == text:
        alt_val = rng.randint(50, 950)
        perturbed += f" (Revised interim figure: ${alt_val}.5 million)."
    return perturbed


def _create_conflict_passage(orig: Dict[str, Any], idx: int, rng: random.Random) -> Dict[str, Any]:
    text = orig.get("text", "")
    conflicted_text = _perturb_numbers(text, rng)
    return {
        "passage_id": f"{orig.get('passage_id', 'p')}_conflict_{idx}",
        "text": conflicted_text,
        "source_type": orig.get("source_type", "paragraph"),
        "doc_id": orig.get("doc_id", "conflict_doc"),
        "is_noise": True,
        "noise_type": "conflict",
    }


def _create_redundant_passage(orig: Dict[str, Any], idx: int, rng: random.Random) -> Dict[str, Any]:
    templates = [
        "{text}",
        "As noted previously: {text}",
        "Management summary confirmation: {text}",
        "Footnote cross-reference: {text}",
        "Audited reconciliation confirms: {text}",
    ]
    template = rng.choice(templates)
    return {
        "passage_id": f"{orig.get('passage_id', 'p')}_redundant_{idx}",
        "text": template.format(text=orig.get("text", "")),
        "source_type": orig.get("source_type", "paragraph"),
        "doc_id": orig.get("doc_id", "redundant_doc"),
        "is_noise": True,
        "noise_type": "redundancy",
    }


_DISTRACTOR_TEMPLATES = [
    "Operating cash flow for the corporate division was $45.2 million compared to $38.7 million in the prior period.",
    "Depreciation and amortization expenses totaled $112.4 million, representing an increase of 4.5% year-over-year.",
    "The effective tax rate for international jurisdictions was 18.2%, down from 22.0% reported previously.",
    "Restructuring and transaction-related costs were approximately $14.8 million for the fiscal quarter.",
    "General and administrative expenses increased to $78.3 million driven primarily by stock-based compensation.",
    "Capital expenditures for property, plant and equipment were $230.5 million during the twelve months ended.",
]


def _create_semantic_passage(idx: int, corpus: Optional[List[Dict[str, Any]]], rng: random.Random) -> Dict[str, Any]:
    if corpus and len(corpus) > 0:
        pick = rng.choice(corpus)
        return {
            "passage_id": f"{pick.get('passage_id', 'distractor')}_semantic_{idx}",
            "text": pick.get("text", ""),
            "source_type": pick.get("source_type", "paragraph"),
            "doc_id": pick.get("doc_id", "distractor_doc"),
            "is_noise": True,
            "noise_type": "semantic",
        }
    text = rng.choice(_DISTRACTOR_TEMPLATES)
    return {
        "passage_id": f"distractor_{idx}",
        "text": text,
        "source_type": "paragraph",
        "doc_id": f"distractor_doc_{idx}",
        "is_noise": True,
        "noise_type": "semantic",
    }


def inject_noise(
    passages: List[Dict[str, Any]],
    noise_type: str,
    severity: int,
    corpus: Optional[List[Dict[str, Any]]] = None,
    seed: int = 42,
    max_total_passages: int = 10,
) -> List[Dict[str, Any]]:
    """Inject noise into a list of retrieved passages according to noise_type and severity.

    Parameters:
      passages: Clean retrieved passages (e.g. top-5).
      noise_type: 'clean', 'conflict', 'redundancy', or 'semantic'.
      severity: 0 (clean), 1 (light), 2 (moderate), 3 (heavy).
      corpus: Optional full corpus for sampling real distractors.
      seed: Random seed for deterministic reproducibility.
      max_total_passages: Upper bound on final passage list size.
    """
    if severity == 0 or noise_type in ("clean", "none", None):
        return [copy.deepcopy(p) for p in passages]

    count = SEVERITY_COUNTS.get(severity, severity)
    rng = random.Random(seed)
    result = [copy.deepcopy(p) for p in passages]

    if not result:
        # Fallback if passages empty
        result = [{"passage_id": "dummy_p", "text": "No content available.", "source_type": "paragraph"}]

    injected = []
    if noise_type == "conflict":
        for i in range(count):
            target = rng.choice(result[:min(len(result), 3)])
            injected.append(_create_conflict_passage(target, i + 1, rng))

    elif noise_type == "redundancy":
        for i in range(count):
            target = result[0]  # Duplicate the most relevant passage
            injected.append(_create_redundant_passage(target, i + 1, rng))

    elif noise_type in ("semantic", "distraction"):
        for i in range(count):
            injected.append(_create_semantic_passage(i + 1, corpus, rng))

    else:
        raise ValueError(f"Unknown noise_type: {noise_type}. Expected 'conflict', 'redundancy', or 'semantic'.")

    # Interleave noise into the passage ranking
    final_passages = []
    inj_idx = 0
    for p in result:
        final_passages.append(p)
        if inj_idx < len(injected):
            final_passages.append(injected[inj_idx])
            inj_idx += 1
    while inj_idx < len(injected):
        final_passages.append(injected[inj_idx])
        inj_idx += 1

    return final_passages[:max_total_passages]

