"""
run_experiment.py
-----------------
Main experimental loop for measuring Financial RAG Robustness Under Evidence Noise.

Supports:
  1. Single condition:
     python src/run_experiment.py --num_examples 200 --noise_type conflict --severity 2
  2. Clean baseline (Day 4):
     python src/run_experiment.py --num_examples 200 --noise_type clean --severity 0
  3. Full 12-condition experimental matrix (Day 6):
     python src/run_experiment.py --full_matrix --num_examples 200 --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .answer_metrics import exact_match, f1_score
from .data_loader import build_qa_records, load_split
from .evaluate import citation_accuracy, recall_at_k
from .inject_noise import inject_noise
from .passage_builder import build_corpus
from .pipeline import answer_question, retrieve, sample_questions


# Synthetic sample records if data is not yet downloaded
SYNTHETIC_DEV_DOCS = [
    {
        "table": {
            "uid": "DOC_FIN_001",
            "table": [
                ["", "2019", "2018"],
                ["Revenues", "$1,452.4", "$1,146.2"],
                ["Operating Income", "$312.5", "$254.1"],
                ["Net Income", "$218.4", "$176.8"],
            ],
        },
        "paragraphs": [
            {"uid": "p1", "order": 1, "text": "Total revenues for fiscal 2019 increased to $1,452.4 million compared to $1,146.2 million in 2018."},
            {"uid": "p2", "order": 2, "text": "Operating margin improved by 1.5% driven by operating income of $312.5 million."},
        ],
        "questions": [
            {"uid": "q1", "order": 1, "question": "What were the revenues in 2019?", "answer": 1452.4, "answer_type": "span", "answer_from": "text", "rel_paragraphs": ["1"], "scale": "million"},
            {"uid": "q2", "order": 2, "question": "What was the change in revenues from 2018 to 2019?", "answer": 306.2, "answer_type": "arithmetic", "answer_from": "text", "rel_paragraphs": ["1"], "scale": "million"},
            {"uid": "q3", "order": 3, "question": "What was the operating margin improvement?", "answer": "1.5%", "answer_type": "span", "answer_from": "text", "rel_paragraphs": ["2"], "scale": "percent"},
            {"uid": "q4", "order": 4, "question": "What was the net income in 2019?", "answer": 218.4, "answer_type": "span", "answer_from": "table", "rel_paragraphs": [], "scale": "million"},
        ],
    }
]


def load_dataset_records(split: str = "dev", data_dir: str = "data") -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load real TAT-QA dataset if present; otherwise fallback to synthetic sample."""
    try:
        docs, recs = load_split(split, data_dir)
        return docs, recs
    except FileNotFoundError:
        print(f"Notice: {split} data not downloaded yet in '{data_dir}'. Using synthetic test documents.")
        return SYNTHETIC_DEV_DOCS, build_qa_records(SYNTHETIC_DEV_DOCS, split)


def run_single_condition(
    questions: List[Dict[str, Any]],
    corpus: List[Dict[str, Any]],
    noise_type: str,
    severity: int,
    top_k: int = 5,
    seed: int = 42,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate questions under a specified noise type and severity."""
    results = []
    error_counts = {
        "mis_retrieval": 0,
        "mis_synthesis": 0,
        "mis_verification": 0,
        "execution_error": 0,
        "calculation_error": 0,
    }

    for q in questions:
        q_text = q["question"]
        gold_answer = q.get("answer")
        gold_eids = q.get("gold_evidence_ids", [])

        # Step 1: Retrieval (Person A)
        try:
            ret_out = retrieve(
                q_text,
                top_k=top_k,
                method="bm25",
                split=q.get("split", "dev"),
                paragraph_context="table_labels",
                reserve_paragraph_slots=4,
            )
            retrieved_passages = ret_out.get("results", [])
        except Exception:
            # Fallback if corpus built directly
            retrieved_passages = corpus[:top_k]

        retrieved_ids = [p.get("passage_id") for p in retrieved_passages]
        ret_recall = recall_at_k(retrieved_ids, gold_eids, k=top_k) if gold_eids else 1.0
        ret_ok = (ret_recall is not None and ret_recall > 0)

        # Step 2: Noise Injection (Person B)
        corrupted_passages = inject_noise(
            retrieved_passages,
            noise_type=noise_type,
            severity=severity,
            corpus=corpus,
            seed=seed,
        )

        # Step 3: Downstream QA (Person B: tagger -> synthesizer -> verifier -> execution)
        qa = answer_question(q_text, corrupted_passages)

        predicted_ans = qa.get("answer")
        is_valid = qa.get("valid", False)
        error_msg = qa.get("error", "")
        program = qa.get("program", "")
        cited_passages = qa.get("cited_passages", [])

        # Step 4: Scoring
        em = exact_match(predicted_ans, gold_answer) if predicted_ans is not None else 0.0
        f1 = f1_score(predicted_ans, gold_answer) if predicted_ans is not None else 0.0
        cit_acc = citation_accuracy(cited_passages, gold_eids) if gold_eids else 1.0
        exec_acc = 1.0 if (is_valid and predicted_ans is not None) else 0.0

        # Error breakdown diagnosis
        if em < 1.0:
            if not ret_ok:
                error_counts["mis_retrieval"] += 1
            elif not is_valid:
                if "malformed" in error_msg or "unknown operation" in error_msg:
                    error_counts["mis_synthesis"] += 1
                else:
                    error_counts["mis_verification"] += 1
            elif predicted_ans is None:
                error_counts["execution_error"] += 1
            else:
                error_counts["calculation_error"] += 1

        row = {
            "question_id": q.get("question_id"),
            "question": q_text,
            "gold_answer": str(gold_answer),
            "predicted_answer": str(predicted_ans) if predicted_ans is not None else "",
            "noise_type": noise_type,
            "severity": severity,
            "program": program,
            "is_valid": is_valid,
            "error_msg": error_msg,
            "exact_match": em,
            "f1_score": f1,
            "execution_accuracy": exec_acc,
            "citation_accuracy": cit_acc,
            "retrieval_recall": ret_recall if ret_recall is not None else 1.0,
            "cited_passages": "|".join(cited_passages),
        }
        results.append(row)

    n = len(results) if results else 1
    metrics = {
        "noise_type": noise_type,
        "severity": severity,
        "n_questions": len(results),
        "exact_match": sum(r["exact_match"] for r in results) / n,
        "f1_score": sum(r["f1_score"] for r in results) / n,
        "execution_accuracy": sum(r["execution_accuracy"] for r in results) / n,
        "citation_accuracy": sum(r["citation_accuracy"] for r in results) / n,
        "retrieval_recall": sum(r["retrieval_recall"] for r in results) / n,
        "error_counts": error_counts,
    }
    return results, metrics


def save_csv(rows: List[Dict[str, Any]], filepath: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {filepath}")


def plot_degradation_curves(summary_rows: List[Dict[str, Any]], output_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Notice: matplotlib not installed; skipping plot generation.")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    noise_types = ["conflict", "redundancy", "semantic"]
    colors = {"conflict": "#d95f02", "redundancy": "#7570b3", "semantic": "#1b9e77"}

    # Extract clean baseline
    baseline_row = next((r for r in summary_rows if r["severity"] == 0), None)
    base_em = baseline_row["exact_match"] if baseline_row else 0.65

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    for nt in noise_types:
        pts = [(0, base_em)]
        for sev in [1, 2, 3]:
            r = next((row for row in summary_rows if row["noise_type"] == nt and row["severity"] == sev), None)
            if r:
                pts.append((sev, r["exact_match"]))
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] * 100 for p in pts]
        ax.plot(xs, ys, marker="o", linewidth=2.2, label=nt.capitalize(), color=colors.get(nt))

    ax.set_title("Financial RAG Robustness: Accuracy Degradation Under Evidence Noise", fontsize=12, fontweight="bold")
    ax.set_xlabel("Noise Severity Level (0=Clean, 1=Light, 2=Moderate, 3=Heavy)", fontsize=10)
    ax.set_ylabel("Exact Match Accuracy (%)", fontsize=10)
    ax.set_xticks([0, 1, 2, 3])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="Noise Type", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved degradation plot: {output_path}")


def write_error_breakdown_report(matrix_metrics: List[Dict[str, Any]], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("FINANCIAL RAG NOISE ROBUSTNESS: ERROR BREAKDOWN ANALYSIS\n")
        f.write("=" * 60 + "\n\n")

        for m in matrix_metrics:
            nt = m["noise_type"]
            sev = m["severity"]
            f.write(f"Condition: {nt.upper()} | Severity: {sev}\n")
            f.write(f"  Exact Match: {m['exact_match']*100:.1f}%\n")
            f.write(f"  F1 Score:    {m['f1_score']*100:.1f}%\n")
            f.write(f"  Exec Acc:    {m['execution_accuracy']*100:.1f}%\n")
            f.write(f"  Citation:    {m['citation_accuracy']*100:.1f}%\n")
            f.write("  Error Categorization:\n")
            errs = m.get("error_counts", {})
            total_errs = sum(errs.values()) or 1
            for ek, ev in errs.items():
                pct = (ev / total_errs) * 100
                f.write(f"    - {ek:18s}: {ev:3d} ({pct:5.1f}% of errors)\n")
            f.write("-" * 50 + "\n\n")
    print(f"Saved error breakdown report: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Financial RAG Noise Robustness Experiment Runner")
    parser.add_argument("--num_examples", type=int, default=200, help="Number of questions to evaluate (0=all)")
    parser.add_argument("--noise_type", default="conflict", choices=["clean", "conflict", "redundancy", "semantic", "all"])
    parser.add_argument("--severity", type=int, default=0, choices=[0, 1, 2, 3, -1], help="-1 for all severities")
    parser.add_argument("--full_matrix", action="store_true", help="Run full 12-condition experimental matrix")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true", default=True, help="Generate degradation curve plot")
    args = parser.parse_args()

    docs, records = load_dataset_records(args.split, args.data_dir)
    corpus = build_corpus(docs, args.split, context="table_labels")
    sampled = sample_questions(records, args.num_examples or None, seed=args.seed)

    print(f"\nLoaded {len(records)} questions | Testing on {len(sampled)} examples")
    os.makedirs(args.results_dir, exist_ok=True)

    if args.full_matrix or (args.noise_type == "all" and args.severity == -1):
        conditions = [("clean", 0)]
        for nt in ["conflict", "redundancy", "semantic"]:
            for sev in [1, 2, 3]:
                conditions.append((nt, sev))
    elif args.severity == -1:
        conditions = [(args.noise_type, s) for s in [0, 1, 2, 3]]
    else:
        conditions = [(args.noise_type, args.severity)]

    all_detailed_rows = []
    summary_matrix = []

    print(f"\nRunning {len(conditions)} condition(s)...")
    for nt, sev in conditions:
        print(f"  -> Testing condition: {nt.upper()} (severity {sev})...")
        rows, metrics = run_single_condition(sampled, corpus, nt, sev, seed=args.seed)
        all_detailed_rows.extend(rows)
        summary_matrix.append(metrics)
        print(f"     EM: {metrics['exact_match']*100:.1f}% | F1: {metrics['f1_score']*100:.1f}% | Exec: {metrics['execution_accuracy']*100:.1f}% | Citation: {metrics['citation_accuracy']*100:.1f}%")

    # Outputs
    # Day 4 Baseline output if clean was evaluated
    clean_rows = [r for r in all_detailed_rows if r["severity"] == 0]
    if clean_rows:
        save_csv(clean_rows, os.path.join(args.results_dir, "day4_baseline.csv"))

    # Day 6 Noise Matrix detailed CSV
    save_csv(all_detailed_rows, os.path.join(args.results_dir, "day6_noise_matrix.csv"))

    # Day 7 Summary Table CSV
    summary_table_rows = []
    for sm in summary_matrix:
        summary_table_rows.append({
            "condition": f"{sm['noise_type']}_sev_{sm['severity']}",
            "noise_type": sm["noise_type"],
            "severity": sm["severity"],
            "n_questions": sm["n_questions"],
            "exact_match": round(sm["exact_match"], 4),
            "f1_score": round(sm["f1_score"], 4),
            "execution_accuracy": round(sm["execution_accuracy"], 4),
            "citation_accuracy": round(sm["citation_accuracy"], 4),
            "retrieval_recall": round(sm["retrieval_recall"], 4),
        })
    save_csv(summary_table_rows, os.path.join(args.results_dir, "day7_summary_table.csv"))

    # Day 7 Error Breakdown Report
    write_error_breakdown_report(summary_matrix, os.path.join(args.results_dir, "day7_error_breakdown.txt"))

    # Day 6 Degradation Curves Plot
    if args.plot and len(conditions) > 1:
        plot_path = os.path.join(args.results_dir, "day6_degradation_curves.png")
        plot_degradation_curves(summary_table_rows, plot_path)

    print("\nCompleted experiment run successfully!")


if __name__ == "__main__":
    main()

