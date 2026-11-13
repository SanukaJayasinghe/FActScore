"""
Phase 2: Baseline Model Evaluation & Visualization
CM3604 Deep Learning - FActScore Fact-Checking Project
"""

import os
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from factscore_retrieval_interface import read_passages, read_fact_examples
from factcheck import AlwaysEntailedFactChecker, RandomGuessFactChecker, FactExample


def evaluate_model(examples, fact_checker):
    gold_indexer = ["S", "NS"]
    cm = [[0, 0], [0, 0]]  # [gold, pred] -> [[S->S, S->NS], [NS->S, NS->NS]]

    for ex in examples:
        converted_label = "NS" if ex.label == "IR" else ex.label
        gold_idx = gold_indexer.index(converted_label)
        pred_label = fact_checker.predict(ex.fact, ex.passages)
        pred_idx = gold_indexer.index(pred_label)
        cm[gold_idx][pred_idx] += 1

    total = sum(sum(row) for row in cm)
    correct = cm[0][0] + cm[1][1]
    accuracy = correct / total

    # Class S (idx 0)
    num_correct_S = cm[0][0]
    num_gold_S = cm[0][0] + cm[0][1]
    num_pred_S = cm[0][0] + cm[1][0]
    prec_S = num_correct_S / num_pred_S if num_pred_S > 0 else 0.0
    rec_S = num_correct_S / num_gold_S if num_gold_S > 0 else 0.0
    f1_S = (2 * prec_S * rec_S / (prec_S + rec_S)) if (prec_S + rec_S) > 0 else 0.0

    # Class NS (idx 1)
    num_correct_NS = cm[1][1]
    num_gold_NS = cm[1][0] + cm[1][1]
    num_pred_NS = cm[0][1] + cm[1][1]
    prec_NS = num_correct_NS / num_pred_NS if num_pred_NS > 0 else 0.0
    rec_NS = num_correct_NS / num_gold_NS if num_gold_NS > 0 else 0.0
    f1_NS = (2 * prec_NS * rec_NS / (prec_NS + rec_NS)) if (prec_NS + rec_NS) > 0 else 0.0

    macro_f1 = (f1_S + f1_NS) / 2.0

    return {
        "confusion_matrix": cm,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "prec_S": prec_S,
        "rec_S": rec_S,
        "f1_S": f1_S,
        "prec_NS": prec_NS,
        "rec_NS": rec_NS,
        "f1_NS": f1_NS,
        "macro_f1": macro_f1
    }


def plot_confusion_matrices(results_dict, output_path="results/baseline_confusion_matrices.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    for ax, (model_name, res) in zip(axes, results_dict.items()):
        cm = np.array(res["confusion_matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

        # Labels
        ax.set_title(f"{model_name}\n(Accuracy: {res['accuracy']*100:.2f}%)", fontsize=12, fontweight="bold", pad=12)
        classes = ["Supported (S)", "Not Supported (NS)"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=10)

        # Annotations inside cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, f"{cm[i, j]}",
                        ha="center", va="center",
                        color=color, fontsize=14, fontweight="bold")

        ax.set_ylabel("True Gold Label", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved baseline confusion matrices to {output_path}")


def plot_metrics_comparison(results_dict, output_path="results/baseline_metrics_comparison.png"):
    metrics = ["accuracy", "f1_S", "f1_NS", "macro_f1"]
    metric_labels = ["Accuracy", "F1 (Supported)", "F1 (Not Supported)", "Macro F1"]

    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    colors = ["#1f77b4", "#ff7f0e"]
    for i, (model_name, res) in enumerate(results_dict.items()):
        values = [res[m] * 100 for m in metrics]
        offset = (i - 0.5) * width
        rects = ax.bar(x + offset, values, width, label=model_name, color=colors[i], edgecolor="black", alpha=0.85)

        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("Baseline Performance Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Chance Level (50%)")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved baseline metrics comparison plot to {output_path}")


def save_metrics_csv(results_dict, output_path="results/baseline_metrics.csv"):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "Accuracy (%)", "Prec_S (%)", "Rec_S (%)", "F1_S (%)",
            "Prec_NS (%)", "Rec_NS (%)", "F1_NS (%)", "Macro_F1 (%)",
            "TP (S->S)", "FN (S->NS)", "FP (NS->S)", "TN (NS->NS)"
        ])
        for model_name, res in results_dict.items():
            cm = res["confusion_matrix"]
            writer.writerow([
                model_name,
                f"{res['accuracy']*100:.2f}",
                f"{res['prec_S']*100:.2f}",
                f"{res['rec_S']*100:.2f}",
                f"{res['f1_S']*100:.2f}",
                f"{res['prec_NS']*100:.2f}",
                f"{res['rec_NS']*100:.2f}",
                f"{res['f1_NS']*100:.2f}",
                f"{res['macro_f1']*100:.2f}",
                cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            ])
    print(f"Saved baseline metrics table to {output_path}")


def main():
    passages_dict = read_passages("data/passages_bm25_ChatGPT_humfacts.jsonl")
    examples = read_fact_examples("data/dev_labeled_ChatGPT.jsonl", passages_dict)

    print(f"Evaluating baselines on {len(examples)} examples...")

    # 1. Always Entailed
    always_checker = AlwaysEntailedFactChecker()
    always_res = evaluate_model(examples, always_checker)

    # 2. Random Guess (fixed seed 42 for exact reproducibility)
    np.random.seed(42)
    random_checker = RandomGuessFactChecker()
    random_res = evaluate_model(examples, random_checker)

    results_dict = {
        "Always Supported": always_res,
        "Random Guess": random_res
    }

    print("\n" + "=" * 75)
    print("                PHASE 2: BASELINE EVALUATION RESULTS")
    print("=" * 75)
    for name, res in results_dict.items():
        cm = res["confusion_matrix"]
        print(f"\n[{name.upper()}]")
        print(f"  Confusion Matrix:  [[TP={cm[0][0]:3d}, FN={cm[0][1]:3d}], [FP={cm[1][0]:3d}, TN={cm[1][1]:3d}]]")
        print(f"  Accuracy:          {res['correct']}/{res['total']} = {res['accuracy']*100:.2f}%")
        print(f"  Supported (S):     Prec={res['prec_S']*100:.2f}% | Rec={res['rec_S']*100:.2f}% | F1={res['f1_S']*100:.2f}%")
        print(f"  Not Supported (NS): Prec={res['prec_NS']*100:.2f}% | Rec={res['rec_NS']*100:.2f}% | F1={res['f1_NS']*100:.2f}%")
        print(f"  Macro F1:          {res['macro_f1']*100:.2f}%")
    print("=" * 75 + "\n")

    # Save CSV and Plots
    save_metrics_csv(results_dict)
    plot_confusion_matrices(results_dict)
    plot_metrics_comparison(results_dict)


if __name__ == "__main__":
    main()

