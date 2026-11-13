"""
Phase 3: Task 1 - Word Overlap Systematic Ablations & Visualizations
CM3604 Deep Learning - FActScore Fact-Checking Project
"""

import os
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import re
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from factscore_retrieval_interface import read_passages, read_fact_examples
from factcheck import WordRecallThresholdFactChecker


def clean_wiki_tags(text: str) -> str:
    return re.sub(r'</?s>', ' ', text)


def get_tokens(text: str, remove_stops: bool, do_stem: bool, stemmer, stop_words):
    cleaned = clean_wiki_tags(text).lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', cleaned)
    if remove_stops:
        tokens = [t for t in tokens if t not in stop_words]
    if do_stem:
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens


def evaluate_threshold_sweep(scores, golds):
    best_acc = 0.0
    best_tau = 0.0
    best_metrics = {}

    thresholds = np.linspace(0.0, 1.0, 101)
    acc_curve = []

    for tau in thresholds:
        preds = ["S" if s >= tau else "NS" for s in scores]
        correct = sum(1 for p, g in zip(preds, golds) if p == g)
        acc = correct / len(golds)
        acc_curve.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_tau = tau
            tp = sum(1 for p, g in zip(preds, golds) if p == "S" and g == "S")
            fp = sum(1 for p, g in zip(preds, golds) if p == "S" and g == "NS")
            fn = sum(1 for p, g in zip(preds, golds) if p == "NS" and g == "S")
            tn = sum(1 for p, g in zip(preds, golds) if p == "NS" and g == "NS")

            p_S = tp / (tp + fp) if (tp + fp) else 0.0
            r_S = tp / (tp + fn) if (tp + fn) else 0.0
            f1_S = 2 * p_S * r_S / (p_S + r_S) if (p_S + r_S) else 0.0

            p_NS = tn / (tn + fn) if (tn + fn) else 0.0
            r_NS = tn / (tn + fp) if (tn + fp) else 0.0
            f1_NS = 2 * p_NS * r_NS / (p_NS + r_NS) if (p_NS + r_NS) else 0.0

            best_metrics = {
                "accuracy": acc,
                "best_tau": tau,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "prec_S": p_S, "rec_S": r_S, "f1_S": f1_S,
                "prec_NS": p_NS, "rec_NS": r_NS, "f1_NS": f1_NS,
                "macro_f1": (f1_S + f1_NS) / 2.0
            }

    return best_metrics, thresholds, acc_curve


def run_ablations(examples):
    stemmer = PorterStemmer()
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        nltk.download("stopwords", quiet=True)
        stop_words = set(stopwords.words("english"))

    golds = ["NS" if ex.label == "IR" else ex.label for ex in examples]

    configurations = [
        ("Raw Tokens (No stops, No stem)", False, False, "word_recall_max"),
        ("Stopwords Removed, No stem", True, False, "word_recall_max"),
        ("Stopwords Kept, Porter Stem", False, True, "word_recall_max"),
        ("No Stops, Porter Stem (Max Passage - CHOSEN)", True, True, "word_recall_max"),
        ("No Stops, Porter Stem (Concat Passages)", True, True, "word_recall_concat"),
        ("No Stops, Porter Stem (Jaccard Similarity)", True, True, "jaccard_max"),
        ("No Stops, Porter Stem (Dice Coefficient)", True, True, "dice_max"),
    ]

    ablation_results = []
    curves_dict = {}

    for name, rem_stops, do_stem, metric in configurations:
        scores = []
        for ex in examples:
            f_tokens = set(get_tokens(ex.fact, rem_stops, do_stem, stemmer, stop_words))
            if not f_tokens:
                scores.append(0.0)
                continue

            p_tokens_list = [set(get_tokens(p["text"], rem_stops, do_stem, stemmer, stop_words)) for p in ex.passages]

            if metric == "word_recall_max":
                sc = max((len(f_tokens & p_toks) / len(f_tokens) for p_toks in p_tokens_list if p_toks), default=0.0)
            elif metric == "word_recall_concat":
                p_cat = set().union(*p_tokens_list) if p_tokens_list else set()
                sc = len(f_tokens & p_cat) / len(f_tokens) if f_tokens else 0.0
            elif metric == "jaccard_max":
                sc = max((len(f_tokens & p_toks) / len(f_tokens | p_toks) for p_toks in p_tokens_list if (f_tokens | p_toks)), default=0.0)
            elif metric == "dice_max":
                sc = max((2 * len(f_tokens & p_toks) / (len(f_tokens) + len(p_toks)) for p_toks in p_tokens_list if (f_tokens or p_toks)), default=0.0)
            else:
                sc = 0.0

            scores.append(sc)

        best_m, ths, curve = evaluate_threshold_sweep(scores, golds)
        best_m["name"] = name
        ablation_results.append(best_m)
        curves_dict[name] = (ths, curve, best_m["best_tau"], best_m["accuracy"])

    return ablation_results, curves_dict


def plot_threshold_curves(curves_dict, output_path="results/word_overlap_threshold_curve.png"):
    plt.figure(figsize=(10, 6), dpi=300)

    styles = [
        ("#7f7f7f", "--", 1.2),
        ("#1f77b4", "-.", 1.4),
        ("#9467bd", ":", 1.4),
        ("#2ca02c", "-", 2.5),  # Highlight chosen
        ("#d62728", "--", 1.4),
        ("#ff7f0e", "-.", 1.4),
        ("#8c564b", ":", 1.4)
    ]

    for (name, (ths, curve, best_tau, best_acc)), (color, ls, lw) in zip(curves_dict.items(), styles):
        label = f"{name} (Peak: {best_acc*100:.1f}% @ tau={best_tau:.2f})"
        plt.plot(ths, [c * 100 for c in curve], label=label, color=color, linestyle=ls, linewidth=lw)

    plt.axhline(75.0, color="black", linestyle="--", linewidth=1.0, label="Coursework Target (75.0%)")
    plt.axhline(50.68, color="gray", linestyle=":", linewidth=0.8, label="Baseline Floor (50.68%)")

    plt.title("Task 1: Word Overlap Threshold Optimization & Ablation Curves", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Classification Decision Threshold (tau)", fontsize=11, fontweight="bold")
    plt.ylabel("Dataset Accuracy (%)", fontsize=11, fontweight="bold")
    plt.ylim(45, 90)
    plt.xlim(0.0, 1.0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=9, loc="lower left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved threshold tuning curves to {output_path}")


def plot_confusion_matrix(cm, accuracy, output_path="results/word_overlap_confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    classes = ["Supported (S)", "Not Supported (NS)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    color=color, fontsize=15, fontweight="bold")

    ax.set_ylabel("True Gold Label", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_title(f"Task 1: Word Overlap Confusion Matrix\n(Accuracy: {accuracy*100:.2f}%)",
                 fontsize=12, fontweight="bold", pad=12)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix heatmap to {output_path}")


def save_ablation_csv(ablation_results, output_path="results/word_overlap_ablation.csv"):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Ablation Configuration", "Optimal Threshold (tau)", "Accuracy (%)",
            "Precision_S (%)", "Recall_S (%)", "F1_S (%)",
            "Precision_NS (%)", "Recall_NS (%)", "F1_NS (%)", "Macro_F1 (%)",
            "TP", "FP", "FN", "TN"
        ])
        for r in ablation_results:
            writer.writerow([
                r["name"],
                f"{r['best_tau']:.2f}",
                f"{r['accuracy']*100:.2f}",
                f"{r['prec_S']*100:.2f}",
                f"{r['rec_S']*100:.2f}",
                f"{r['f1_S']*100:.2f}",
                f"{r['prec_NS']*100:.2f}",
                f"{r['rec_NS']*100:.2f}",
                f"{r['f1_NS']*100:.2f}",
                f"{r['macro_f1']*100:.2f}",
                r["tp"], r["fp"], r["fn"], r["tn"]
            ])
    print(f"Saved ablation comparison table to {output_path}")


def save_predictions_csv(examples, fact_checker, output_path="results/word_overlap_predictions.csv"):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Fact", "GoldLabel", "PredictedLabel", "Correct", "MaxRecallScore"])
        for idx, ex in enumerate(examples, 1):
            gold = "NS" if ex.label == "IR" else ex.label
            pred = fact_checker.predict(ex.fact, ex.passages)
            f_tokens = set(fact_checker.tokenize(ex.fact))
            max_rec = max((fact_checker.compute_recall(f_tokens, p["text"]) for p in ex.passages), default=0.0)
            writer.writerow([idx, ex.fact, gold, pred, "YES" if pred == gold else "NO", f"{max_rec:.4f}"])
    print(f"Saved fact-level predictions to {output_path}")


def main():
    passages_dict = read_passages("data/passages_bm25_ChatGPT_humfacts.jsonl")
    examples = read_fact_examples("data/dev_labeled_ChatGPT.jsonl", passages_dict)

    print("Running Task 1 systematic ablations...")
    ablation_results, curves_dict = run_ablations(examples)

    print("\n" + "=" * 95)
    print("                 PHASE 3: TASK 1 WORD OVERLAP ABLATION RESULTS")
    print("=" * 95)
    print(f"{'Configuration':<45} | {'tau':<4} | {'Accuracy':<8} | {'F1 (S)':<8} | {'F1 (NS)':<8} | {'Macro F1':<8}")
    print("-" * 95)
    for r in ablation_results:
        print(f"{r['name']:<45} | {r['best_tau']:<4.2f} | {r['accuracy']*100:<7.2f}% | {r['f1_S']*100:<7.2f}% | {r['f1_NS']*100:<7.2f}% | {r['macro_f1']*100:<7.2f}%")
    print("=" * 95 + "\n")

    # Generate Evidence Files
    save_ablation_csv(ablation_results)
    plot_threshold_curves(curves_dict)

    # Evaluate Chosen Model
    chosen_checker = WordRecallThresholdFactChecker(threshold=0.65)
    save_predictions_csv(examples, chosen_checker)

    # Plot Confusion Matrix of chosen model
    best_config = [r for r in ablation_results if "CHOSEN" in r["name"]][0]
    cm = np.array([[best_config["tp"], best_config["fn"]], [best_config["fp"], best_config["tn"]]])
    plot_confusion_matrix(cm, best_config["accuracy"])


if __name__ == "__main__":
    main()

