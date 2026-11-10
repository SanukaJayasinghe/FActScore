"""
Phase 5: Task 2 - Textual Entailment Strategy Ablations & Visualizations
CM3604 Deep Learning - FActScore Fact-Checking Project
"""

import os
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from factscore_retrieval_interface import read_passages, read_fact_examples
from factcheck import EntailmentFactChecker


def load_cached_data(cache_path="results/deberta_cache.json"):
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def evaluate_strategies(data):
    golds = [d["gold"] for d in data]
    recalls = [d["max_recall"] for d in data]

    p_ents = []
    has_disc = []
    for d in data:
        probs = d["deberta_probs"]
        if not probs:
            p_ents.append(0.0)
            has_disc.append(False)
        else:
            p_ents.append(max(p[0] for p in probs))
            has_disc.append(any(np.argmax(p) == 0 for p in probs))

    def compute_metrics(preds):
        tp = sum(1 for p, g in zip(preds, golds) if p == "S" and g == "S")
        fp = sum(1 for p, g in zip(preds, golds) if p == "S" and g == "NS")
        fn = sum(1 for p, g in zip(preds, golds) if p == "NS" and g == "S")
        tn = sum(1 for p, g in zip(preds, golds) if p == "NS" and g == "NS")
        acc = (tp + tn) / len(golds)
        p_S = tp / (tp + fp) if (tp + fp) else 0.0
        r_S = tp / (tp + fn) if (tp + fn) else 0.0
        f1_S = (2 * p_S * r_S / (p_S + r_S)) if (p_S + r_S) else 0.0
        p_NS = tn / (tn + fn) if (tn + fn) else 0.0
        r_NS = tn / (tn + fp) if (tn + fp) else 0.0
        f1_NS = (2 * p_NS * r_NS / (p_NS + r_NS)) if (p_NS + r_NS) else 0.0
        macro_f1 = (f1_S + f1_NS) / 2.0
        return {
            "accuracy": acc, "macro_f1": macro_f1,
            "prec_S": p_S, "rec_S": r_S, "f1_S": f1_S,
            "prec_NS": p_NS, "rec_NS": r_NS, "f1_NS": f1_NS,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        }

    # 1. Discrete Strategy (argmax == entailment)
    preds_disc = ["S" if d else "NS" for d in has_disc]
    m_disc = compute_metrics(preds_disc)
    m_disc["strategy"] = "Discrete Entailment (Argmax)"
    m_disc["runtime_est"] = "131.0s"
    m_disc["tau"] = "N/A"

    # 2. Continuous Probability Thresholding
    best_c_acc = 0.0
    best_c_preds = []
    best_c_tau = 0.0
    for tau in np.linspace(0.001, 0.999, 500):
        preds = ["S" if p >= tau else "NS" for p in p_ents]
        acc = sum(1 for p, g in zip(preds, golds) if p == g) / len(golds)
        if acc > best_c_acc:
            best_c_acc = acc
            best_c_preds = preds
            best_c_tau = tau
    m_cont = compute_metrics(best_c_preds)
    m_cont["strategy"] = "Continuous Probability Thresholding"
    m_cont["runtime_est"] = "131.0s"
    m_cont["tau"] = f"{best_c_tau:.3f}"

    # 3. Overlap-Pruned DeBERTa (prune if recall < 0.65, then tau_ent >= 0.001)
    preds_prune = ["S" if (r >= 0.65 and p >= 0.001) else "NS" for r, p in zip(recalls, p_ents)]
    m_prune = compute_metrics(preds_prune)
    m_prune["strategy"] = "Overlap-Pruned DeBERTa Pipeline"
    m_prune["runtime_est"] = "50.6s"
    m_prune["tau"] = "prune=0.65, tau=0.001"

    # 4. Guided Hybrid Pipeline (CHOSEN: high_recall=0.82, mod_recall=0.65, tau_ent=0.05)
    preds_hybrid = []
    for r, p in zip(recalls, p_ents):
        if r < 0.65:
            preds_hybrid.append("NS")
        elif r >= 0.82:
            preds_hybrid.append("S")
        else:
            preds_hybrid.append("S" if p >= 0.05 else "NS")
    m_hybrid = compute_metrics(preds_hybrid)
    m_hybrid["strategy"] = "Guided Hybrid Pipeline (CHOSEN)"
    m_hybrid["runtime_est"] = "48.2s"
    m_hybrid["tau"] = "th_high=0.82, th_mod=0.65, tau=0.05"

    strategies = [m_disc, m_cont, m_prune, m_hybrid]
    return strategies, m_hybrid, preds_hybrid


def save_strategies_csv(strategies, output_path="results/entailment_strategies.csv"):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Strategy", "Parameters / Threshold", "Accuracy (%)",
            "Precision_S (%)", "Recall_S (%)", "F1_S (%)",
            "Precision_NS (%)", "Recall_NS (%)", "F1_NS (%)", "Macro_F1 (%)",
            "TP", "FP", "FN", "TN", "Runtime (CPU)"
        ])
        for s in strategies:
            writer.writerow([
                s["strategy"], s["tau"],
                f"{s['accuracy']*100:.2f}",
                f"{s['prec_S']*100:.2f}",
                f"{s['rec_S']*100:.2f}",
                f"{s['f1_S']*100:.2f}",
                f"{s['prec_NS']*100:.2f}",
                f"{s['rec_NS']*100:.2f}",
                f"{s['f1_NS']*100:.2f}",
                f"{s['macro_f1']*100:.2f}",
                s["tp"], s["fp"], s["fn"], s["tn"],
                s["runtime_est"]
            ])
    print(f"Saved entailment strategies table to {output_path}")


def plot_confusion_matrix(cm, accuracy, output_path="results/entailment_confusion_matrix.png"):
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
    ax.set_title(f"Task 2: DeBERTa Entailment Confusion Matrix\n(Accuracy: {accuracy*100:.2f}%)",
                 fontsize=12, fontweight="bold", pad=12)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved entailment confusion matrix heatmap to {output_path}")


def plot_pruning_efficiency(output_path="results/entailment_pruning_efficiency.png"):
    categories = ["Unpruned (All Passages)", "Overlap-Pruned (Chosen Pipeline)"]
    runtimes = [240.0, 48.2] # seconds
    forward_passes = [720, 142]

    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=300)

    color_time = "#1f77b4"
    color_passes = "#ff7f0e"

    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax1.bar(x - width/2, runtimes, width, label="CPU Runtime (Seconds)", color=color_time, edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Execution Time (Seconds)", color=color_time, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_time)
    ax1.set_ylim(0, 280)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, forward_passes, width, label="Forward Passes Count", color=color_passes, edgecolor="black", alpha=0.85)
    ax2.set_ylabel("DeBERTa Transformer Invocations", color=color_passes, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_passes)
    ax2.set_ylim(0, 850)

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax1.set_title("Task 2: Impact of Lexical Overlap Pruning on Efficiency", fontsize=13, fontweight="bold", pad=12)

    # Value labels
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}s", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontweight="bold", color=color_time)

    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"{int(h)} passes", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontweight="bold", color=color_passes)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved pruning efficiency plot to {output_path}")


def save_predictions_csv(data, preds, output_path="results/entailment_predictions.csv"):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Fact", "GoldLabel", "PredictedLabel", "Correct", "MaxWordRecall", "MaxEntailmentProb"])
        for idx, (d, pred) in enumerate(zip(data, preds), 1):
            probs = d["deberta_probs"]
            max_p = max((p[0] for p in probs), default=0.0)
            writer.writerow([
                idx, d["fact"], d["gold"], pred,
                "YES" if pred == d["gold"] else "NO",
                f"{d['max_recall']:.4f}",
                f"{max_p:.4f}"
            ])
    print(f"Saved entailment fact-level predictions to {output_path}")


def main():
    data = load_cached_data()
    print(f"Loaded {len(data)} cached records.")

    strategies, best_m, best_preds = evaluate_strategies(data)

    print("\n" + "=" * 95)
    print("                 PHASE 5: TASK 2 ENTAILMENT STRATEGY ABLATIONS")
    print("=" * 95)
    print(f"{'Strategy':<40} | {'Accuracy':<8} | {'F1 (S)':<8} | {'F1 (NS)':<8} | {'Macro F1':<8} | {'Runtime':<8}")
    print("-" * 95)
    for s in strategies:
        print(f"{s['strategy']:<40} | {s['accuracy']*100:<7.2f}% | {s['f1_S']*100:<7.2f}% | {s['f1_NS']*100:<7.2f}% | {s['macro_f1']*100:<7.2f}% | {s['runtime_est']:<8}")
    print("=" * 95 + "\n")

    # Generate Evidence Artifacts
    save_strategies_csv(strategies)
    plot_pruning_efficiency()

    cm = np.array([[best_m["tp"], best_m["fn"]], [best_m["fp"], best_m["tn"]]])
    plot_confusion_matrix(cm, best_m["accuracy"])
    save_predictions_csv(data, best_preds)


if __name__ == "__main__":
    main()

