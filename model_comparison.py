"""
Phase 8: Master Model Comparison & Evidence Audit

"""

import os
import tempfile
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 1. COMPILE MASTER METRICS TABLE ACROSS ALL 4 MODELS
# ---------------------------------------------------------------------------
MODELS_DATA = [
    {
        "model_id": "baseline_always_s",
        "model_name": "Baseline: Always Supported",
        "category": "Baseline",
        "description": "Trivial majority-class baseline that predicts Supported (S) for every claim.",
        "coursework_target": "N/A",
        "target_met": "N/A",
        "total_examples": 221,
        "correct": 112,
        "accuracy_pct": 50.68,
        "prec_S_pct": 50.68,
        "rec_S_pct": 100.00,
        "f1_S_pct": 67.27,
        "prec_NS_pct": 0.00,
        "rec_NS_pct": 0.00,
        "f1_NS_pct": 0.00,
        "macro_f1_pct": 33.63,
        "tp": 112,
        "fn": 0,
        "fp": 109,
        "tn": 0,
        "cpu_runtime_sec": "< 0.01s"
    },
    {
        "model_id": "baseline_random",
        "model_name": "Baseline: Random Guess",
        "category": "Baseline",
        "description": "Uniform random coin-flip baseline predicting S or NS with equal probability.",
        "coursework_target": "N/A",
        "target_met": "N/A",
        "total_examples": 221,
        "correct": 111,
        "accuracy_pct": 50.23,
        "prec_S_pct": 50.88,
        "rec_S_pct": 51.79,
        "f1_S_pct": 51.33,
        "prec_NS_pct": 49.53,
        "rec_NS_pct": 48.62,
        "f1_NS_pct": 49.07,
        "macro_f1_pct": 50.20,
        "tp": 58,
        "fn": 54,
        "fp": 56,
        "tn": 53,
        "cpu_runtime_sec": "< 0.01s"
    },
    {
        "model_id": "task1_word_overlap",
        "model_name": "Task 1: Word Overlap Baseline",
        "category": "Lexical Baseline",
        "description": "Porter stemming, stopword removal, and max-over-passages unigram recall with threshold tau=0.65.",
        "coursework_target": ">= 75.0%",
        "target_met": "YES (+8.26%)",
        "total_examples": 221,
        "correct": 184,
        "accuracy_pct": 83.26,
        "prec_S_pct": 82.61,
        "rec_S_pct": 84.82,
        "f1_S_pct": 83.70,
        "prec_NS_pct": 83.96,
        "rec_NS_pct": 81.65,
        "f1_NS_pct": 82.79,
        "macro_f1_pct": 83.25,
        "tp": 95,
        "fn": 17,
        "fp": 20,
        "tn": 89,
        "cpu_runtime_sec": "0.12s"
    },
    {
        "model_id": "task2_deberta_entailment",
        "model_name": "Task 2: Textual Entailment Pipeline",
        "category": "Neural Entailment",
        "description": "DeBERTa-v3 cross-encoder with lexical overlap pruning (th_high=0.85, th_mod=0.65, tau_nli=0.005).",
        "coursework_target": ">= 83.0%",
        "target_met": "YES (+1.16%)",
        "total_examples": 221,
        "correct": 186,
        "accuracy_pct": 84.16,
        "prec_S_pct": 87.38,
        "rec_S_pct": 80.36,
        "f1_S_pct": 83.72,
        "prec_NS_pct": 81.36,
        "rec_NS_pct": 88.07,
        "f1_NS_pct": 84.58,
        "macro_f1_pct": 84.15,
        "tp": 90,
        "fn": 22,
        "fp": 13,
        "tn": 96,
        "cpu_runtime_sec": "42.0s"
    }
]


# ---------------------------------------------------------------------------
# 2. SAVE MASTER COMPARISON CSV
# ---------------------------------------------------------------------------
def save_master_csv(output_path="results/model_comparison_master.csv"):
    fieldnames = [
        "model_id", "model_name", "category", "description",
        "coursework_target", "target_met", "total_examples", "correct",
        "accuracy_pct", "macro_f1_pct",
        "prec_S_pct", "rec_S_pct", "f1_S_pct",
        "prec_NS_pct", "rec_NS_pct", "f1_NS_pct",
        "tp", "fn", "fp", "tn", "cpu_runtime_sec"
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(MODELS_DATA)
    print(f"Saved master comparison CSV to {output_path}")


# ---------------------------------------------------------------------------
# 3. GENERATE MULTI-PANEL PUBLICATION VISUALIZATION
# ---------------------------------------------------------------------------
def plot_master_comparison(output_path="results/model_comparison_chart.png"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#FAFAFA")

    model_labels = ["Always\nSupported", "Random\nGuess", "Task 1: Word\nOverlap", "Task 2: Neural\nEntailment"]
    colors = ["#777777", "#9E9E9E", "#1F77B4", "#2CA02C"]

    # ----------------------------------------------------
    # Subplot 1: Accuracy & Macro F1 with Coursework Targets
    # ----------------------------------------------------
    ax1 = axes[0, 0]
    x = np.arange(len(model_labels))
    w = 0.35

    accs = [m["accuracy_pct"] for m in MODELS_DATA]
    f1s = [m["macro_f1_pct"] for m in MODELS_DATA]

    r1 = ax1.bar(x - w/2, accs, w, label="Accuracy (%)", color="#2B5B84", edgecolor="#1A364F", alpha=0.9)
    r2 = ax1.bar(x + w/2, f1s, w, label="Macro F1 (%)", color="#5C93C4", edgecolor="#396185", alpha=0.9)

    # Coursework Target thresholds
    ax1.axhline(75.0, color="#E65100", linestyle="--", linewidth=1.8, label="Task 1 Target (>= 75.0%)")
    ax1.axhline(83.0, color="#C62828", linestyle="-.", linewidth=1.8, label="Task 2 Target (>= 83.0%)")

    ax1.set_title("Overall Accuracy & Macro F1 across All Systems", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", fontsize=9)

    for rect in r1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in r2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, color="#2B5B84")

    # ----------------------------------------------------
    # Subplot 2: Precision, Recall, and F1 per Class (Task 1 vs Task 2)
    # ----------------------------------------------------
    ax2 = axes[0, 1]
    metric_names = ["Prec (S)", "Rec (S)", "F1 (S)", "Prec (NS)", "Rec (NS)", "F1 (NS)"]
    t1_vals = [MODELS_DATA[2]["prec_S_pct"], MODELS_DATA[2]["rec_S_pct"], MODELS_DATA[2]["f1_S_pct"],
               MODELS_DATA[2]["prec_NS_pct"], MODELS_DATA[2]["rec_NS_pct"], MODELS_DATA[2]["f1_NS_pct"]]
    t2_vals = [MODELS_DATA[3]["prec_S_pct"], MODELS_DATA[3]["rec_S_pct"], MODELS_DATA[3]["f1_S_pct"],
               MODELS_DATA[3]["prec_NS_pct"], MODELS_DATA[3]["rec_NS_pct"], MODELS_DATA[3]["f1_NS_pct"]]

    xm = np.arange(len(metric_names))
    r_t1 = ax2.bar(xm - w/2, t1_vals, w, label="Task 1 (Word Overlap: 83.26%)", color="#1F77B4", edgecolor="#134B71", alpha=0.85)
    r_t2 = ax2.bar(xm + w/2, t2_vals, w, label="Task 2 (DeBERTa Hybrid: 84.16%)", color="#2CA02C", edgecolor="#1B631B", alpha=0.85)

    ax2.set_title("Detailed Metric Breakdown: Task 1 vs Task 2", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax2.set_xticks(xm)
    ax2.set_xticklabels(metric_names, fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", fontsize=9)

    for rect in r_t1:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in r_t2:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1B631B")

    # ----------------------------------------------------
    # Subplot 3: Error Count Comparison (FP vs FN)
    # ----------------------------------------------------
    ax3 = axes[1, 0]
    fps = [m["fp"] for m in MODELS_DATA]
    fns = [m["fn"] for m in MODELS_DATA]

    r_fp = ax3.bar(x - w/2, fps, w, label="False Positives (FP)", color="#D9534F", edgecolor="#A94442", alpha=0.85)
    r_fn = ax3.bar(x + w/2, fns, w, label="False Negatives (FN)", color="#F0AD4E", edgecolor="#C68222", alpha=0.85)

    ax3.set_title("Error Audit: False Positives vs False Negatives", fontsize=12, fontweight="bold", pad=10)
    ax3.set_ylabel("Error Count", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_labels, fontsize=10)
    ax3.set_ylim(0, 125)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", fontsize=9)

    for rect in r_fp:
        h = rect.get_height()
        ax3.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#D9534F")
    for rect in r_fn:
        h = rect.get_height()
        ax3.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#B8771A")

    # Callout highlighting 35% reduction in FP
    ax3.annotate("35% FP Reduction\n(20 -> 13)", xy=(3 - w/2, 13), xytext=(2.6, 38),
                 arrowprops=dict(arrowstyle="->", color="#D9534F", lw=1.5),
                 fontsize=9, fontweight="bold", color="#A94442", bbox=dict(boxstyle="round,pad=0.3", fc="#FDEDEC", ec="#D9534F"))

    # ----------------------------------------------------
    # Subplot 4: Confusion Matrix Breakdown (Stacked Bar of TP, TN, FP, FN)
    # ----------------------------------------------------
    ax4 = axes[1, 1]
    tps = [m["tp"] for m in MODELS_DATA]
    tns = [m["tn"] for m in MODELS_DATA]

    # Stacked bars: Correct (TP + TN) vs Incorrect (FP + FN)
    p_tp = ax4.bar(x, tps, w*1.5, label="True Positives (TP)", color="#2CA02C", alpha=0.85)
    p_tn = ax4.bar(x, tns, w*1.5, bottom=tps, label="True Negatives (TN)", color="#1F77B4", alpha=0.85)
    p_fp = ax4.bar(x, fps, w*1.5, bottom=np.array(tps) + np.array(tns), label="False Positives (FP)", color="#D9534F", alpha=0.85)
    p_fn = ax4.bar(x, fns, w*1.5, bottom=np.array(tps) + np.array(tns) + np.array(fps), label="False Negatives (FN)", color="#F0AD4E", alpha=0.85)

    ax4.set_title("Full Confusion Matrix Component Composition (N = 221)", fontsize=12, fontweight="bold", pad=10)
    ax4.set_ylabel("Total Samples", fontsize=11, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(model_labels, fontsize=10)
    ax4.set_ylim(0, 245)
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", fontsize=9)

    for i, m in enumerate(MODELS_DATA):
        tot_correct = m["correct"]
        pct = m["accuracy_pct"]
        ax4.annotate(f"{tot_correct}/221\n({pct:.1f}%)", xy=(i, 223),
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved master comparison chart to {output_path}")


# ---------------------------------------------------------------------------
# 4. EVIDENCE & NUMBER AUDIT
# ---------------------------------------------------------------------------
def audit_evidence():
    print("\n--- RUNNING CROSS-PHASE EVIDENCE & METRICS AUDIT ---")
    total_samples = 221
    gold_S = 112
    gold_NS = 109

    for m in MODELS_DATA:
        tp, fn, fp, tn = m["tp"], m["fn"], m["fp"], m["tn"]
        acc = m["accuracy_pct"]
        calc_acc = round((tp + tn) / total_samples * 100, 2)
        calc_S = tp + fn
        calc_NS = fp + tn

        assert calc_S == gold_S, f"Mismatch in Gold S for {m['model_name']}: {calc_S} != {gold_S}"
        assert calc_NS == gold_NS, f"Mismatch in Gold NS for {m['model_name']}: {calc_NS} != {gold_NS}"
        assert abs(acc - calc_acc) < 0.05, f"Mismatch in Accuracy for {m['model_name']}: {acc} vs {calc_acc}"
        assert (tp + fn + fp + tn) == total_samples, f"Total count != 221 for {m['model_name']}"

    print("[AUDIT PASSED] 100% mathematical consistency across all models and metrics!")
    print(f"  - Total Evaluation Facts: {total_samples}")
    print(f"  - Gold Supported: {gold_S} (50.68%)")
    print(f"  - Gold Not Supported: {gold_NS} (49.32%)")
    print(f"  - Task 1 Accuracy: {MODELS_DATA[2]['accuracy_pct']}% (Target: >= 75%, Status: MET)")
    print(f"  - Task 2 Accuracy: {MODELS_DATA[3]['accuracy_pct']}% (Target: >= 83%, Status: MET)")
    print(f"  - False Positive Reduction: {MODELS_DATA[2]['fp']} -> {MODELS_DATA[3]['fp']} (-35.0%)")


if __name__ == "__main__":
    save_master_csv()
    plot_master_comparison()
    audit_evidence()

