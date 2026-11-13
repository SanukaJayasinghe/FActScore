"""
Phase 1: Exploratory Data Analysis (EDA) Script
CM3604 Deep Learning - FActScore Fact-Checking Project
"""

import json
import csv
import math
import statistics
from collections import Counter
from typing import List, Dict


def load_dataset(labels_path: str, passages_path: str):
    print(f"Loading passages from {passages_path}...")
    fact_to_passages = {}
    with open(passages_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            name = entry["name"]
            for p in entry["passages"]:
                if p["title"] != name:
                    raise ValueError(f"Mismatch: {name} != {p['title']}")
            fact_to_passages[entry["sent"]] = entry["passages"]

    print(f"Loading labeled facts from {labels_path}...")
    biographies = []
    examples = []
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            bio = json.loads(line)
            biographies.append(bio)
            if bio.get("annotations"):
                for sent in bio["annotations"]:
                    atomic_facts = sent.get("human-atomic-facts")
                    if atomic_facts:
                        for fact_obj in atomic_facts:
                            text = fact_obj["text"]
                            label = fact_obj["label"]
                            if text in fact_to_passages:
                                examples.append({
                                    "topic": bio.get("topic", "Unknown"),
                                    "fact": text,
                                    "label": label,
                                    "binary_label": "NS" if label == "IR" else label,
                                    "passages": fact_to_passages[text]
                                })
                            else:
                                print(f"Warning: Missing passage for fact: {text}")

    return biographies, examples, fact_to_passages


def compute_statistics(biographies: List[dict], examples: List[dict], fact_to_passages: Dict):
    total_bios = len(biographies)
    total_facts = len(examples)
    total_passages_in_file = len(fact_to_passages)

    raw_label_counts = Counter(ex["label"] for ex in examples)
    binary_label_counts = Counter(ex["binary_label"] for ex in examples)

    fact_word_lens = [len(ex["fact"].split()) for ex in examples]
    fact_char_lens = [len(ex["fact"]) for ex in examples]
    passages_per_fact = [len(ex["passages"]) for ex in examples]

    passage_word_lens = []
    passage_char_lens = []
    for ex in examples:
        for p in ex["passages"]:
            passage_word_lens.append(len(p["text"].split()))
            passage_char_lens.append(len(p["text"]))

    # Lexical vocabulary stats
    fact_vocab = set()
    for ex in examples:
        fact_vocab.update(w.lower().strip(".,;:\"'!?()[]{}") for w in ex["fact"].split())
    fact_vocab.discard("")

    passage_vocab = set()
    for ex in examples:
        for p in ex["passages"]:
            passage_vocab.update(w.lower().strip(".,;:\"'!?()[]{}") for w in p["text"].split())
    passage_vocab.discard("")

    # Simple preliminary lexical overlap per class
    # (Fraction of fact words present in concatenated passages)
    overlaps_S = []
    overlaps_NS = []
    for ex in examples:
        f_words = set(w.lower().strip(".,;:\"'!?()[]{}") for w in ex["fact"].split())
        f_words.discard("")
        p_text_all = " ".join(p["text"] for p in ex["passages"]).lower()
        if not f_words:
            overlap = 0.0
        else:
            matched = sum(1 for w in f_words if w in p_text_all)
            overlap = matched / len(f_words)

        if ex["binary_label"] == "S":
            overlaps_S.append(overlap)
        else:
            overlaps_NS.append(overlap)

    stats = {
        "total_biographies": total_bios,
        "total_examples": total_facts,
        "total_passages_in_cache": total_passages_in_file,
        "label_S_count": raw_label_counts["S"],
        "label_S_pct": (raw_label_counts["S"] / total_facts) * 100,
        "label_NS_count": raw_label_counts["NS"],
        "label_NS_pct": (raw_label_counts["NS"] / total_facts) * 100,
        "label_IR_count": raw_label_counts["IR"],
        "label_IR_pct": (raw_label_counts["IR"] / total_facts) * 100,
        "binary_S_count": binary_label_counts["S"],
        "binary_S_pct": (binary_label_counts["S"] / total_facts) * 100,
        "binary_NS_count": binary_label_counts["NS"],
        "binary_NS_pct": (binary_label_counts["NS"] / total_facts) * 100,
        "fact_words_mean": statistics.mean(fact_word_lens),
        "fact_words_median": statistics.median(fact_word_lens),
        "fact_words_std": statistics.stdev(fact_word_lens) if len(fact_word_lens) > 1 else 0.0,
        "fact_words_min": min(fact_word_lens),
        "fact_words_max": max(fact_word_lens),
        "passages_per_fact_mean": statistics.mean(passages_per_fact),
        "passages_per_fact_min": min(passages_per_fact),
        "passages_per_fact_max": max(passages_per_fact),
        "passage_words_mean": statistics.mean(passage_word_lens),
        "passage_words_median": statistics.median(passage_word_lens),
        "passage_words_min": min(passage_word_lens),
        "passage_words_max": max(passage_word_lens),
        "unique_fact_vocabulary": len(fact_vocab),
        "unique_passage_vocabulary": len(passage_vocab),
        "fact_words_in_passage_vocab_pct": (len(fact_vocab & passage_vocab) / len(fact_vocab)) * 100,
        "mean_overlap_S": statistics.mean(overlaps_S) if overlaps_S else 0.0,
        "mean_overlap_NS": statistics.mean(overlaps_NS) if overlaps_NS else 0.0,
    }

    return stats


def save_reports(stats: dict, examples: List[dict]):
    # 1. Summary CSV
    summary_path = "results/eda_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in stats.items():
            if isinstance(v, float):
                writer.writerow([k, f"{v:.4f}"])
            else:
                writer.writerow([k, v])
    print(f"Saved EDA summary table to {summary_path}")

    # 2. Detailed per-fact inspection CSV
    details_path = "results/eda_facts_detailed.csv"
    with open(details_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Topic", "Fact", "RawLabel", "BinaryLabel", "WordCount", "NumPassages", "TotalPassageChars"])
        for idx, ex in enumerate(examples, 1):
            p_chars = sum(len(p["text"]) for p in ex["passages"])
            writer.writerow([
                idx,
                ex["topic"],
                ex["fact"],
                ex["label"],
                ex["binary_label"],
                len(ex["fact"].split()),
                len(ex["passages"]),
                p_chars
            ])
    print(f"Saved detailed per-fact table to {details_path}")


def print_eda_presentation(stats: dict):
    print("=" * 70)
    print("           CM3604 DEEP LEARNING - PHASE 1 EDA SUMMARY")
    print("=" * 70)
    print(f"Total Biographies:                 {stats['total_biographies']}")
    print(f"Total Atomic Fact Claims:          {stats['total_examples']}")
    print(f"Unique Facts in Passage Cache:     {stats['total_passages_in_cache']}")
    print("-" * 70)
    print("RAW LABEL DISTRIBUTION:")
    print(f"  Supported (S):                   {stats['label_S_count']:3d} ({stats['label_S_pct']:.2f}%)")
    print(f"  Not Supported (NS):              {stats['label_NS_count']:3d} ({stats['label_NS_pct']:.2f}%)")
    print(f"  Irrelevant (IR):                 {stats['label_IR_count']:3d} ({stats['label_IR_pct']:.2f}%)")
    print("-" * 70)
    print("BINARY EVALUATION DISTRIBUTION (IR -> NS):")
    print(f"  Supported (S):                   {stats['binary_S_count']:3d} ({stats['binary_S_pct']:.2f}%)")
    print(f"  Not Supported (NS):              {stats['binary_NS_count']:3d} ({stats['binary_NS_pct']:.2f}%)")
    print(f"  -> Majority Baseline Accuracy:   {stats['binary_S_pct']:.2f}%")
    print("-" * 70)
    print("TEXT & PASSAGE LENGTH METRICS:")
    print(f"  Fact Length (Words):             Mean={stats['fact_words_mean']:.2f}, Median={stats['fact_words_median']}, Min={stats['fact_words_min']}, Max={stats['fact_words_max']}")
    print(f"  Passages Per Fact:               Mean={stats['passages_per_fact_mean']:.2f}, Min={stats['passages_per_fact_min']}, Max={stats['passages_per_fact_max']}")
    print(f"  Passage Length (Words):          Mean={stats['passage_words_mean']:.2f}, Median={stats['passage_words_median']}, Min={stats['passage_words_min']}, Max={stats['passage_words_max']}")
    print("-" * 70)
    print("LEXICAL OVERLAP PRELIMINARY SIGNALS:")
    print(f"  Mean Raw Overlap for 'S' Facts:  {stats['mean_overlap_S'] * 100:.2f}%")
    print(f"  Mean Raw Overlap for 'NS' Facts: {stats['mean_overlap_NS'] * 100:.2f}%")
    print(f"  -> Overlap Gap (S vs NS):        {(stats['mean_overlap_S'] - stats['mean_overlap_NS']) * 100:+.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    bios, exs, passages = load_dataset(
        "data/dev_labeled_ChatGPT.jsonl",
        "data/passages_bm25_ChatGPT_humfacts.jsonl"
    )
    stats = compute_statistics(bios, exs, passages)
    print_eda_presentation(stats)
    save_reports(stats, exs)

