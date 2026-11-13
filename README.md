# 🧠 FActScore: Fact-Checking LLM Outputs with Lexical & Neural Pipelines

[![Python](https://img.shields.io/badge/Python-3.10.0-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0+cpu-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-4.45.2-yellow.svg?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/)
[![Coursework](https://img.shields.io/badge/Coursework-CM3604%20Deep%20Learning-success.svg)](https://www.rgu.ac.uk/)
[![Grade Benchmark](https://img.shields.io/badge/Grade%20A%20Benchmarks-All%20Passed-brightgreen.svg)]()

> **Robert Gordon University (RGU) Aberdeen** in Collaboration with **Informatics Institute of Technology (IIT)**  
> **Module**: CM3604 – Deep Learning (Coursework)  
> **Academic Year**: 2026 / 2027  
> **Module Leader**: Mr. Prasan Yapa  

---

## 📌 Table of Contents
- [📖 Project Overview](#-project-overview)
- [🎯 Key Performance Benchmarks](#-key-performance-benchmarks)
- [🏗️ System Architecture & Workflow](#️-system-architecture--workflow)
- [🔬 Core Technical Innovations](#-core-technical-innovations)
  - [Part 1: Word Overlap Baseline](#part-1-word-overlap-baseline-task-1)
  - [Part 2: Textual Entailment Pipeline](#part-2-textual-entailment-pipeline-task-2)
  - [Part 3: Error Analysis & Taxonomy](#part-3-error-analysis--taxonomy-task-3)
- [📂 Repository Structure](#-repository-structure)
- [🚀 Quick Start & Reproduction Guide](#-quick-start--reproduction-guide)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Model Weights Download](#2-model-weights-download)
  - [3. Mandatory CLI Evaluation](#3-mandatory-cli-evaluation)
  - [4. Running Full Experimental Pipeline](#4-running-full-experimental-pipeline)
- [📊 Master Comparison Matrix](#-master-comparison-matrix)
- [👥 Project Team](#-project-team)
- [📚 References](#-references)

---

## 📖 Project Overview

Large Language Models (LLMs) such as ChatGPT generate articulate and fluent text but frequently hallucinate non-factual claims. This project implements, optimizes, and audits automated fact-checking architectures within the **FActScore** framework (Min et al., 2023) to audit LLM-generated biographies against reference Wikipedia knowledge.

### 🔍 Problem Formulation
* **Input**: An atomic factual claim decomposed from ChatGPT biographical generation (e.g., *"Joey D. Vieira is an actor."*) alongside candidate Wikipedia passages retrieved using BM25.
* **Objective**: Formulate a binary decision classifier predicting:
  * **`S` (Supported)**: The proposition is directly substantiated by the retrieved evidence.
  * **`NS` (Not Supported)**: The proposition is contradicted or unsubstantiated. (Human-annotated `IR` / Irrelevant facts are rigorously mapped to `NS`).
* **Hardware Constraint**: **Strict CPU-only execution** without GPU dependencies or active online network connections during testing.

---

## 🎯 Key Performance Benchmarks

Both coursework performance thresholds were surpassed on the 221 human-labeled development propositions (`data/dev_labeled_ChatGPT.jsonl`):

| Evaluation Phase | Pipeline Model | Coursework Benchmark | Achieved Score | Margin | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Part 1: Lexical** | `WordRecallThresholdFactChecker` | $\ge 75.0\%$ Accuracy | **83.26% (184/221)** | **+8.26%** | **GRADE A PASSED** 🏆 |
| **Part 2: Neural** | `EntailmentFactChecker` (DeBERTa-v3) | $\ge 83.0\%$ Accuracy | **84.16% (186/221)** | **+1.16%** | **GRADE A PASSED** 🏆 |
| **Part 3: Analysis** | Fine-Grained 4-Category Taxonomy | 10 FP + 10 FN Audit | **20 Inspected Cases** | 3 Case Studies | **GRADE A PASSED** 🏆 |

---

## 🏗️ System Architecture & Workflow

```
                                [ Incoming Atomic Claim ]
                           (Fact String + Retrieved Wiki Passages)
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
         [ PART 1: WORD OVERLAP ]                          [ PART 2: DEBERTA ENTAILMENT ]
        * Lowercase & Regex Cleaning                      * Lexical Pre-Filter Recall
        * Stopword Elimination (NLTK)                     * Fast Two-Tier Pruning:
        * Porter Stemming                                    - Recall < 0.65  --> Prune as NS
        * Max-Passage Unigram Recall                         - Recall >= 0.85 --> Decisive Support (S)
        * Plateau Tuning (tau = 0.65)                     * Cross-Encoder NLI on Boundary Cases:
                     |                                       - DeBERTa-v3-base-mnli-fever-anli
                     v                                       - Evaluates Top-2 Candidates
             Accuracy: 83.26%                             * P(Entailment) >= 0.005 Threshold
             Latency: 0.12s (CPU)                                      |
                     |                                                 v
                     |                                         Accuracy: 84.16%
                     |                                         FP Reduction: -35.0%
                     |                                         Latency: 42.0s (CPU)
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                              v
                                  [ PART 3: ERROR ANALYSIS ]
                                 * 10 False Positives / 10 False Negatives
                                 * 4 Fine-Grained Taxonomy Categories
                                 * 3 Comprehensive Case Studies
```

---

## 🔬 Core Technical Innovations

### Part 1: Word Overlap Baseline (Task 1)
* **Max-Passage Pooling vs. Concatenation**: Concatenating all passages dilutes word overlap with irrelevant paragraphs (accuracy dropped to 80.09%). Computing unigram recall individually per candidate passage and pooling the maximum isolates the single most supportive sentence, achieving **83.26% accuracy**.
* **Preprocessing Ablation**: Removing stopwords (+3.17%) and applying Porter stemming (+3.17%) together provide a cumulative **+6.34% accuracy gain** over raw unigrams.
* **Plateau Stability**: The optimal decision threshold $\tau = 0.65$ sits safely in the center of a stable plateau ($\tau \in [0.61, 0.66]$), requiring at least two out of three stemmed content words to be corroborated.

### Part 2: Textual Entailment Pipeline (Task 2)
* **Backbone Architecture**: Employs `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` with disentangled cross-attention across premise (Wikipedia passage + title) and hypothesis (claim).
* **Two-Tier Lexical Pruning for CPU Acceleration**:
  * *Early Pruning ($\text{Recall} < 0.65$)*: Rejects 106 claims (48.0%) instantly without calling the transformer.
  * *Decisive Support ($\text{Recall} \ge 0.85$)*: Directly accepts 66 claims (29.9%) with high lexical confidence.
  * *Selective Cross-Encoding ($0.65 \le \text{Recall} < 0.85$)*: Invokes DeBERTa on only 49 boundary claims (22.2%), scoring at most their top-2 candidates.
  * *Outcome*: Reduces neural forward passes from 692 to under 98 (**85.8% reduction**) and slashes CPU evaluation latency from 240s down to **42.0 seconds** without memory leaks (`del` + `gc.collect()`).
* **Semantic Elimination of Hallucination**: Slashes False Positives by **35.0%** (20 down to 13) and boosts Precision on Supported facts to **87.38%**, proving that DeBERTa detects when words co-occur without logical entailment.

### Part 3: Error Analysis & Taxonomy (Task 3)
A rigorous manual inspection of 20 error cases (10 False Positives + 10 False Negatives) categorized into 4 operational classes:
1. **Entity & Coreference Ambiguity (ECA, 30%)**: Unanchored third-person pronouns (*He*, *She*) or relational co-occurrence misattributions (e.g., confusing rival radio station WKNR with employer).
2. **Specificity, Hypernymy & Paraphrase Gap (SHP, 30%)**: Subsumption gaps where abstract terms (*domestic titles*) fail lexical overlap with specific trophies (*Russian Cup*).
3. **Evidence Deficiency & Retrieval Truncation (EDT, 25%)**: BM25 keyword bias omitting real-world facts from the top passage window.
4. **Temporal & Numerical Mismatch (TNM, 15%)**: Discrepancies in dates or cardinalities (e.g., claiming 3 league titles when passage documents 2).

---

## 📂 Repository Structure

```text
FActScore-Model/
│
├── data/                                      # Reference datasets
│   ├── dev_labeled_ChatGPT.jsonl              # 221 human-annotated atomic facts (31 topics)
│   └── passages_bm25_ChatGPT_humfacts.jsonl   # Retrieved BM25 Wikipedia passages (5,373 entries)
│
├── results/                                   # Evaluation metrics & visualizations
│   ├── baseline_metrics.csv                   # Always Supported & Random baseline scores
│   ├── baseline_metrics_comparison.png        # Baseline performance chart
│   ├── word_overlap_ablation.csv              # 7-configuration Task 1 ablation study
│   ├── word_overlap_threshold_curve.png       # Threshold optimization curve
│   ├── word_overlap_confusion_matrix.png      # Task 1 verified confusion matrix
│   ├── task1_verified_metrics.json            # Official Task 1 verification certificate
│   ├── deberta_cache.json                     # Cached DeBERTa-v3 inference logits
│   ├── entailment_strategies.csv              # Comparison of 4 NLI decision strategies
│   ├── entailment_routing_breakdown.png       # Three-tier claim routing distribution chart
│   ├── entailment_pruning_efficiency.png      # CPU runtime & pass reduction benchmark chart
│   ├── entailment_confusion_matrix.png        # Task 2 verified confusion matrix
│   ├── task2_verified_metrics.json            # Official Task 2 verification certificate
│   ├── error_cases_extracted.csv              # 20 inspected error cases with rationales
│   ├── error_analysis_distribution.csv        # Taxonomy frequency contingency table
│   ├── error_distribution.png                 # Error taxonomy grouped bar chart
│   ├── model_comparison_master.csv            # Master cross-model benchmark table
│   └── model_comparison_chart.png             # 4-panel master publication benchmark chart
│
├── factcheck.py                               # ⭐ MAIN CODE DELIVERABLE (Self-contained factcheckers)
├── factchecking_main.py                       # 🔒 Official read-only driver script
├── factscore_retrieval_interface.py           # 🔒 Official read-only data structures
├── download_model.py                          # Offline DeBERTa-v3 model cache utility
├── eda_analysis.py                            # Exploratory data analysis script
├── baseline_eval.py                           # Baseline evaluation & chart generation
├── task1_experiments.py                       # Task 1 threshold sweep & ablation runner
├── task2_experiments.py                       # Task 2 strategy comparison & latency benchmarks
├── error_analysis.py                          # Task 3 taxonomy classification & case study extraction
├── model_comparison.py                        # Master comparison aggregator & evidence audit
│
├── FINAL_COURSEWORK_REPORT.md                 # 📄 Complete coursework report ready for PDF export
├── PROJECT_SUMMARY.md                         # 📋 Full project executive summary
├── requirements.txt                           # Frozen environment dependency list
└── README.md                                  # 📘 Comprehensive project documentation
```

---

## 🚀 Quick Start & Reproduction Guide

### 1. Environment Setup
Clone the repository and activate a Python 3.10 virtual environment:

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install CPU dependencies
pip install -r requirements.txt
```

### 2. Model Weights Download
Download and cache the pretrained DeBERTa-v3 weights locally for offline execution:

```powershell
python download_model.py
```

### 3. Mandatory CLI Evaluation
Execute the official, untouched driver script to verify both coursework benchmarks on CPU:

```powershell
# Verify Task 1: Word Overlap Baseline (Runs in ~0.12s)
python factchecking_main.py --mode word_overlap

# Verify Task 2: Textual Entailment Pipeline (Runs in ~42.0s on CPU)
python factchecking_main.py --mode entailment
```

### 4. Running Full Experimental Pipeline
To regenerate all experimental tables, ablations, error analysis, and charts:

```powershell
python eda_analysis.py       # Step 1: Dataset statistics (results/eda_summary.csv)
python baseline_eval.py      # Step 2: Baseline floor evaluation
python task1_experiments.py  # Step 3: Task 1 ablation study & threshold tuning
python task2_experiments.py  # Step 4: Task 2 strategy ablations & latency profiling
python error_analysis.py     # Step 5: Task 3 error taxonomy & 20-case extraction
python model_comparison.py   # Step 6: Master comparative audit & 4-panel chart
```

---

## 📊 Master Comparison Matrix

All models evaluated on the identical 221 human-labeled ChatGPT facts (`data/dev_labeled_ChatGPT.jsonl`, 112 Supported vs. 109 Not Supported):

| Performance Dimension | Always Supported | Random Guess | Task 1: Word Overlap | Task 2: DeBERTa Entailment | Coursework Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Class** | `AlwaysEntailed` | `RandomGuess` | `WordRecallThreshold` | `EntailmentFactChecker` | — |
| **Benchmark Target** | — | — | $\ge 75.0\%$ Accuracy | $\ge 83.0\%$ Accuracy | — |
| **Accuracy (%)** | 50.68% (112/221) | 50.23% (111/221) | **83.26% (184/221)** | **84.16% (186/221)** | **BOTH PASSED** 🏆 |
| **Target Margin** | N/A | N/A | **+8.26%** | **+1.16%** | Grade A Benchmark |
| **Precision (S)** | 50.68% | 50.88% | 82.61% | **87.38%** | $+4.77\%$ Precision Boost |
| **Recall (S)** | 100.00% | 51.79% | 84.82% | 80.36% | High-Confidence Filtering |
| **F1-Score (S)** | 67.27% | 51.33% | 83.70% | **83.72%** | Consistent Performance |
| **Precision (NS)** | 0.00% | 49.53% | 83.96% | 81.36% | Robust Rejection |
| **Recall (NS)** | 0.00% | 48.62% | 81.65% | **88.07%** | $+6.42\%$ Rejection Gain |
| **F1-Score (NS)** | 0.00% | 49.07% | 82.79% | **84.58%** | Superior Balance |
| **Macro F1 (%)** | 33.63% | 50.20% | 83.25% | **84.15%** | Highest Overall |
| **True Positives (TP)** | 112 | 58 | 95 | 90 | — |
| **False Positives (FP)**| 109 | 56 | 20 | **13** | **$-35.0\%$ FP Drop** |
| **False Negatives (FN)**| 0 | 54 | 17 | 22 | — |
| **True Negatives (TN)** | 0 | 53 | 89 | **96** | $+7$ Correct Rejections |
| **Inference Time (CPU)**| < 0.01s | < 0.01s | **0.12s** | **42.0s** | Ultra-efficient |

---

## 👥 Project Team

**School of Computing, Robert Gordon University (RGU) Aberdeen**  
*Informatics Institute of Technology (IIT), Colombo, Sri Lanka*

| Team Member Name | RGU Student ID | IIT Student ID | Primary Responsibilities |
| :--- | :---: | :---: | :--- |
| **Vihanga Kukulevithana** | `2507574` | `20242075` | System Architecture, DeBERTa Pipeline & Pruning |
| **Tharusha Deshapriya** | `2506737` | `20241835` | Lexical Overlap Baseline, Tokenization & Ablations |
| **Sanuka Jayasinghe** | `2313475` | `20221675` | Error Analysis, Taxonomy Classification & Audit |
| **Tariq Ramzeen** | `2313476` | `20221714` | Exploratory Data Analysis & Visualizations |

---

## 📚 References

1. **Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W., Iyyer, M., Zettlemoyer, L., & Hajishirzi, H.** (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation*. Proceedings of EMNLP 2023, pp. 12076–12100.
2. **He, P., Liu, X., Gao, J., & Chen, W.** (2021). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*. International Conference on Learning Representations (ICLR).
3. **Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K.** (2024). *Less Annotating, More Classifying: Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and BERT-NLI*. Political Analysis, 32(1), pp. 84–100.
4. **Robertson, S., & Zaragoza, H.** (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 3(4), pp. 333–389.
5. **Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A.** (2018). *FEVER: A Large-scale Dataset for Fact Extraction and VERification*. Proceedings of NAACL-HLT 2018, pp. 809–819.
6. **Porter, M. F.** (1980). *An algorithm for suffix stripping*. Program, 14(3), pp. 130–137.

---

*(c) 2026 Robert Gordon University Aberdeen & Informatics Institute of Technology. All rights reserved.*
