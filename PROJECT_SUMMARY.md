# Project Summary & Implementation Work Plan

## CM3604 Deep Learning: Fact-Checking LLM Outputs (FActScore)
**Institution:** Robert Gordon University (RGU) Aberdeen  
**Module:** CM3604 – Deep Learning  
**Execution Environment:** Python 3.10 (CPU-only mode — No GPU/CUDA)  
**Target Benchmarks:**
- **Task 1 (Word Overlap Baseline):** $\ge 75\%$ Accuracy
- **Task 2 (Textual Entailment with DeBERTa):** $\ge 83\%$ Accuracy
- **Task 3 (Error Analysis & Report):** 10 FP + 10 FN categorized into 2–4 fine-grained error classes, statistical distribution, and 3 detailed case studies.

---

## 1. What We Did Today

1. **Deep Specification & Rubric Analysis**:
   - Analyzed `deepLearning-CW.pdf` and the coursework specification for CM3604.
   - Identified all criteria required for a **Grade A** submission across Part 1 (40%), Part 2 (40%), and Part 3 (20%).
   - Confirmed the submission deliverables: `factcheck.py` (code) and `[Written_Submission].pdf` (written error analysis report).
   - Confirmed the mandatory CLI interface commands:
     ```bash
     python factchecking_main.py --mode word_overlap
     python factchecking_main.py --mode entailment
     ```

2. **Codebase & Architecture Inspection**:
   - Examined `factcheck.py` starter classes: `FactExample`, `EntailmentModel`, `FactChecker`, `RandomGuessFactChecker`, `AlwaysEntailedFactChecker`, `WordRecallThresholdFactChecker`, and `EntailmentFactChecker`.
   - Examined the driver script `factscore_retrieval_interface.py` / `factchecking_main.py`, noting the evaluation logic: mapping `IR` (Irrelevant) to `NS` (Not Supported) for binary evaluation against `S` (Supported).
   - Analyzed sample JSONL data structures for atomic fact claims and BM25-retrieved Wikipedia passages with `<s>...</s>` markers.

3. **Workflow Confirmation**:
   - Validated and approved the user's proposed 10-step progression (Phases 0 through 9).
   - Confirmed user constraint: **CPU-only execution** (no GPU/CUDA used throughout any phase).

4. **Phase 0 (Environment & Repository Setup) Execution**:
   - Created the project directory structure:
     - `data/` (for input JSONL datasets)
     - `results/` (for EDA statistics, ablation tables, comparison CSVs, and plots)
   - Created Python 3.10 virtual environment (`.venv`) to ensure isolated dependency management.
   - Created `requirements.txt` containing the exact version specifications:
     - `numpy==2.0.2`
     - `torch==2.5.0`
     - `torchvision==0.20.0`
     - `tqdm==4.66.5`
     - `spacy==3.8.2`
     - `nltk==3.9.1`
     - `transformers==4.45.2`
     - `sentencepiece==0.2.0`
     - `protobuf==5.28.3`
     - `rouge-score==0.1.2`
   - Created `factcheck.py` with initial starter class stubs.
   - Created `factchecking_main.py` and `factscore_retrieval_interface.py` with explicit UTF-8 file reading to ensure smooth execution on Windows.
   - Created `.gitignore` excluding `.venv/`, `__pycache__/`, and temporary files.
   - Formulated git commit message: `chore: initialize Phase 0 project structure, environment, and baseline stubs`.

---

## 2. What Is Pending

1. **Dataset Files Placement**:
   - Copying the full dataset files into the `data/` directory:
     - `data/dev_labeled_ChatGPT.jsonl`
     - `data/passages_bm25_ChatGPT_humfacts.jsonl`
2. **Virtual Environment Dependency Installation**:
   - Completing package installation inside `.venv` via pip (or using installed system/wheel packages).
3. **Execution of Subsequent Phases (Phases 1 to 9)**:
   - EDA generation, baseline benchmarking, Task 1 development and verification, Task 2 DeBERTa development and CPU optimization, Task 3 error extraction and taxonomy, audit, and final submission report compilation.

---

## 3. What Is the Next Step

- **Immediate Next Step**:
  Once the two JSONL dataset files are placed inside `data/` and dependencies are confirmed, begin **Phase 1: Dataset Understanding & Exploratory Data Analysis (EDA)**.
  - Load and inspect all records from both JSONL files.
  - Generate distribution statistics: label counts (`S`, `NS`, `IR`), fact lengths, passage sentence counts, and vocabulary statistics.
  - Save `results/eda_summary.csv` and summary charts for inclusion in the final report.

---

## 4. Full Implementation Work Plan (Phases 0 to 9)

```
  Phase 0: Setup & Environment Verification (COMPLETED)
       │
  Phase 1: Dataset Understanding & Exploratory Data Analysis (EDA)
       │
  Phase 2: Baseline Model Evaluation (Random & Always Supported)
       │
  Phase 3: Task 1 — Word Overlap Model Development (Target: ≥ 75%)
       │
  Phase 4: Task 1 Official Verification & Threshold Freezing
       │
  Phase 5: Task 2 — DeBERTa Textual Entailment Pipeline (Target: ≥ 83%, CPU Optimized)
       │
  Phase 6: Task 2 Official Verification, Memory Safety & Pruning
       │
  Phase 7: Task 3 — Error Analysis & Taxonomy (10 FP + 10 FN)
       │
  Phase 8: Comprehensive Evidence Audit & Model Comparison
       │
  Phase 9: Final Code Cleanup, PDF Report Generation & Submission Check
```

---

### Phase 0: Setup & Environment Initialization [DONE]
- **Objective**: Establish isolated project workspace, create directory skeleton, configure Python 3.10 environment, and define official entry points.
- **Key Deliverables**:
  - `data/` and `results/` folders created.
  - Python 3.10 virtual environment (`.venv`) initialized.
  - `requirements.txt`, `factcheck.py`, `factchecking_main.py`, `factscore_retrieval_interface.py`, `.gitignore` created.
  - Official CLI execution verified in CPU mode.

---

### Phase 1: Dataset Understanding & Exploratory Data Analysis (EDA)
- **Objective**: Rigorously analyze dataset distributions, claim formats, and retrieved Wikipedia context passages.
- **Actions**:
  1. Parse `dev_labeled_ChatGPT.jsonl` and map atomic facts to Wikipedia passages via `passages_bm25_ChatGPT_humfacts.jsonl`.
  2. Compute class distribution: frequency and percentage of `S` (Supported), `NS` (Not Supported), and `IR` (Irrelevant). Verify `IR -> NS` binary conversion.
  3. Analyze text length distributions: word counts of atomic facts, character lengths of retrieved passages, number of passages per fact.
  4. Inspect text cleaning requirements: identify XML-like `<s>` and `</s>` tags in retrieved passages.
- **Outputs & Evidence**:
  - `results/eda_summary.csv`: Comprehensive quantitative dataset metrics.
  - Summary plots of claim lengths and class distributions.
- **Quality Gate**: 100% of facts successfully mapped to retrieved passages with zero missing keys.

---

### Phase 2: Baseline Model Evaluation
- **Objective**: Measure and record reference performance floors using trivial baselines.
- **Actions**:
  1. Evaluate `RandomGuessFactChecker` on the dataset.
  2. Evaluate `AlwaysEntailedFactChecker` (majority class baseline).
  3. Compute confusion matrices, overall Accuracy, class-wise Precision, Recall, and F1 scores.
- **Outputs & Evidence**:
  - `results/baseline_metrics.csv` containing baseline performance numbers.
  - Recorded benchmark floor to compare against Tasks 1 and 2.
- **Quality Gate**: Baseline evaluation numbers documented and reproducible via CLI.

---

### Phase 3: Task 1 — Word Overlap Model Development (Target: $\ge 75\%$ Accuracy)
- **Objective**: Implement, systematically ablate, and optimize `WordRecallThresholdFactChecker` to exceed the $75.0\%$ coursework requirement.
- **Actions**:
  1. **Preprocessing Pipeline**:
     - Case normalization, punctuation stripping.
     - Stopword removal (NLTK English stopwords).
     - Morphological normalization comparison: Porter Stemmer vs. Lemmatization.
  2. **Scoring Function Exploration**:
     - Word Recall (proportion of fact keywords found in candidate passage).
     - Jaccard Similarity ($|A \cap B| / |A \cup B|$).
     - Dice Coefficient ($2 |A \cap B| / (|A| + |B|)$).
     - TF-IDF Cosine Similarity.
     - ROUGE-1 / ROUGE-L overlap scores.
  3. **Passage Aggregation**:
     - Compare max score across individual passages vs. pooled passage representation.
  4. **Threshold Tuning ($\tau$)**:
     - Sweep decision threshold $\tau \in [0.0, 1.0]$ with fine increments (0.01–0.02) to maximize binary classification accuracy.
- **Outputs & Evidence**:
  - `results/word_overlap_ablation.csv`: Full ablation table comparing preprocessing options, similarity metrics, and thresholds.
  - `results/word_overlap_threshold_curve.png`: Accuracy curve vs. threshold $\tau$.
- **Quality Gate**: Classification accuracy strictly exceeds $75.0\%$ on the evaluation dataset.

---

### Phase 4: Task 1 Official Verification & Threshold Freezing
- **Objective**: Freeze optimal settings in `factcheck.py` and validate the official submission command.
- **Actions**:
  1. Hardcode optimal preprocessing and decision threshold $\tau$ inside `WordRecallThresholdFactChecker.predict()`.
  2. Execute the official coursework command:
     ```bash
     python factchecking_main.py --mode word_overlap
     ```
  3. Verify that accuracy $\ge 75\%$ is achieved out-of-the-box with no manual configuration.
- **Outputs & Evidence**:
  - CLI log documenting successful run, confusion matrix, precision, recall, and final accuracy.
- **Quality Gate**: CLI command runs flawlessly without errors and achieves $\ge 75\%$ accuracy.

---

### Phase 5: Task 2 — Textual Entailment Pipeline (Target: $\ge 83\%$ Accuracy, CPU Optimized)
- **Objective**: Implement `EntailmentModel` and `EntailmentFactChecker` using pretrained `DeBERTa-v3-base-mnli-fever-anli`, optimized for CPU inference.
- **Actions**:
  1. **Passage Preprocessing & Sentence Splitting**:
     - Clean `<s>` and `</s>` markup tags.
     - Segment passages into discrete candidate premise sentences using NLTK / regular expressions.
  2. **Lightweight Lexical Pruning**:
     - Filter candidate premise sentences using a fast word-overlap threshold.
     - Discard unpromising sentences prior to neural inference to drastically accelerate CPU runtime (60–80% speedup).
  3. **Entailment Scoring & Aggregation**:
     - Evaluate `(premise_sentence, atomic_fact)` pairs with DeBERTa.
     - Extract Softmax probabilities for `[entailment, neutral, contradiction]`.
     - Implement continuous scoring: $\text{Score}(\text{Fact}) = \max_{p} P(\text{Entailment} \mid p, \text{Fact})$.
  4. **Decision Boundary Optimization**:
     - Compare discrete classification (any sentence predicted as entailment) vs. continuous thresholding ($P(\text{Entailment}) \ge \theta$).
     - Tune threshold $\theta$ to achieve maximum accuracy ($\ge 83\%$).
  5. **Memory & CPU Safety**:
     - Explicit tensor deletion (`del inputs, outputs, logits`) and garbage collection (`gc.collect()`).
- **Outputs & Evidence**:
  - `results/entailment_strategies.csv`: Comparison table of discrete vs. continuous decisions, pruning cutoffs, and thresholds.
  - Recorded inference runtime and memory metrics.
- **Quality Gate**: Textual entailment accuracy strictly exceeds $83.0\%$ on CPU without memory issues.

---

### Phase 6: Task 2 Official Verification & Performance Freezing
- **Objective**: Freeze selected entailment parameters and validate the official submission command.
- **Actions**:
  1. Lock optimal pruning and entailment threshold inside `EntailmentFactChecker`.
  2. Execute official coursework command:
     ```bash
     python factchecking_main.py --mode entailment
     ```
  3. Verify reproducibility and ensure no OOM or crashes occur during full evaluation.
- **Outputs & Evidence**:
  - Official CLI execution log confirming accuracy $\ge 83\%$.
- **Quality Gate**: Command executes without errors and meets all assignment criteria.

---

### Phase 7: Task 3 — Qualitative & Quantitative Error Analysis (20 Marks)
- **Objective**: Conduct comprehensive error analysis on the final entailment model to satisfy all Part 3 rubric requirements.
- **Actions**:
  1. **Error Extraction**:
     - Extract all prediction failures: 10 False Positives (predicted $S$, gold $NS$) and 10 False Negatives (predicted $NS$, gold $S$).
  2. **Fine-Grained Taxonomy Definition**:
     - Define 2 to 4 precise linguistic and semantic error categories with 1–2 sentence operational definitions (e.g., *Entity & Coreference Mismatch*, *Partial Information / Granularity Gap*, *Temporal / Numerical Discrepancy*, *Retrieval Incompleteness*).
  3. **Error Categorization & Statistics**:
     - Assign every inspected error to an operational category.
     - Compile aggregate distribution tables and frequency percentages.
  4. **Detailed Case Studies**:
     - Select 3 representative examples and document:
       - (a) Fact claim and retrieved evidence text
       - (b) Ground truth label
       - (c) Model prediction
       - (d) Assigned error category
       - (e) 1–3 sentence rationale explaining why the failure occurred.
- **Outputs & Evidence**:
  - `results/error_cases_extracted.csv`: Complete listing and classification of 20 error cases.
  - Error distribution summary table and category bar chart.
  - Detailed case study writeups for report inclusion.
- **Quality Gate**: 10 FP + 10 FN fully categorized with sound justifications.

---

### Phase 8: Final Comparison & Evidence Audit
- **Objective**: Compile a comprehensive comparative analysis across all systems and verify internal data integrity.
- **Actions**:
  1. Build master comparative table:
     - Random Guess Baseline
     - Always Entailed Baseline
     - Word Overlap Classifier (Task 1)
     - DeBERTa Entailment Model (Task 2)
  2. Compare Accuracy, Class-wise Precision, Recall, and F1.
  3. Audit all numbers, tables, and figures to guarantee 100% consistency with experimental output logs.
- **Outputs & Evidence**:
  - `results/model_comparison_master.csv`.
  - Comprehensive comparison chart showing model evolution.
- **Quality Gate**: All reported metrics verified against raw logs.

---

### Phase 9: Final Code Cleanup, PDF Report Generation & Submission Packaging
- **Objective**: Prepare final deliverables for submission in strict compliance with university rules.
- **Actions**:
  1. Clean `factcheck.py` to ensure only clean, self-contained implementation code is retained.
  2. Compile the formal written report into PDF format containing:
     - Introduction and task framing.
     - Part 1 Word Overlap design, preprocessing, ablations, and threshold curves.
     - Part 2 Textual Entailment architecture, sentence splitting, pruning efficiency, and decision strategies.
     - Part 3 Error Analysis: taxonomy definitions, distribution statistics, and 3 detailed case studies.
     - Critical reflection and future improvements.
  3. Perform final verification run of both official CLI commands.
- **Outputs & Evidence**:
  - `factcheck.py`: Final submitted source file.
  - `CM3604_FActScore_Coursework_Report.pdf`: Final submission PDF document.
- **Quality Gate**: All files validated, formatted, and ready for upload to CampusMoodle.

