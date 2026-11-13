"""
Phase 7: Task 3 - Error Analysis & Taxonomy Implementation
CM3604 Deep Learning - FActScore Fact-Checking Project
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
# 1. TAXONOMY DEFINITION (4 Fine-Grained Linguistic & Semantic Categories)
# ---------------------------------------------------------------------------
TAXONOMY = {
    "ECA": {
        "name": "Entity & Coreference Ambiguity (ECA)",
        "short": "Entity & Coreference",
        "definition": "The model misattributes properties or actions across entities mentioned in competing contexts, or fails to resolve ambiguous third-person pronouns (he, she, they) across sentence boundaries."
    },
    "TNM": {
        "name": "Temporal & Numerical Granularity Mismatch (TNM)",
        "short": "Temporal & Numerical",
        "definition": "The claim and evidence align on entities and predicates but contradict on exact dates, calendar years, temporal intervals, or numerical cardinalities."
    },
    "SHP": {
        "name": "Specificity, Hypernymy & Paraphrase Gap (SHP)",
        "short": "Specificity & Hypernymy",
        "definition": "The claim involves abstract hypernyms, compound modifiers, or pragmatic inferences (e.g. former implies left) that cannot be verified by surface word overlap or single-step NLI."
    },
    "EDT": {
        "name": "Evidence Deficiency & Retrieval Truncation (EDT)",
        "short": "Evidence Deficiency",
        "definition": "The retrieved Wikipedia passages fail to contain necessary context for genuine ground-truth facts due to BM25 ranking bias, passage window truncation, or missing entity background."
    }
}


# ---------------------------------------------------------------------------
# 2. THE 20 INSPECTED ERROR CASES (10 False Positives + 10 False Negatives)
# ---------------------------------------------------------------------------
INSPECTED_ERRORS = [
    # --- 10 FALSE POSITIVES (Model predicted S, Gold label is NS) ---
    {
        "id": "FP-01",
        "index": 1,
        "error_type": "FP",
        "category_code": "SHP",
        "fact": "Marianne McAndrew is an American.",
        "gold": "NS",
        "pred": "S",
        "recall": 1.0,
        "evidence_title": "Marianne McAndrew",
        "evidence_snippet": "Marianne Christine McAndrew is an actress known for her role as Irene Molloy in the film 'Hello, Dolly!' Career. 'Hello, Dolly!' was McAndrew's first credited film role.",
        "rationale": "The evidence substantiates her acting career in Hollywood but never explicitly confirms American nationality. High lexical overlap from the biography header caused the model to hallucinate factual support for nationality without explicit proof."
    },
    {
        "id": "FP-02",
        "index": 6,
        "error_type": "FP",
        "category_code": "EDT",
        "fact": "Cleveland is in Ohio.",
        "gold": "NS",
        "pred": "S",
        "recall": 1.0,
        "evidence_title": "Mike Trivisonno",
        "evidence_snippet": "Cleveland's eastern suburbs. He was married to his high school sweetheart Linda Conforto, who died of lung and brain cancer in 2009, at age 61. The couple have three children...",
        "rationale": "While the proposition is geographically true in the real world, the retrieved passage about Mike Trivisonno only mentions 'Cleveland's eastern suburbs' and completely omits 'Ohio'. The model committed a false positive by relying on pre-trained parametric knowledge rather than retrieved evidence."
    },
    {
        "id": "FP-03",
        "index": 68,
        "error_type": "FP",
        "category_code": "ECA",
        "fact": "Trivisonno worked at WKNR.",
        "gold": "NS",
        "pred": "S",
        "recall": 1.0,
        "evidence_title": "Mike Trivisonno",
        "evidence_snippet": "...in the ratings, even though it was usually up against Cleveland Indians games on WKNR. It did so well, the decision was made to put Trivisonno in afternoon drive-time...",
        "rationale": "The evidence states that Trivisonno's radio broadcast competed 'against' games aired on WKNR. The model falsely conflated co-occurrence in the sports broadcast context with an employer-employee relationship."
    },
    {
        "id": "FP-04",
        "index": 125,
        "error_type": "FP",
        "category_code": "EDT",
        "fact": "Mayo Clinic is an organization.",
        "gold": "NS",
        "pred": "S",
        "recall": 0.6667,
        "evidence_title": "Travis Oliphant",
        "evidence_snippet": "Travis Oliphant Travis Oliphant is an American data scientist and businessman. He is a co-founder of NumFOCUS, 501(c)(3) nonprofit charity in the United States, and sits on its advisory board.",
        "rationale": "The retrieved passage discusses NumFOCUS and Anaconda, but contains no reference to Mayo Clinic. The model suffered from an evidence deficiency where high baseline token overlap on generic tokens triggered an unsupported entailment decision."
    },
    {
        "id": "FP-05",
        "index": 126,
        "error_type": "FP",
        "category_code": "ECA",
        "fact": "He has worked at Mayo Clinic.",
        "gold": "NS",
        "pred": "S",
        "recall": 0.6667,
        "evidence_title": "Travis Oliphant",
        "evidence_snippet": "Travis Oliphant Travis Oliphant is an American data scientist and businessman. He is a co-founder of NumFOCUS, 501(c)(3) nonprofit charity in the United States, and sits on its advisory board.",
        "rationale": "The passage details Travis Oliphant's business ventures. The model confused external medical institution associations with documented employment, misattributing employment history to the biographical subject."
    },
    {
        "id": "FP-06",
        "index": 134,
        "error_type": "FP",
        "category_code": "SHP",
        "fact": "He served as a sniper.",
        "gold": "NS",
        "pred": "S",
        "recall": 1.0,
        "evidence_title": "Rob Furlong",
        "evidence_snippet": "Rob Furlong Rob Furlong (born 11 November 1976) is a Canadian former military sniper who, from March 2002 until November 2009, held the world record for the longest confirmed sniper kill in combat...",
        "rationale": "The passage describes Furlong as a 'former military sniper'. In FActScore annotation guidelines, atomic propositions with unverified sub-predicates receive NS. The model predicted S due to exact unigram overlap on 'sniper'."
    },
    {
        "id": "FP-07",
        "index": 147,
        "error_type": "FP",
        "category_code": "TNM",
        "fact": "With CSKA, he won three Russian Premier League titles.",
        "gold": "NS",
        "pred": "S",
        "recall": 0.6667,
        "evidence_title": "Daniel Carvalho",
        "evidence_snippet": "CSKA Moscow - Russian Premier League: 2005, 2006 - Russian Cup: 2004–05, 2005–06, 2008–09 - Russian Super Cup: 2004, 2006, 2007 - UEFA Cup: 2004–05.",
        "rationale": "The evidence lists only two Russian Premier League titles (2005, 2006). High unigram overlap on tournament names masked the cardinality contradiction (3 vs 2), causing a false positive."
    },
    {
        "id": "FP-08",
        "index": 170,
        "error_type": "FP",
        "category_code": "ECA",
        "fact": "Kerser is from Sydney.",
        "gold": "NS",
        "pred": "S",
        "recall": 1.0,
        "evidence_title": "Kerser",
        "evidence_snippet": "benchmark for all major battles that have proceeded it. At the time of the battle, it elevated both rapper's careers greatly really showcasing the difference in culture between Sydney and Melbourne hip hop.",
        "rationale": "The text references a cultural rivalry between Sydney and Melbourne rap scenes. The model improperly attributed geographic origin ('from Sydney') to Kerser simply because Sydney was mentioned in the passage."
    },
    {
        "id": "FP-09",
        "index": 186,
        "error_type": "FP",
        "category_code": "TNM",
        "fact": "Shahnaz married in 1959.",
        "gold": "NS",
        "pred": "S",
        "recall": 0.6667,
        "evidence_title": "Shahnaz Pahlavi",
        "evidence_snippet": "Her first marriage, at age sixteen, was to Ardeshir Zahedi on 11 October 1957, at Golestan Palace in Tehran. He was one-time Iranian foreign minister and twice Iranian ambassador...",
        "rationale": "The evidence explicitly confirms her marriage occurred in 1957. The model identified the marriage event and numerical year token, but failed to perform the exact temporal match required to detect that 1959 contradicts 1957."
    },
    {
        "id": "FP-10",
        "index": 193,
        "error_type": "FP",
        "category_code": "SHP",
        "fact": "Jidenna's music is a fusion of hip-hop.",
        "gold": "NS",
        "pred": "S",
        "recall": 0.8,
        "evidence_title": "Jidenna",
        "evidence_snippet": "Mobisson released his first album with Black Spadez as their final project at Milton Academy, where Mobisson graduated in 2003. In 2008, after graduating from Stanford...",
        "rationale": "The passage details musical groups and academic history, but does not verify the specific musical classification 'fusion of hip-hop'. High vocabulary similarity led to semantic over-generalization."
    },

    # --- 10 FALSE NEGATIVES (Model predicted NS, Gold label is S) ---
    {
        "id": "FN-01",
        "index": 10,
        "error_type": "FN",
        "category_code": "SHP",
        "fact": "Gerhard Fischer invented the first handheld, battery-operated metal detector.",
        "gold": "S",
        "pred": "NS",
        "recall": 0.5556,
        "evidence_title": "Gerhard Fischer (inventor)",
        "evidence_snippet": "...the hand-held metal detector is disputed, the hand-held metal detector was made in 1925 and was first patented by Dr. Gerhard Fisher in 1931. A metal detector had been invented some forty years earlier...",
        "rationale": "The ground truth is supported, but the presence of complex descriptive modifiers ('first handheld, battery-operated') lowered unigram recall to 0.5556, triggering lexical early-pruning before DeBERTa could evaluate semantic entailment."
    },
    {
        "id": "FN-02",
        "index": 49,
        "error_type": "FN",
        "category_code": "ECA",
        "fact": "He moved to Dutch club FC Groningen.",
        "gold": "S",
        "pred": "NS",
        "recall": 0.6,
        "evidence_title": "Ko Itakura",
        "evidence_snippet": "Ko Itakura. Club career. Born in Yokohama, Ko Itakura joined J1 League club Kawasaki Frontale in 2015... In January 2019, he joined Premier League club Manchester City, but was immediately loaned to FC Groningen until summer 2020.",
        "rationale": "The text confirms his loan move to FC Groningen, but the fact uses the unanchored pronoun 'He'. The combination of pronoun resolution and loan transfer wording produced insufficient recall (0.60), leading to early false rejection."
    },
    {
        "id": "FN-03",
        "index": 50,
        "error_type": "FN",
        "category_code": "TNM",
        "fact": "He made the move in 2019.",
        "gold": "S",
        "pred": "NS",
        "recall": 0.6667,
        "evidence_title": "Ko Itakura",
        "evidence_snippet": "In January 2019, he joined Premier League club Manchester City, but was immediately loaned to Dutch Eredivisie club FC Groningen until summer 2020.",
        "rationale": "The temporal transfer occurred in January 2019. The model struggled to link the phrase 'made the move' to the two-part loan agreement described in the premise, failing the temporal inference threshold."
    },
    {
        "id": "FN-04",
        "index": 100,
        "error_type": "FN",
        "category_code": "EDT",
        "fact": "He served for many years as a pastor.",
        "gold": "S",
        "pred": "NS",
        "recall": 0.5,
        "evidence_title": "Malcolm Hedding",
        "evidence_snippet": "Malcolm Hedding Malcolm Hedding is an ordained minister with the Assemblies of God and a leader of the Christian Zionist movement. Early life and ministry. Hedding was born in South Africa...",
        "rationale": "While the passage confirms he was an 'ordained minister', the specific duration 'for many years as a pastor' was absent from the retrieved top passages due to retrieval truncation, forcing an early lexical rejection."
    },
    {
        "id": "FN-05",
        "index": 135,
        "error_type": "FN",
        "category_code": "SHP",
        "fact": "Furlong left the military.",
        "gold": "S",
        "pred": "NS",
        "recall": 0.6667,
        "evidence_title": "Rob Furlong",
        "evidence_snippet": "Rob Furlong Rob Furlong (born 11 November 1976) is a Canadian former military sniper who, from March 2002 until November 2009, held the world record for the longest confirmed sniper kill...",
        "rationale": "The premise describes Furlong as a 'former military sniper'. Validating that he 'left the military' requires a pragmatic inference that 'former' entails past departure. DeBERTa assigned low entailment probability due to the lack of explicit resignation text."
    },
    {
        "id": "FN-06",
        "index": 145,
        "error_type": "FN",
        "category_code": "SHP",
        "fact": "With CSKA, he won domestic titles.",
        "gold": "S",
        "pred": "NS",
        "recall": 0.3333,
        "evidence_title": "Daniel Carvalho",
        "evidence_snippet": "CSKA Moscow - Russian Premier League: 2005, 2006 - Russian Cup: 2004–05, 2005–06, 2008–09 - Russian Super Cup: 2004, 2006, 2007 - UEFA Cup: 2004–05.",
        "rationale": "The evidence lists specific cup and league triumphs. The claim expresses this through the hypernym 'domestic titles'. Word overlap recall was only 0.33, causing immediate rejection before neural cross-encoding could resolve the hypernym relationship."
    },
    {
        "id": "FN-07",
        "index": 162,
        "error_type": "FN",
        "category_code": "EDT",
        "fact": "Rio de Janeiro is in Brazil.",
        "gold": "S",
        "pred": "NS",
        "recall": 0.25,
        "evidence_title": "Jonathan Haagensen",
        "evidence_snippet": "Jonathan Haagensen Jonathan Haagensen Cerqueira (born February 23, 1983) is a Brazilian actor, model, and singer. Life and career. A resident of the Vidigal community since his birth...",
        "rationale": "The biographical passage mentions the 'Vidigal community' and 'Brazilian actor', but the geographical containment relationship 'Rio de Janeiro is in Brazil' was not explicit in the retrieved passage text, leading to an evidence deficiency rejection."
    },
    {
        "id": "FN-08",
        "index": 190,
        "error_type": "FN",
        "category_code": "EDT",
        "fact": "They divorced in 1964.",
        "gold": "S",
        "pred": "NS",
        "recall": 0.0,
        "evidence_title": "Shahnaz Pahlavi",
        "evidence_snippet": "Shahnaz Pahlavi Shahnaz Pahlavi (, born 27 October 1940) is the first child of the Shah of Iran, Mohammad Reza Pahlavi, and his first wife, Princess Fawzia of Egypt.",
        "rationale": "The retrieved passage focuses entirely on Shahnaz's lineage and early childhood, completely omitting the divorce event and date. With unigram recall of 0.0, the pipeline correctly recognized zero evidence, but BM25 retrieval failed to surface the true fact."
    },
    {
        "id": "FN-09",
        "index": 214,
        "error_type": "FN",
        "category_code": "ECA",
        "fact": "He has since then starred in 'New Heart.'.",
        "gold": "S",
        "pred": "NS",
        "recall": 0.75,
        "evidence_title": "Ji Sung",
        "evidence_snippet": "Ji Sung is a South Korean actor. He is best known for his roles in the television dramas 'All In' (2003), 'Save the Last Dance for Me' (2004), 'New Heart' (2007)...",
        "rationale": "Although 'New Heart' appears in the filmography, the third-person subject pronoun 'He' and the temporal phrase 'since then' attenuated DeBERTa's cross-encoder score below the activation threshold."
    },
    {
        "id": "FN-10",
        "index": 220,
        "error_type": "FN",
        "category_code": "ECA",
        "fact": "He has since then starred in 'Familiar Wife.'.",
        "gold": "S",
        "pred": "NS",
        "recall": 0.75,
        "evidence_title": "Ji Sung",
        "evidence_snippet": "In 2018, Ji starred in the fantasy romance comedy drama 'Familiar Wife'. The same year he starred in the period film 'Fengshui'.",
        "rationale": "The sentence confirms starring in 'Familiar Wife', but the subject is written as 'Ji' in the evidence and 'He' in the claim. The lack of coreference resolution in candidate sentence matching caused the entailment score to fall just short of the decision boundary."
    }
]


# ---------------------------------------------------------------------------
# 3. COMPUTE AGGREGATE FREQUENCY DISTRIBUTION
# ---------------------------------------------------------------------------
def compute_distribution(errors):
    counts = {code: {"FP": 0, "FN": 0, "Total": 0} for code in TAXONOMY}
    for err in errors:
        code = err["category_code"]
        err_type = err["error_type"]
        counts[code][err_type] += 1
        counts[code]["Total"] += 1
    
    total_errors = len(errors)
    distribution_rows = []
    for code, data in counts.items():
        pct = (data["Total"] / total_errors) * 100.0
        distribution_rows.append({
            "category_code": code,
            "category_name": TAXONOMY[code]["name"],
            "category_definition": TAXONOMY[code]["definition"],
            "fp_count": data["FP"],
            "fn_count": data["FN"],
            "total_count": data["Total"],
            "percentage": round(pct, 2)
        })
    return distribution_rows, counts


# ---------------------------------------------------------------------------
# 4. SAVE EXTRACTED ERRORS & DISTRIBUTION CSVs
# ---------------------------------------------------------------------------
def save_csvs(errors, distribution_rows):
    # 1. Error cases extracted
    fieldnames = [
        "id", "index", "error_type", "category_code", "category_name",
        "fact", "gold", "pred", "recall", "evidence_title", "evidence_snippet", "rationale"
    ]
    with open("results/error_cases_extracted.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for err in errors:
            row = dict(err)
            row["category_name"] = TAXONOMY[err["category_code"]]["name"]
            writer.writerow(row)
    print("Saved results/error_cases_extracted.csv (20 error cases: 10 FP, 10 FN)")

    # 2. Distribution table
    dist_fields = ["category_code", "category_name", "category_definition", "fp_count", "fn_count", "total_count", "percentage"]
    with open("results/error_analysis_distribution.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=dist_fields)
        writer.writeheader()
        writer.writerows(distribution_rows)
    print("Saved results/error_analysis_distribution.csv")


# ---------------------------------------------------------------------------
# 5. GENERATE PUBLICATION-QUALITY VISUALIZATION
# ---------------------------------------------------------------------------
def plot_error_distribution(distribution_rows):
    categories = [row["category_code"] for row in distribution_rows]
    cat_labels = [TAXONOMY[code]["short"] for code in categories]
    fp_counts = [row["fp_count"] for row in distribution_rows]
    fn_counts = [row["fn_count"] for row in distribution_rows]
    total_counts = [row["total_count"] for row in distribution_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#FAFAFA")

    # Subplot 1: Grouped Bar Chart (FP vs FN)
    x = np.arange(len(categories))
    width = 0.35

    rects1 = ax1.bar(x - width/2, fp_counts, width, label='False Positives (FP: 10)', color='#D9534F', edgecolor='#A94442', alpha=0.9)
    rects2 = ax1.bar(x + width/2, fn_counts, width, label='False Negatives (FN: 10)', color='#337AB7', edgecolor='#2E6DA4', alpha=0.9)

    ax1.set_title("Error Frequency by Fine-Grained Category (FP vs FN)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel("Number of Errors", fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_labels, fontsize=10, rotation=15, ha='right')
    ax1.set_ylim(0, max(max(fp_counts), max(fn_counts)) + 1.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', frameon=True, facecolor='#FFFFFF')

    # Value labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{int(h)}', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color='#D9534F')
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{int(h)}', xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color='#337AB7')

    # Subplot 2: Total Category Distribution (Horizontal Bar Chart)
    y = np.arange(len(categories))
    colors = ['#5BC0DE', '#F0AD4E', '#5CB85C', '#9970AB']
    rects_tot = ax2.barh(y, total_counts, color=colors, edgecolor='#555555', height=0.55, alpha=0.85)

    ax2.set_title("Total Error Share Across Categories (N = 20)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Total Errors", fontsize=11, fontweight='bold')
    ax2.set_yticks(y)
    ax2.set_yticklabels(cat_labels, fontsize=10)
    ax2.set_xlim(0, max(total_counts) + 2)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)

    for rect, tot in zip(rects_tot, total_counts):
        w = rect.get_width()
        pct = (tot / sum(total_counts)) * 100
        ax2.annotate(f'{tot} ({pct:.0f}%)', xy=(w, rect.get_y() + rect.get_height() / 2),
                     xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontweight='bold')

    plt.tight_layout()
    plot_path = "results/error_distribution.png"
    plt.savefig(plot_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved visualization to {plot_path}")


# ---------------------------------------------------------------------------
# 6. MAIN EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    distribution_rows, counts = compute_distribution(INSPECTED_ERRORS)
    save_csvs(INSPECTED_ERRORS, distribution_rows)
    plot_error_distribution(distribution_rows)
    print("\nError Distribution Summary:")
    for row in distribution_rows:
        print(f"  [{row['category_code']}] {row['category_name']}: {row['total_count']} ({row['percentage']}%) [FP: {row['fp_count']}, FN: {row['fn_count']}]")

