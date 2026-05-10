# nlp_bertopic.py
# =============================================================================
# BERTopic pipeline — ERP Market Intelligence (v2, improved)
#
# IMPROVEMENTS OVER v1 (based on topic output analysis):
#
#   1. PRODUCT NAME STOPWORDS
#      Problem: topics like "0_quickbooks_netsuite_sap" — product names dominate
#      keywords, drowning out the actual feature/sentiment signal.
#      Fix: inject all product names, abbreviations, and vendor names into the
#      CountVectorizer stopword list so c-TF-IDF never picks them as keywords.
#      We already know which product a review belongs to via product_id — these
#      words add zero information to topic discrimination.
#
#   2. NOISE TOPIC FILTERING
#      Problem: topic 4 dislikes = "dislikes, dislike, dislike software" — the
#      field name itself leaked into the corpus and formed a junk topic.
#      Fix: pre-processing strips the field label words; a post-fit filter also
#      drops any topic where >50% of top keywords are in the stopword list.
#
#   3. FOUR TEXT FIELDS, NOT TWO
#      Problem: `use_case` and `recommendations` were ignored despite containing
#      rich signal ("solving multi-entity consolidation" / "get a good
#      implementation partner").
#      Fix: run four separate models — likes, dislikes, use_case, recommendations.
#      Each has distinct vocabulary and different analytical value:
#        - likes          → feature strengths (PMF score input)
#        - dislikes       → pain points (competitive displacement input)
#        - use_case       → job-to-be-done signal (segment intelligence)
#        - recommendations → advice to buyers (switching signal input)
#
#   4. LOWER OUTLIER RATE
#      Problem: 28.6% (likes) and 24.6% (dislikes) of reviews unassigned to
#      any topic — too much signal lost.
#      Fix: reduce min_topic_size from 15 → 10, reduce UMAP n_components 5 → 10
#      (more dimensions = finer cluster separation = fewer outliers).
#
# INPUT:
#   data/all_products_normalized.csv
#   Required columns: review_id, product_id, product_name, likes, dislikes,
#                     use_case, recommendations, reviewer_company_size,
#                     source_platform, review_date, overall_rating
#
# OUTPUT (data/bertopic/):
#   {field}_topics.csv         — per-review: topic_id, probability, metadata
#   topic_info_{field}.csv     — per-topic: size, top keywords (product names removed)
#   topic_dist_{field}.csv     — topic × product cross-tabulation
#   model_{field}/             — saved BERTopic model
#
# USAGE:
#   python nlp_bertopic.py
#   python nlp_bertopic.py --fields likes dislikes          # run subset of fields
#   python nlp_bertopic.py --min-topic-size 10              # default, lower = more topics
#   python nlp_bertopic.py --no-save-model                  # skip model serialization
#
# RUNTIME: ~30-45 min CPU for all 4 fields (~15K docs each pass)
#
# PAPERS:
#   Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based
#   TF-IDF procedure. arXiv:2203.05794
# =============================================================================

from __future__ import annotations

import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nlp_bertopic")


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

INPUT_CSV  = "data/g2/normalized/all_products_normalized.csv"
OUTPUT_DIR = Path("data/bertopic/v2")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Lowered from 15 → 10 to reduce outlier rate (was 28% in v1)
MIN_TOPIC_SIZE = 10

NR_TOPICS = "auto"

# Minimum text length — short strings like "-", "N/A", "Nothing" add noise
MIN_CHARS = 30

# Text fields to run BERTopic on.
# Each gets its own model, output CSVs, and saved model directory.
TEXT_FIELDS = ["likes", "dislikes", "use_case", "recommendations"]

META_COLS = [
    "review_id",
    "product_id",
    "product_name",
    "reviewer_company_size",
    "source_platform",
    "review_date",
    "overall_rating",
]

# ---------------------------------------------------------------------------
# PRODUCT NAME STOPWORDS
# ---------------------------------------------------------------------------
# WHY: Product names appear constantly in reviews because reviewers naturally
# name the product they're reviewing. This makes product names the highest
# TF-IDF terms in every topic, masking the actual feature/sentiment signal.
# We already know which product each review belongs to via product_id.
# Removing these words forces BERTopic to cluster on WHAT users say,
# not WHO they're talking about.
#
# Includes: full product names, vendor names, common abbreviations,
# product-specific jargon (qb, qbo, s4, bc, gp), and the field name words
# themselves (likes, dislikes, dislike) which leaked into v1 topics.

PRODUCT_STOPWORDS: list[str] = [
    # Oracle NetSuite
    "netsuite", "net suite", "oracle", "oracle netsuite",
    # SAP
    "sap", "s4", "s4hana", "s 4hana", "4hana", "hana", "s/4hana", "s4 hana",
    # Microsoft Dynamics
    "microsoft", "dynamics", "dynamics 365", "business central", "bc",
    "ms dynamics", "ms", "gp", "navision", "nav", "d365",
    # QuickBooks
    "quickbooks", "quickbooks online", "quickbooks enterprise",
    "quickbooks desktop", "quickbooks pro", "qb", "qbo", "qb online",
    "qb desktop", "qb enterprise", "intuit quickbooks",
    # Intuit Enterprise Suite
    "intuit", "intuit enterprise", "intuit enterprise suite", "ies",
    # Sage Intacct
    "sage", "intacct", "sage intacct", "intact",  # typo variant
    # Workday
    "workday", "workday financial", "workday financial management",
    # Acumatica
    "acumatica",
    # Xero
    "xero",
    # Certinia
    "certinia", "financialforce",  # former name
    # Generic ERP terms that add no discriminative value
    "erp", "software", "system", "platform", "tool", "product",
    "application", "app", "solution",
    # Field name leakage (v1 problem — these words appeared in topic keywords)
    "likes", "dislikes", "dislike", "like", "don like", "don dislike",
    "isn dislike", "say dislike", "think dislike",
]


# ---------------------------------------------------------------------------
# STEP 1: LOAD AND PREPROCESS
# ---------------------------------------------------------------------------

def load_and_preprocess(input_csv: str) -> pd.DataFrame:
    """
    Load normalized CSV, clean all four text fields, add {field}_clean columns.

    Cleaning per field:
      - Normalize escaped line breaks (\\r\\n → space)
      - Strip whitespace
      - Replace "nan" strings with actual NaN
      - Apply MIN_CHARS length filter (short texts hurt topic coherence)
    """
    log.info("Loading %s ...", input_csv)
    df = pd.read_csv(input_csv, low_memory=False)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    for col in TEXT_FIELDS:
        if col not in df.columns:
            log.warning("Column '%s' not found — will be skipped", col)
            df[f"{col}_clean"] = np.nan
            continue

        cleaned = (
            df[col]
            .astype(str)
            .str.replace(r"\\r\\n|\\n|\\r", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .replace("nan", np.nan)
        )

        # Length filter
        df[f"{col}_clean"] = cleaned.where(
            cleaned.str.len().fillna(0) >= MIN_CHARS, other=np.nan
        )

        n_valid = df[f"{col}_clean"].notna().sum()
        log.info("  %-20s usable docs: %d / %d", col, n_valid, len(df))

    return df


# ---------------------------------------------------------------------------
# STEP 2: BUILD BERTOPIC MODEL
# ---------------------------------------------------------------------------

def build_topic_model(min_topic_size: int = MIN_TOPIC_SIZE, nr_topics=NR_TOPICS):
    """
    Build BERTopic with explicit sub-components and product name stopwords.

    Key changes from v1:
      - UMAP n_components 5 → 10: more dimensions = finer cluster separation
        = fewer outliers assigned to topic -1
      - CountVectorizer stop_words extended with PRODUCT_STOPWORDS: removes
        all product/vendor names from c-TF-IDF keyword extraction
      - min_df lowered 5 → 3: catches terms in smaller but coherent clusters

    Pipeline (Grootendorst 2022):
      SentenceTransformer → UMAP → HDBSCAN → c-TF-IDF → topic merging
    """
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
    from umap import UMAP

    # Merge sklearn English stopwords with our product stopwords
    combined_stopwords = list(ENGLISH_STOP_WORDS) + PRODUCT_STOPWORDS

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # n_components raised to 10 (was 5) to preserve more cluster structure.
    # More dimensions → HDBSCAN can find finer-grained clusters → fewer outliers.
    umap_model = UMAP(
        n_neighbors=15,
        n_components=10,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # Extended stopwords are the key fix — product names are removed from the
    # vocabulary that c-TF-IDF uses to label topics.
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=combined_stopwords,
        min_df=3,              # lowered from 5 → catches smaller coherent clusters
    )

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        nr_topics=nr_topics,
        calculate_probabilities=True,
        verbose=True,
    )

    return model


# ---------------------------------------------------------------------------
# STEP 3: POST-FIT NOISE FILTER
# ---------------------------------------------------------------------------

def is_noise_topic(topic_id: int, keywords: str) -> bool:
    """
    Return True if a topic is a junk/noise topic that should be flagged.

    A topic is noise if the majority of its top keywords are in the
    stopword list — meaning the stopword injection didn't fully suppress
    it (can happen with very high-frequency terms that survive min_df).

    This does NOT remove the topic from the data — it adds an
    `is_noise` flag to topic_info so analysts can filter downstream.

    Args:
        topic_id: Topic ID (-1 is always noise by definition).
        keywords: Comma-separated top_keywords string.

    Returns:
        True if topic should be flagged as noise.
    """
    if topic_id == -1:
        return True

    if not keywords or keywords == "outlier — no coherent topic":
        return True

    kw_list   = [k.strip().lower() for k in keywords.split(",")]
    stopset   = set(PRODUCT_STOPWORDS)
    noise_kws = sum(1 for k in kw_list if any(s in k for s in stopset))

    # If more than half the top keywords are stopwords, it's noise
    return noise_kws > len(kw_list) / 2


# ---------------------------------------------------------------------------
# STEP 4: FIT AND EXTRACT OUTPUTS
# ---------------------------------------------------------------------------

def fit_and_extract(
    df: pd.DataFrame,
    field: str,
    output_dir: Path,
    min_topic_size: int,
    nr_topics,
    save_model: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit BERTopic on one text field and produce three output DataFrames.

    Args:
        df:             Cleaned DataFrame from load_and_preprocess().
        field:          One of TEXT_FIELDS — "likes", "dislikes", etc.
        output_dir:     Directory to write CSVs and saved model.
        min_topic_size: Minimum documents per topic.
        nr_topics:      Target topic count or "auto".
        save_model:     Whether to save fitted model to disk.

    Returns:
        (doc_df, info_df, dist_df)
    """
    text_col = f"{field}_clean"
    sub  = df[df[text_col].notna()].copy()
    docs = sub[text_col].tolist()

    if not docs:
        log.warning("No usable documents for field '%s' — skipping", field)
        empty = pd.DataFrame()
        return empty, empty, empty

    log.info("Fitting [%s] model on %d documents ...", field, len(docs))

    model = build_topic_model(min_topic_size=min_topic_size, nr_topics=nr_topics)
    topics, probs = model.fit_transform(docs)

    n_found = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = topics.count(-1) if isinstance(topics, list) else (topics == -1).sum()
    log.info(
        "[%s] Topics found: %d | Outliers: %d (%.1f%%)",
        field, n_found, n_outliers, n_outliers / len(docs) * 100,
    )

    # --- Document-level output ---
    doc_df = sub[META_COLS].copy().reset_index(drop=True)
    doc_df["topic_id"]    = topics
    doc_df["probability"] = np.round(
        probs.max(axis=1) if hasattr(probs, "ndim") and probs.ndim > 1 else probs, 4
    )
    doc_df["text_field"]  = field
    doc_df["text"]        = sub[text_col].values

    # --- Topic info output ---
    info = model.get_topic_info()

    def top_keywords(tid: int) -> str:
        if tid == -1:
            return "outlier — no coherent topic"
        words = model.get_topic(tid)
        return ", ".join([w for w, _ in words[:10]]) if words else ""

    info["top_keywords"] = info["Topic"].apply(top_keywords)
    info["is_noise"]     = info.apply(
        lambda r: is_noise_topic(r["Topic"], r["top_keywords"]), axis=1
    )

    info_df = info.rename(columns={
        "Topic": "topic_id",
        "Count": "doc_count",
        "Name":  "topic_name",
    })[["topic_id", "doc_count", "topic_name", "top_keywords", "is_noise"]]

    # --- Product × topic distribution ---
    # Exclude noise topics from the distribution so downstream analytics are clean
    clean_topic_ids = info_df[~info_df["is_noise"]]["topic_id"].tolist()
    dist_df = (
        doc_df[doc_df["topic_id"].isin(clean_topic_ids)]
        .groupby(["product_name", "topic_id"])
        .size()
        .reset_index(name="doc_count")
        .merge(info_df[["topic_id", "top_keywords"]], on="topic_id", how="left")
        .sort_values(["product_name", "doc_count"], ascending=[True, False])
    )

    # --- Save ---
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_path  = output_dir / f"{field}_topics.csv"
    info_path = output_dir / f"topic_info_{field}.csv"
    dist_path = output_dir / f"topic_dist_{field}.csv"

    doc_df.to_csv(doc_path,  index=False)
    info_df.to_csv(info_path, index=False)
    dist_df.to_csv(dist_path, index=False)

    log.info("Saved → %s", doc_path)
    log.info("Saved → %s", info_path)
    log.info("Saved → %s", dist_path)

    if save_model:
        model_path = output_dir / f"model_{field}"
        model.save(
            str(model_path),
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=EMBEDDING_MODEL,
        )
        log.info("Model saved → %s", model_path)

    return doc_df, info_df, dist_df


# ---------------------------------------------------------------------------
# STEP 5: SUMMARY REPORT
# ---------------------------------------------------------------------------

def print_summary(field: str, info_df: pd.DataFrame, dist_df: pd.DataFrame) -> None:
    """Print a readable console summary after fitting."""
    if info_df.empty:
        print(f"\n[{field}] No output — skipped.")
        return

    print(f"\n{'='*65}")
    print(f"  BERTOPIC RESULTS — {field.upper()}")
    print(f"{'='*65}")

    total      = info_df["doc_count"].sum()
    outliers   = info_df[info_df["topic_id"] == -1]["doc_count"].sum() if -1 in info_df["topic_id"].values else 0
    noise_cnt  = info_df[info_df["is_noise"] & (info_df["topic_id"] != -1)]["doc_count"].sum()
    clean      = info_df[~info_df["is_noise"]]

    print(f"  Clean topics: {len(clean)}")
    print(f"  Total docs:   {total}")
    print(f"  Outliers:     {outliers} ({outliers/total*100:.1f}%)")
    print(f"  Noise topics: {info_df['is_noise'].sum()} flagged ({noise_cnt} docs)")

    print(f"\n  TOP 10 CLEAN TOPICS:")
    top = clean[clean["topic_id"] != -1].nlargest(10, "doc_count")
    for _, row in top.iterrows():
        print(f"    [{row['topic_id']:>3}] {row['doc_count']:>5} docs  {row['top_keywords'][:55]}")

    if not dist_df.empty:
        print(f"\n  DOMINANT TOPIC PER PRODUCT:")
        top_per = dist_df.loc[dist_df.groupby("product_name")["doc_count"].idxmax()]
        for _, row in top_per.sort_values("product_name").iterrows():
            print(f"    {row['product_name']:<35} [{row['topic_id']:>3}] {row['top_keywords'][:35]}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(
    input_csv: str,
    fields: list[str],
    min_topic_size: int,
    nr_topics,
    save_model: bool,
) -> None:

    run_start = datetime.now()
    log.info("BERTopic pipeline v2 starting — %s", run_start.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Fields to run: %s", fields)
    log.info("Product stopwords injected: %d terms", len(PRODUCT_STOPWORDS))

    df = load_and_preprocess(input_csv)

    for field in fields:
        log.info("\n%s\n--- FIELD: %s ---\n%s", "="*50, field.upper(), "="*50)
        doc_df, info_df, dist_df = fit_and_extract(
            df=df,
            field=field,
            output_dir=OUTPUT_DIR,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            save_model=save_model,
        )
        print_summary(field, info_df, dist_df)

    duration = (datetime.now() - run_start).total_seconds() / 60
    log.info("\nAll fields complete in %.1f minutes.", duration)
    log.info("Outputs in: %s", OUTPUT_DIR.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BERTopic pipeline v2 — ERP reviews (product names removed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 4 fields
  python nlp_bertopic.py

  # Run only likes and dislikes
  python nlp_bertopic.py --fields likes dislikes

  # Run with custom min topic size
  python nlp_bertopic.py --min-topic-size 10

  # Fix topic count to 30 per model
  python nlp_bertopic.py --nr-topics 30

  # Skip saving model (faster, less disk)
  python nlp_bertopic.py --no-save-model
        """
    )
    parser.add_argument("--input",          default=INPUT_CSV)
    parser.add_argument("--fields",         nargs="+", default=TEXT_FIELDS,
                        choices=TEXT_FIELDS,
                        help="Which text fields to run (default: all 4)")
    parser.add_argument("--min-topic-size", type=int, default=MIN_TOPIC_SIZE)
    parser.add_argument("--nr-topics",      default=NR_TOPICS)
    parser.add_argument("--no-save-model",  action="store_true")

    args = parser.parse_args()
    nr = args.nr_topics if args.nr_topics == "auto" else int(args.nr_topics)

    main(
        input_csv=args.input,
        fields=args.fields,
        min_topic_size=args.min_topic_size,
        nr_topics=nr,
        save_model=not args.no_save_model,
    )
