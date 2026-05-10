# nlp_bertopic.py
# =============================================================================
# BERTopic pipeline for ERP Market Intelligence.
#
# Runs two separate BERTopic models:
#   - Model A: fit on `likes`    text → what customers love per product
#   - Model B: fit on `dislikes` text → pain points and complaints per product
#
# WHY SEPARATE MODELS:
#   Positive and negative sentiment cluster around fundamentally different
#   vocabulary. Merging them into one model causes sentiment-opposite reviews
#   of the same feature (e.g. "great support" vs "terrible support") to get
#   grouped into one noisy topic. Separate models = clean, actionable topics.
#   (Grootendorst 2022 — BERTopic)
#
# WHY ONE GLOBAL MODEL (not per-product):
#   A global model produces shared topic IDs across all products, so you can
#   directly compare "how often does topic 7 (implementation pain) appear in
#   NetSuite vs QuickBooks reviews?" without alignment. Product-level filtering
#   happens in the output CSV, not in the model.
#
# INPUT:
#   data/all_products_normalized.csv   (14,933 rows, 28 columns)
#   Required columns: review_id, product_id, product_name, likes, dislikes,
#                     reviewer_company_size, source_platform, review_date
#
# OUTPUT (all written to data/bertopic/):
#   likes_topics.csv        — one row per review: topic_id, label, probability
#   dislikes_topics.csv     — same for dislikes
#   topic_info_likes.csv    — topic summary: id, size, top keywords
#   topic_info_dislikes.csv
#   topic_dist_likes.csv    — topic size by product (cross-product comparison)
#   topic_dist_dislikes.csv
#   model_likes/            — saved BERTopic model (reload without re-fitting)
#   model_dislikes/         — saved BERTopic model
#
# USAGE:
#   python nlp_bertopic.py
#   python nlp_bertopic.py --input data/all_products_normalized.csv
#   python nlp_bertopic.py --min-topic-size 20
#   python nlp_bertopic.py --nr-topics 30
#   python nlp_bertopic.py --no-save-model
#
# RUNTIME ESTIMATE:
#   ~15-25 min on CPU for 14,933 rows with all-MiniLM-L6-v2 embeddings.
#   Embedding is the slow step — it runs once per field (likes / dislikes).
#   Progress bars are shown at each stage.
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
OUTPUT_DIR = Path("data/bertopic")

# Embedding model — free, local, no API key needed.
# Downloads ~90MB on first run, cached to ~/.cache/huggingface/ after that.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# BERTopic HDBSCAN settings.
# min_topic_size: minimum documents per topic.
#   Lower = more granular topics; too low = noisy micro-topics.
#   For ~7000 docs per field, 15 is a sensible floor.
# nr_topics: "auto" lets BERTopic merge similar topics automatically.
#   Set to an integer (e.g. 30) to fix the topic count.
MIN_TOPIC_SIZE = 15
NR_TOPICS      = "auto"

# Drop texts shorter than this — catches "-", "N/A", "Nothing" etc.
MIN_CHARS = 30

# Metadata columns preserved in the per-document output CSV.
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
# STEP 1: LOAD AND PREPROCESS
# ---------------------------------------------------------------------------

def load_and_preprocess(input_csv: str) -> pd.DataFrame:
    """
    Load the normalized CSV and clean text fields ready for BERTopic.

    Cleaning:
      - Normalize \\r\\n line break sequences (common in G2 review exports)
      - Drop rows where both likes AND dislikes are null
      - Drop texts shorter than MIN_CHARS (e.g. "N/A", "-", "Nothing")

    Returns DataFrame with `likes_clean` and `dislikes_clean` columns added.
    """
    log.info("Loading %s ...", input_csv)
    df = pd.read_csv(input_csv, low_memory=False)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    for col in ["likes", "dislikes"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"\\r\\n|\\n|\\r", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .replace("nan", np.nan)
        )

    before = len(df)
    df = df.dropna(subset=["likes", "dislikes"], how="all")
    log.info("Dropped %d rows with no text in either field", before - len(df))

    # Apply minimum length filter — short texts hurt topic coherence
    df["likes_clean"] = df["likes"].where(
        df["likes"].str.len().fillna(0) >= MIN_CHARS, other=np.nan
    )
    df["dislikes_clean"] = df["dislikes"].where(
        df["dislikes"].str.len().fillna(0) >= MIN_CHARS, other=np.nan
    )

    log.info(
        "Usable likes: %d  |  Usable dislikes: %d",
        df["likes_clean"].notna().sum(),
        df["dislikes_clean"].notna().sum(),
    )
    return df


# ---------------------------------------------------------------------------
# STEP 2: BUILD BERTOPIC MODEL
# ---------------------------------------------------------------------------

def build_topic_model(min_topic_size: int = MIN_TOPIC_SIZE, nr_topics=NR_TOPICS):
    """
    Build a BERTopic model with explicit sub-components.

    BERTopic pipeline (Grootendorst 2022):
      1. SentenceTransformer embeddings — dense 384-dim vector per document
      2. UMAP — dimensionality reduction (384-dim → 5-dim)
      3. HDBSCAN — density-based clustering (no pre-specified cluster count)
      4. c-TF-IDF — class-based TF-IDF for per-topic keyword extraction
      5. Topic merging (nr_topics)

    Each component is instantiated explicitly so every hyperparameter
    is visible and auditable for the academic write-up.

    Returns: configured BERTopic instance (not yet fitted).
    """
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    log.info("Building BERTopic model (min_topic_size=%d, nr_topics=%s)",
             min_topic_size, nr_topics)

    # Embedding model — runs locally, downloaded once to HuggingFace cache
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # UMAP: reduce embeddings to 5-dim for clustering.
    # n_neighbors=15 — local neighbourhood size for manifold construction.
    # min_dist=0.0 — tighter clusters (recommended for HDBSCAN downstream).
    # metric="cosine" — appropriate for sentence embedding space.
    # random_state=42 — reproducibility: same seed = same output.
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # HDBSCAN: density-based clustering — no need to specify cluster count.
    # min_cluster_size = min_topic_size: minimum documents to form a topic.
    # prediction_data=True: required for BERTopic's probability estimation.
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # CountVectorizer for c-TF-IDF keyword extraction.
    # ngram_range=(1,2): unigrams + bigrams — captures "customer support".
    # stop_words="english": removes common words that pollute topic keywords.
    # min_df=5: ignore terms in fewer than 5 docs (rare = noise).
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=5,
    )

    # c-TF-IDF: BERTopic's class-based TF-IDF.
    # reduce_frequent_words=True: downweights cross-topic common terms.
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        nr_topics=nr_topics,
        calculate_probabilities=True,  # per-doc topic confidence scores
        verbose=True,
    )

    return model


# ---------------------------------------------------------------------------
# STEP 3: FIT AND EXTRACT OUTPUTS
# ---------------------------------------------------------------------------

def fit_and_extract(
    df: pd.DataFrame,
    text_col: str,
    label: str,
    output_dir: Path,
    min_topic_size: int,
    nr_topics,
    save_model: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit BERTopic on one text field and produce three output DataFrames.

    Args:
        df:             Cleaned DataFrame from load_and_preprocess().
        text_col:       Column to fit on: "likes_clean" or "dislikes_clean".
        label:          "likes" or "dislikes" — used in output filenames.
        output_dir:     Directory to write CSVs and saved model.
        min_topic_size: Minimum documents per topic.
        nr_topics:      Target topic count or "auto".
        save_model:     Save fitted model to disk for later reuse.

    Returns:
        (doc_df, info_df, dist_df)
        doc_df  — per-document topic assignment + metadata
        info_df — per-topic summary (size, keywords)
        dist_df — product × topic cross-tabulation
    """
    sub  = df[df[text_col].notna()].copy()
    docs = sub[text_col].tolist()

    log.info("Fitting [%s] model on %d documents ...", label, len(docs))

    model = build_topic_model(min_topic_size=min_topic_size, nr_topics=nr_topics)

    # Core BERTopic call:
    #   topics — list of topic IDs per document (-1 = outlier, no clear topic)
    #   probs  — probability array (shape: n_docs or n_docs × n_topics)
    topics, probs = model.fit_transform(docs)

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    log.info("Done. Topics found: %d (excl. outliers)", n_topics)

    # --- Document-level output ---
    doc_df = sub[META_COLS].copy().reset_index(drop=True)
    doc_df["topic_id"]    = topics
    doc_df["probability"] = np.round(
        probs.max(axis=1) if hasattr(probs, "ndim") and probs.ndim > 1 else probs, 4
    )
    doc_df["text_field"]  = label
    doc_df["text"]        = sub[text_col].values

    # --- Topic info output ---
    info = model.get_topic_info()

    def top_keywords(tid: int) -> str:
        if tid == -1:
            return "outlier — no coherent topic"
        words = model.get_topic(tid)
        return ", ".join([w for w, _ in words[:10]]) if words else ""

    info["top_keywords"] = info["Topic"].apply(top_keywords)
    info_df = info.rename(columns={
        "Topic": "topic_id",
        "Count": "doc_count",
        "Name":  "topic_name",
    })[["topic_id", "doc_count", "topic_name", "top_keywords"]]

    # --- Product × topic distribution ---
    dist_df = (
        doc_df[doc_df["topic_id"] != -1]
        .groupby(["product_name", "topic_id"])
        .size()
        .reset_index(name="doc_count")
        .merge(info_df[["topic_id", "top_keywords"]], on="topic_id", how="left")
        .sort_values(["product_name", "doc_count"], ascending=[True, False])
    )

    # --- Save to disk ---
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_path  = output_dir / f"{label}_topics.csv"
    info_path = output_dir / f"topic_info_{label}.csv"
    dist_path = output_dir / f"topic_dist_{label}.csv"

    doc_df.to_csv(doc_path,  index=False)
    info_df.to_csv(info_path, index=False)
    dist_df.to_csv(dist_path, index=False)

    log.info("Saved → %s", doc_path)
    log.info("Saved → %s", info_path)
    log.info("Saved → %s", dist_path)

    if save_model:
        model_path = output_dir / f"model_{label}"
        model.save(
            str(model_path),
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=EMBEDDING_MODEL,
        )
        log.info("Model saved → %s", model_path)

    return doc_df, info_df, dist_df


# ---------------------------------------------------------------------------
# STEP 4: SUMMARY REPORT
# ---------------------------------------------------------------------------

def print_summary(label: str, info_df: pd.DataFrame, dist_df: pd.DataFrame) -> None:
    """Print a readable console summary after fitting — top topics globally and per product."""
    print(f"\n{'='*60}")
    print(f"  BERTOPIC RESULTS — {label.upper()}")
    print(f"{'='*60}")

    n_topics      = len(info_df[info_df["topic_id"] != -1])
    total_docs    = info_df["doc_count"].sum()
    outlier_count = info_df[info_df["topic_id"] == -1]["doc_count"].sum() if (-1 in info_df["topic_id"].values) else 0

    print(f"  Topics:       {n_topics}")
    print(f"  Total docs:   {total_docs}")
    print(f"  Outliers:     {outlier_count} ({outlier_count/total_docs*100:.1f}% — topic -1)")

    print(f"\n  TOP 10 GLOBAL TOPICS:")
    top = info_df[info_df["topic_id"] != -1].nlargest(10, "doc_count")
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

def main(input_csv: str, min_topic_size: int, nr_topics, save_model: bool) -> None:

    run_start = datetime.now()
    log.info("BERTopic pipeline starting — %s", run_start.strftime("%Y-%m-%d %H:%M:%S"))

    df = load_and_preprocess(input_csv)

    log.info("\n--- LIKES MODEL ---")
    likes_doc, likes_info, likes_dist = fit_and_extract(
        df=df, text_col="likes_clean", label="likes",
        output_dir=OUTPUT_DIR, min_topic_size=min_topic_size,
        nr_topics=nr_topics, save_model=save_model,
    )

    log.info("\n--- DISLIKES MODEL ---")
    dislikes_doc, dislikes_info, dislikes_dist = fit_and_extract(
        df=df, text_col="dislikes_clean", label="dislikes",
        output_dir=OUTPUT_DIR, min_topic_size=min_topic_size,
        nr_topics=nr_topics, save_model=save_model,
    )

    print_summary("likes",    likes_info,    likes_dist)
    print_summary("dislikes", dislikes_info, dislikes_dist)

    duration = (datetime.now() - run_start).total_seconds() / 60
    log.info("Pipeline complete in %.1f minutes.", duration)
    log.info("All outputs in: %s", OUTPUT_DIR.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BERTopic pipeline — ERP review likes and dislikes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nlp_bertopic.py
  python nlp_bertopic.py --input data/all_products_normalized.csv
  python nlp_bertopic.py --min-topic-size 10
  python nlp_bertopic.py --nr-topics 30
  python nlp_bertopic.py --no-save-model
        """
    )
    parser.add_argument("--input",          default=INPUT_CSV)
    parser.add_argument("--min-topic-size", type=int, default=MIN_TOPIC_SIZE)
    parser.add_argument("--nr-topics",      default=NR_TOPICS)
    parser.add_argument("--no-save-model",  action="store_true")

    args = parser.parse_args()
    nr = args.nr_topics if args.nr_topics == "auto" else int(args.nr_topics)

    main(
        input_csv=args.input,
        min_topic_size=args.min_topic_size,
        nr_topics=nr,
        save_model=not args.no_save_model,
    )
