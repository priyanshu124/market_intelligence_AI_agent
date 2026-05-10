# nlp_bertopic.py
# =============================================================================
# BERTopic pipeline — ERP Market Intelligence (v4)
#
# ROOT CAUSE OF v3 KEYWORD POLLUTION (diagnosed from output):
#
#   v3 keywords still showed "the, and, to, of, is" and product names despite
#   them being in the CountVectorizer stopword list. Root cause:
#
#   BERTopic's reduce_outliers() + update_topics() regenerates topic keywords
#   using a FRESH internal vectorizer that does NOT inherit the stopword list
#   from the original CountVectorizer. This is a known BERTopic behaviour —
#   update_topics() accepts a new vectorizer_model parameter, but if omitted
#   it falls back to a default vectorizer with no custom stopwords.
#
# FIXES IN v4:
#
#   1. EXPLICIT update_topics() WITH VECTORIZER
#      After reduce_outliers(), call update_topics(vectorizer_model=...) and
#      pass the same configured CountVectorizer explicitly. This ensures
#      stopwords are applied during keyword re-generation.
#
#   2. TWO-STAGE TOPIC REPRESENTATION
#      Stage 1: fit_transform() for clustering (UMAP + HDBSCAN)
#      Stage 2: update_topics() with full stopword vectorizer for keywords
#      This is the BERTopic-recommended approach for custom stopwords.
#
#   3. POST-HOC KEYWORD CLEANING
#      After update_topics(), run a final filter that removes any remaining
#      stopword tokens from the keyword lists. This catches edge cases where
#      a term is so frequent it survives CountVectorizer filtering (e.g.
#      "accounting" appears in >50% of docs so IDF weight approaches 0 but
#      c-TF-IDF can still surface it). Replaces polluted keywords with the
#      next-best non-stopword terms from model.get_topic().
#
#   4. PRODUCT-NAME CLUSTER DETECTION
#      Topics where the top-3 keywords are all product names are flagged as
#      product_cluster=True in topic_info. These are still valid (they show
#      which product features are discussed) but are labelled separately so
#      analysts can filter them out of cross-product comparison views.
#
#   5. MEANINGFUL TOPIC LABELS
#      topic_name is now generated as a human-readable label using the first
#      3 non-stopword, non-product-name keywords joined with " | " instead of
#      BERTopic's default "ID_word1_word2_word3" format.
#
# INPUT:  data/all_products_normalized.csv
# OUTPUT: data/bertopic/{field}_topics.csv, topic_info_{field}.csv,
#         topic_dist_{field}.csv, model_{field}/
#
# USAGE:
#   python nlp_bertopic.py
#   python nlp_bertopic.py --fields likes dislikes
#   python nlp_bertopic.py --no-save-model
#
# RUNTIME: ~35-50 min CPU for all 4 fields
#
# PAPERS:
#   Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based
#   TF-IDF procedure. arXiv:2203.05794
# =============================================================================

from __future__ import annotations

import argparse
import logging
import re
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
OUTPUT_DIR = Path("data/bertopic/v3")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Per-field settings — recommendations has ~2.7K docs vs ~14K for others
FIELD_SETTINGS: dict[str, dict] = {
    "likes":           {"min_topic_size": 10, "nr_topics": 60},
    "dislikes":        {"min_topic_size": 10, "nr_topics": 60},
    "use_case":        {"min_topic_size": 10, "nr_topics": 60},
    "recommendations": {"min_topic_size": 5,  "nr_topics": 30},
}

TEXT_FIELDS = list(FIELD_SETTINGS.keys())
MIN_CHARS   = 30

META_COLS = [
    "review_id", "product_id", "product_name",
    "reviewer_company_size", "source_platform",
    "review_date", "overall_rating",
]


# ---------------------------------------------------------------------------
# STOPWORD SETS
# ---------------------------------------------------------------------------

# All product names, vendor names, abbreviations, and UI-specific terms.
# These are removed from keyword extraction because:
#   (a) we already know product_id per review — product names add zero info
#   (b) they're so frequent they dominate c-TF-IDF scores
PRODUCT_STOP: set[str] = {
    # Oracle NetSuite
    "netsuite", "net suite", "oracle",
    # SAP
    "sap", "s4", "s4hana", "hana", "4hana", "s/4hana", "ecc", "brf",
    # Microsoft Dynamics
    "microsoft", "dynamics", "dynamics 365", "business central", "bc",
    "gp", "navision", "nav", "d365", "365", "central",
    # QuickBooks (all variants)
    "quickbooks", "quickbooks online", "quickbooks enterprise", "quickbooks desktop",
    "qb", "qbo", "qbe", "qbd", "quick books", "quickbook",
    "online", "desktop", "pro", "desktop pro", "desktop version", "online version",
    # Intuit / IES
    "intuit", "intuit enterprise", "ies",
    # Sage Intacct (including typo intaact)
    "sage", "intacct", "sage intacct", "intaact",
    # Workday
    "workday", "workday financial",
    # Acumatica (VAR = value-added reseller, Acumatica channel term)
    "acumatica", "aia", "var", "vars",
    # Xero
    "xero",
    # Certinia / FinancialForce
    "certinia", "financialforce", "financial force", "ffa", "ff", "psa",
    # Cross-product generic labels
    "erp", "software", "system", "platform", "tool", "product",
    "application", "solution", "program", "saas",
    "cloud", "cloud based",
    # QuickBooks name fragments
    "books", "book",
    # Field name leakage
    "likes", "dislikes", "dislike",
}

# Common English function words that CountVectorizer should remove but
# sometimes don't when custom stop_words are passed as a list (encoding issues).
# We handle these with a post-hoc filter instead of relying on CountVectorizer.
FUNCTION_WORDS: set[str] = {
    "the", "and", "to", "of", "is", "it", "for", "in", "our", "we",
    "that", "are", "with", "you", "my", "this", "be", "can", "as",
    "have", "not", "at", "so", "an", "or", "all", "i", "a", "was",
    "they", "their", "on", "by", "from", "but", "do", "had", "he",
    "her", "his", "if", "into", "more", "no", "out", "she", "up",
    "us", "very", "when", "which", "who", "will", "your", "been",
    "has", "its", "just", "also", "about", "than", "would", "there",
    "some", "could", "get", "use", "one", "time", "like", "even",
    "other", "how", "any", "many", "does", "make", "need", "know",
    "way", "each", "much", "most", "used", "using", "able", "really",
    "feature", "features",  # too generic across all ERP topics
}

# Meta-commentary words — reviewers saying "no complaints" rather than naming features
META_STOP: set[str] = {
    "downsides", "downside", "complaints", "complain", "negatives", "negative",
    "honestly", "nothing", "anything", "haven", "far", "moment", "head",
    "think", "comes", "mind", "change", "thing", "great", "good", "works",
    "perfectly", "enjoyed", "deserves", "relevant",
}

# Combined set used for post-hoc keyword filtering
ALL_STOP: set[str] = PRODUCT_STOP | FUNCTION_WORDS | META_STOP


# ---------------------------------------------------------------------------
# STEP 1: LOAD AND PREPROCESS
# ---------------------------------------------------------------------------

def load_and_preprocess(input_csv: str) -> pd.DataFrame:
    """
    Load normalized CSV and add {field}_clean columns for all TEXT_FIELDS.

    Cleans: escaped newlines, whitespace, nan strings, min-length filter.
    """
    log.info("Loading %s ...", input_csv)
    df = pd.read_csv(input_csv, low_memory=False)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    for col in TEXT_FIELDS:
        if col not in df.columns:
            log.warning("Column '%s' not found — skipping", col)
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
        df[f"{col}_clean"] = cleaned.where(
            cleaned.str.len().fillna(0) >= MIN_CHARS, other=np.nan
        )
        n = df[f"{col}_clean"].notna().sum()
        log.info("  %-20s usable: %d / %d", col, n, len(df))

    return df


# ---------------------------------------------------------------------------
# STEP 2: BUILD CONFIGURED VECTORIZER (reusable)
# ---------------------------------------------------------------------------

def build_vectorizer():
    """
    Build a CountVectorizer with combined stopwords.

    This is called twice per field:
      (a) passed to BERTopic constructor for initial fitting
      (b) passed to update_topics() after reduce_outliers()

    Passing the same vectorizer instance to update_topics() is the fix for
    the v3 bug where stopwords were bypassed during keyword regeneration.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    # Combine sklearn English stopwords with our domain-specific sets
    # Use a list (not set) — CountVectorizer requires an iterable
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    combined = list(ENGLISH_STOP_WORDS) + list(PRODUCT_STOP) + list(META_STOP)

    return CountVectorizer(
        ngram_range=(1, 2),
        stop_words=combined,
        min_df=5,          # term must appear in ≥5 docs to be a keyword
        max_df=0.85,       # term appearing in >85% of docs adds no discrimination
    )


# ---------------------------------------------------------------------------
# STEP 3: POST-HOC KEYWORD CLEANING
# ---------------------------------------------------------------------------

def clean_keywords(raw_keywords: str, model, topic_id: int) -> str:
    """
    Remove stopword tokens from a topic's keyword string.

    If the top-10 keywords still contain stopwords (function words, product
    names) after CountVectorizer filtering, this fetches the next-best
    keywords from the full c-TF-IDF scores and substitutes them.

    Args:
        raw_keywords: Comma-separated keyword string from model.get_topic_info()
        model:        Fitted BERTopic instance
        topic_id:     Topic ID to look up extended keyword list

    Returns:
        Cleaned comma-separated keyword string with stopwords removed.
    """
    if topic_id == -1 or not raw_keywords:
        return raw_keywords

    # Get extended keyword list (top-30) from the model
    all_words = model.get_topic(topic_id)  # list of (word, score) tuples
    if not all_words:
        return raw_keywords

    # Filter to non-stopword terms
    clean = []
    for word, score in all_words:
        word_lower = word.lower().strip()
        # Skip if any stopword token is in the word or phrase
        is_stop = any(
            word_lower == s or word_lower.startswith(s + " ") or word_lower.endswith(" " + s)
            for s in ALL_STOP
        )
        # Skip single-character tokens and pure numbers
        if not is_stop and len(word_lower) > 2 and not word_lower.isdigit():
            clean.append(word)
        if len(clean) == 10:
            break

    return ", ".join(clean) if clean else raw_keywords


def make_label(keywords: str) -> str:
    """
    Generate a human-readable topic label from the top 3 cleaned keywords.

    BERTopic's default label format is "ID_word1_word2_word3" which is not
    readable in dashboards. This produces "word1 | word2 | word3" instead.
    """
    if not keywords or keywords == "outlier — no coherent topic":
        return "outlier"
    parts = [k.strip() for k in keywords.split(",")[:3]]
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# STEP 4: BUILD AND FIT BERTOPIC MODEL
# ---------------------------------------------------------------------------

def fit_model(docs: list[str], min_topic_size: int, nr_topics: int):
    """
    Build and fit a BERTopic model using the two-stage approach.

    Stage 1 — fit_transform():
      Embeddings → UMAP → HDBSCAN → initial topic assignments
      Vectorizer is passed here for initial c-TF-IDF keyword generation.

    Stage 2 — reduce_outliers() + update_topics():
      Reassign outlier docs to nearest topic by embedding similarity.
      update_topics() is called WITH the same vectorizer to ensure
      stopwords are applied during keyword re-generation (v3 bug fix).

    Returns: (model, topics_list)
    """
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    vectorizer = build_vectorizer()

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=10,   # 10 dims: better cluster separation, fewer outliers
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

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,   # passed here for Stage 1
        ctfidf_model=ctfidf_model,
        nr_topics=nr_topics,
        calculate_probabilities=True,
        verbose=True,
    )

    # Stage 1: cluster
    log.info("Stage 1: fit_transform (%d docs) ...", len(docs))
    topics, probs = model.fit_transform(docs)

    n_outliers_before = list(topics).count(-1)
    log.info("After clustering: %d topics, %d outliers (%.1f%%)",
             len(set(topics)) - 1, n_outliers_before,
             n_outliers_before / len(docs) * 100)

    # Stage 2: reduce outliers and regenerate keywords WITH vectorizer
    log.info("Stage 2: reduce_outliers + update_topics ...")
    try:
        new_topics = model.reduce_outliers(docs, topics, strategy="embeddings")

        # KEY FIX: pass vectorizer_model explicitly so stopwords are applied
        # during keyword regeneration — without this, update_topics() uses
        # a default vectorizer with no custom stopwords (v3 bug)
        model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer)

        topics = new_topics
        n_outliers_after = list(topics).count(-1)
        log.info("After reduce_outliers: %d outliers (%.1f%%)",
                 n_outliers_after, n_outliers_after / len(docs) * 100)
    except Exception as exc:
        log.warning("reduce_outliers failed (%s) — using original topics", exc)

    return model, topics, probs


# ---------------------------------------------------------------------------
# STEP 5: FIT AND EXTRACT OUTPUTS
# ---------------------------------------------------------------------------

def fit_and_extract(
    df: pd.DataFrame,
    field: str,
    output_dir: Path,
    save_model: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit BERTopic on one text field and save three output CSVs.

    Returns: (doc_df, info_df, dist_df)
    """
    settings = FIELD_SETTINGS[field]
    text_col = f"{field}_clean"
    sub      = df[df[text_col].notna()].copy()
    docs     = sub[text_col].tolist()

    if not docs:
        log.warning("No usable documents for '%s' — skipping", field)
        empty = pd.DataFrame()
        return empty, empty, empty

    log.info("Field [%s]: %d docs | min_topic_size=%d | nr_topics=%d",
             field, len(docs), settings["min_topic_size"], settings["nr_topics"])

    model, topics, probs = fit_model(
        docs=docs,
        min_topic_size=settings["min_topic_size"],
        nr_topics=settings["nr_topics"],
    )

    # --- Document-level output ---
    doc_df = sub[META_COLS].copy().reset_index(drop=True)
    doc_df["topic_id"]    = topics
    doc_df["probability"] = np.round(
        probs.max(axis=1) if hasattr(probs, "ndim") and probs.ndim > 1 else probs, 4
    )
    doc_df["text_field"] = field
    doc_df["text"]       = sub[text_col].values

    # --- Topic info output ---
    info = model.get_topic_info()

    rows = []
    for _, row in info.iterrows():
        tid = row["Topic"]

        # Get raw keywords from model
        raw_kw = "outlier — no coherent topic" if tid == -1 else (
            ", ".join([w for w, _ in (model.get_topic(tid) or [])[:10]])
        )

        # Apply post-hoc stopword cleaning
        clean_kw = raw_kw if tid == -1 else clean_keywords(raw_kw, model, tid)

        # Generate human-readable label
        label = make_label(clean_kw)

        # Flag: is this a product-name cluster?
        # A topic is a product cluster if its top-3 keywords are all product names
        top3 = [k.strip().lower() for k in clean_kw.split(",")[:3]]
        is_product_cluster = all(any(p in t for p in PRODUCT_STOP) for t in top3)

        rows.append({
            "topic_id":          tid,
            "doc_count":         row["Count"],
            "label":             label,
            "top_keywords":      clean_kw,
            "is_product_cluster": is_product_cluster,
        })

    info_df = pd.DataFrame(rows)

    # --- Product × topic distribution ---
    # Exclude outliers and product-name clusters from cross-product comparison
    valid_ids = info_df[
        (info_df["topic_id"] != -1) & (~info_df["is_product_cluster"])
    ]["topic_id"].tolist()

    dist_df = (
        doc_df[doc_df["topic_id"].isin(valid_ids)]
        .groupby(["product_name", "topic_id"])
        .size()
        .reset_index(name="doc_count")
        .merge(info_df[["topic_id", "label", "top_keywords"]], on="topic_id", how="left")
        .sort_values(["product_name", "doc_count"], ascending=[True, False])
    )

    # --- Save ---
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_df.to_csv(output_dir / f"{field}_topics.csv",      index=False)
    info_df.to_csv(output_dir / f"topic_info_{field}.csv", index=False)
    dist_df.to_csv(output_dir / f"topic_dist_{field}.csv", index=False)
    log.info("Saved topic_info_%s.csv (%d topics)", field, len(info_df))

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
# STEP 6: SUMMARY REPORT
# ---------------------------------------------------------------------------

def print_summary(field: str, info_df: pd.DataFrame, dist_df: pd.DataFrame) -> None:
    if info_df.empty:
        return

    total    = info_df["doc_count"].sum()
    outliers = info_df[info_df["topic_id"] == -1]["doc_count"].sum() if -1 in info_df["topic_id"].values else 0
    prod_cl  = info_df[info_df["is_product_cluster"] & (info_df["topic_id"] != -1)]
    clean    = info_df[(info_df["topic_id"] != -1) & (~info_df["is_product_cluster"])]

    print(f"\n{'='*70}")
    print(f"  BERTOPIC v4 — {field.upper()}")
    print(f"{'='*70}")
    print(f"  Feature topics:       {len(clean)}")
    print(f"  Product-name clusters:{len(prod_cl)} ({prod_cl['doc_count'].sum()} docs)")
    print(f"  Outliers:             {outliers} ({outliers/total*100:.1f}%)")
    print(f"  Total docs:           {total}")

    print(f"\n  TOP 15 FEATURE TOPICS:")
    top = clean.nlargest(15, "doc_count")
    for _, row in top.iterrows():
        print(f"    [{row['topic_id']:>3}] {row['doc_count']:>5} docs  {row['label']:<30}  {row['top_keywords'][:50]}")

    if not dist_df.empty:
        print(f"\n  DOMINANT FEATURE TOPIC PER PRODUCT:")
        top_per = dist_df.loc[dist_df.groupby("product_name")["doc_count"].idxmax()]
        for _, row in top_per.sort_values("product_name").iterrows():
            print(f"    {row['product_name']:<35} [{row['topic_id']:>3}] {row['label']}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(input_csv: str, fields: list[str], save_model: bool) -> None:
    run_start = datetime.now()
    log.info("BERTopic v4 starting — %s", run_start.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Fields: %s", fields)

    df = load_and_preprocess(input_csv)

    for field in fields:
        log.info("\n%s\n--- FIELD: %s ---\n%s", "="*55, field.upper(), "="*55)
        doc_df, info_df, dist_df = fit_and_extract(
            df=df, field=field, output_dir=OUTPUT_DIR, save_model=save_model,
        )
        print_summary(field, info_df, dist_df)

    duration = (datetime.now() - run_start).total_seconds() / 60
    log.info("Complete in %.1f min. Outputs: %s", duration, OUTPUT_DIR.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BERTopic v4 — stopword bypass fixed, clean feature topics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nlp_bertopic.py
  python nlp_bertopic.py --fields likes dislikes
  python nlp_bertopic.py --no-save-model
        """
    )
    parser.add_argument("--input",         default=INPUT_CSV)
    parser.add_argument("--fields",        nargs="+", default=TEXT_FIELDS, choices=TEXT_FIELDS)
    parser.add_argument("--no-save-model", action="store_true")

    args = parser.parse_args()
    main(input_csv=args.input, fields=args.fields, save_model=not args.no_save_model)