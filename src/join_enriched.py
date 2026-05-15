# join_enriched.py
# =============================================================================
# Build two clean analysis-ready datasets from normalized reviews +
# BERTopic Layer 1 + LLM Layer 2 enrichments.
#
# OUTPUT 1 — reviews_meta.csv  (one row per review)
#   All reviewer and product metadata. No text, no classification.
#   Use for: segment analysis, rating trends, displacement tracking,
#             temporal analysis, incentive bias detection.
#
# OUTPUT 2 — reviews_text_classified.csv  (one row per review x field)
#   All text fields with full BERTopic + LLM classification. Long format.
#   Use for: ABSA, topic frequency, sentiment heatmap, competitive intel.
#
# DROPPED COLUMNS (with reasons):
#   ease_of_use, support_rating, value_rating, likely_to_recommend  -- 100% null (jupri)
#   time_using_product, is_verified                                  -- 100% null (jupri)
#   product_overall_rating, product_review_count                     -- 100% null (jupri)
#   bonus_helpful                                                    -- 99% null
#   scraped_at, extraction_confidence, source_actor                  -- pipeline internals
#   bonus_primary                                                    -- dup of bonus_region
#   bonus_updated                                                    -- dup of bonus_submitted
#   recommendations                                                  -- 81% null (jupri)
#
# USAGE:
#   python join_enriched.py
#   python join_enriched.py --reviews data/g2/normalized/all_products_normalized.csv
#   python join_enriched.py --bertopic-dir data/bertopic/v5
#   python join_enriched.py --no-parquet
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("join_enriched")

REVIEWS_CSV  = "data/g2/normalized/all_products_normalized.csv"
BERTOPIC_DIR = Path("data/bertopic/v5")
OUTPUT_DIR   = Path("data/enriched")
TEXT_FIELDS  = ["likes", "dislikes", "use_case"]

NULL_COLS_TO_DROP = [
    "ease_of_use", "support_rating", "value_rating", "likely_to_recommend",
    "time_using_product", "is_verified",
    "product_overall_rating", "product_review_count",
]

PIPELINE_COLS_TO_DROP = [
    "scraped_at", "extraction_confidence", "source_actor",
    "bonus_primary", "bonus_updated", "bonus_helpful", "recommendations",
]


# ---------------------------------------------------------------------------
# BONUS_DATA EXTRACTION
# ---------------------------------------------------------------------------

def extract_bonus_fields(bonus_val) -> dict:
    """
    Flatten bonus_data JSON into individual named columns.

    Confirmed fields (oracle_netsuite_normalized.json, 2000 records):
      title     -> review_title       (98.7% coverage)
      type      -> review_type        (100% — organic/g2gives/vendor/etc.)
      region    -> reviewer_region    (99.7%)
      url       -> review_url         (100%)
      submitted -> review_submitted_at(100%)
      switched_from -> switched_from_product + switched_from_reason (14%)

    Dropped: primary (dup of region), updated (dup of submitted), helpful (99% null)
    """
    defaults = {
        "review_title":          None,
        "review_type":           None,
        "reviewer_region":       None,
        "review_url":            None,
        "review_submitted_at":   None,
        "switched_from_product": None,
        "switched_from_reason":  None,
        "is_incentivized":       False,
    }

    if not bonus_val:
        return defaults

    try:
        bd = bonus_val if isinstance(bonus_val, dict) else json.loads(str(bonus_val))
    except (json.JSONDecodeError, TypeError):
        return defaults

    incentivized_types = {"g2gives", "g2_incentivized"}

    result = {
        "review_title":        bd.get("title"),
        "review_type":         bd.get("type"),
        "reviewer_region":     bd.get("region"),
        "review_url":          bd.get("url"),
        "review_submitted_at": bd.get("submitted"),
        "is_incentivized":     bd.get("type", "") in incentivized_types,
    }

    sf       = bd.get("switched_from")
    products = []
    reason   = None
    if isinstance(sf, dict):
        products = sf.get("products") or []
        reason   = sf.get("reason") or None
    elif isinstance(sf, list):
        products = sf

    names = [p["name"] for p in products if isinstance(p, dict) and "name" in p]
    result["switched_from_product"] = " | ".join(names) if names else None
    result["switched_from_reason"]  = reason

    return result


# ---------------------------------------------------------------------------
# STEP 1: LOAD AND CLEAN BASE REVIEWS
# ---------------------------------------------------------------------------

def load_reviews(path: str) -> pd.DataFrame:
    """
    Load normalized CSV, expand bonus_data, drop all useless columns.
    Returns clean base DataFrame — one row per review.
    """
    log.info("Loading %s ...", path)
    df = pd.read_csv(path, low_memory=False)
    log.info("  Raw: %d rows, %d columns", len(df), len(df.columns))

    df["review_id"] = df["review_id"].astype(str)

    if "review_date" in df.columns:
        df["review_date"] = (
            pd.to_datetime(df["review_date"], utc=True, errors="coerce")
            .dt.date.astype(str)
            .replace("NaT", None)
        )

    # Drop 100%-null columns
    null_drop = [c for c in NULL_COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=null_drop)

    # Expand bonus_data
    if "bonus_data" in df.columns:
        bonus_df = pd.DataFrame(df["bonus_data"].apply(extract_bonus_fields).tolist())
        existing = [c for c in bonus_df.columns if c in df.columns]
        df = pd.concat([df.drop(columns=["bonus_data"] + existing), bonus_df], axis=1)
        log.info("  bonus_data expanded: %d new columns", len(bonus_df.columns))
    else:
        for col in ["review_title","review_type","reviewer_region","review_url",
                    "review_submitted_at","switched_from_product",
                    "switched_from_reason","is_incentivized"]:
            df[col] = None

    # Drop low-value pipeline columns
    pipeline_drop = [c for c in PIPELINE_COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=pipeline_drop)

    log.info("  Clean base: %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# STEP 2: DATASET 1 — REVIEWS METADATA
# ---------------------------------------------------------------------------

META_COLS_ORDERED = [
    "review_id",
    "product_id",
    "product_name",
    "source_platform",
    "review_date",
    "review_submitted_at",
    "overall_rating",
    "reviewer_name",
    "reviewer_title",
    "reviewer_company_size",
    "reviewer_industry",
    "reviewer_location",
    "reviewer_region",
    "review_title",
    "review_type",
    "is_incentivized",
    "review_url",
    "switched_from_product",
    "switched_from_reason",
]


def build_meta(reviews: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in META_COLS_ORDERED if c in reviews.columns]
    meta      = reviews[available].drop_duplicates("review_id").copy()
    log.info("Dataset 1 (meta): %d rows, %d columns", len(meta), len(meta.columns))
    return meta


# ---------------------------------------------------------------------------
# STEP 3: DATASET 2 — TEXT + CLASSIFICATION
# ---------------------------------------------------------------------------

TEXT_CONTEXT_COLS = [
    "review_id", "product_id", "product_name",
    "reviewer_company_size", "reviewer_industry",
    "overall_rating", "is_incentivized",
    "review_date", "switched_from_product",
]

TEXT_COL_ORDER = [
    "review_id", "product_id", "product_name",
    "reviewer_company_size", "reviewer_industry",
    "overall_rating", "is_incentivized",
    "review_date", "switched_from_product",
    "text_field", "text",
    "topic_id", "probability",
    "top_keywords", "label",
    "macro_group", "group_label",
    "plain_description",
    "is_noise", "is_product_cluster",
]


def build_text_classified(reviews: pd.DataFrame, bertopic_dir: Path) -> pd.DataFrame:
    """
    Build long-format text + classification dataset.

    For each field:
      1. Filter to rows where field has text
      2. Join topic_id + probability from {field}_topics.csv (Layer 1)
      3. Join top_keywords, label, macro_group, group_label,
         plain_description, is_noise, is_product_cluster
         from topic_info_{field}.csv (Layer 2)
      4. Stack all fields vertically
    """
    frames = []

    for field in TEXT_FIELDS:
        if field not in reviews.columns:
            log.warning("'%s' not in reviews — skipping", field)
            continue

        sub = reviews[reviews[field].notna()].copy()
        if sub.empty:
            continue

        ctx   = [c for c in TEXT_CONTEXT_COLS if c in sub.columns]
        frame = sub[ctx].copy()
        frame["text_field"] = field
        frame["text"]       = sub[field].values

        # Join Layer 1: topic_id + probability
        doc_path = bertopic_dir / f"{field}_topics.csv"
        if doc_path.exists():
            doc_df = pd.read_csv(
                doc_path,
                usecols=lambda c: c in ["review_id", "topic_id", "probability"]
            )
            doc_df["review_id"] = doc_df["review_id"].astype(str)
            if "probability" in doc_df.columns:
                doc_df = doc_df.sort_values("probability", ascending=False).drop_duplicates("review_id")
            else:
                doc_df = doc_df.drop_duplicates("review_id")
            frame = frame.merge(doc_df, on="review_id", how="left")
            log.info("  [%-10s] Layer 1 joined: %d rows, topic coverage: %d",
                     field, len(frame), frame["topic_id"].notna().sum())
        else:
            log.warning("  Layer 1 not found: %s", doc_path)
            frame["topic_id"]   = None
            frame["probability"] = None

        # Join Layer 2: all classification + c-TF-IDF fields
        info_path = bertopic_dir / f"topic_info_{field}.csv"
        if info_path.exists() and "topic_id" in frame.columns:
            info_df = pd.read_csv(info_path)
            info_df = info_df.drop(columns=["doc_count"], errors="ignore")

            frame["topic_id"]    = pd.to_numeric(frame["topic_id"],    errors="coerce").astype("Int64")
            info_df["topic_id"]  = pd.to_numeric(info_df["topic_id"],  errors="coerce").astype("Int64")

            frame = frame.merge(info_df, on="topic_id", how="left")
            n_kw  = frame["top_keywords"].notna().sum() if "top_keywords" in frame.columns else 0
            n_grp = frame["macro_group"].notna().sum()  if "macro_group"  in frame.columns else 0
            log.info("  [%-10s] Layer 2 joined: keywords: %d | grouped: %d",
                     field, n_kw, n_grp)
        else:
            log.warning("  Layer 2 not found: %s", info_path)

        frames.append(frame)

    if not frames:
        raise RuntimeError("No text fields produced rows")

    long_df = pd.concat(frames, ignore_index=True)

    ordered   = [c for c in TEXT_COL_ORDER if c in long_df.columns]
    remaining = [c for c in long_df.columns if c not in TEXT_COL_ORDER]
    long_df   = long_df[ordered + remaining]

    log.info("Dataset 2 (text+classified): %d rows, %d columns",
             len(long_df), len(long_df.columns))
    return long_df


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def print_summary(meta: pd.DataFrame, text: pd.DataFrame) -> None:
    print(f"\n{'='*65}")
    print("  OUTPUT DATASETS")
    print(f"{'='*65}")

    print(f"\n  1. reviews_meta.csv  —  {len(meta):,} reviews x {len(meta.columns)} columns")
    for col in meta.columns:
        null_pct = meta[col].isna().mean() * 100
        print(f"     {col:<35} {null_pct:>5.1f}% null")

    print(f"\n  2. reviews_text_classified.csv  —  {len(text):,} rows x {len(text.columns)} columns")
    for col in text.columns:
        null_pct = text[col].isna().mean() * 100
        print(f"     {col:<35} {null_pct:>5.1f}% null")

    print(f"\n  Field breakdown:")
    for field, grp in text.groupby("text_field"):
        n_grp  = grp["macro_group"].notna().sum()  if "macro_group"  in grp.columns else 0
        n_kw   = grp["top_keywords"].notna().sum() if "top_keywords" in grp.columns else 0
        n_noise= (grp["is_noise"] == True).sum()   if "is_noise"    in grp.columns else 0
        print(f"     {field:<12} {len(grp):>6,} rows | "
              f"grouped: {n_grp:,} | keywords: {n_kw:,} | noise: {n_noise:,}")

    if "group_label" in text.columns:
        print(f"\n  Top 10 groups in dislikes (pain points):")
        dis = text[(text["text_field"] == "dislikes") & (text["is_noise"].ne(True))]
        for label, count in dis["group_label"].value_counts().head(10).items():
            print(f"     {label:<45} {count:>5,}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(reviews_csv: str, bertopic_dir: Path, write_parquet: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reviews = load_reviews(reviews_csv)

    log.info("Building Dataset 1 ...")
    meta = build_meta(reviews)
    meta.to_csv(OUTPUT_DIR / "reviews_meta.csv", index=False)
    log.info("Saved -> %s", OUTPUT_DIR / "reviews_meta.csv")

    log.info("Building Dataset 2 ...")
    text = build_text_classified(reviews, bertopic_dir)
    text.to_csv(OUTPUT_DIR / "reviews_text_classified.csv", index=False)
    log.info("Saved -> %s", OUTPUT_DIR / "reviews_text_classified.csv")

    if write_parquet:
        try:
            meta.to_parquet(OUTPUT_DIR / "reviews_meta.parquet",            index=False)
            text.to_parquet(OUTPUT_DIR / "reviews_text_classified.parquet", index=False)
            log.info("Parquet versions saved")
        except ImportError:
            log.warning("pyarrow not installed — skip parquet. Run: pip install pyarrow")

    print_summary(meta, text)
    log.info("Done. Outputs in: %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build two clean analysis datasets")
    p.add_argument("--reviews",      default=REVIEWS_CSV)
    p.add_argument("--bertopic-dir", default=str(BERTOPIC_DIR))
    p.add_argument("--no-parquet",   action="store_true")
    args = p.parse_args()
    main(args.reviews, Path(args.bertopic_dir), not args.no_parquet)