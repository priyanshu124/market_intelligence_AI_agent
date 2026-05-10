# nlp_topic_join.py
# =============================================================================
# Join BERTopic topic assignments back to the full normalized review dataset.
#
# WHY THIS FILE EXISTS:
#   BERTopic produces a doc-level CSV ({field}_topics.csv) that contains:
#     review_id, topic_id, macro_group, probability, text
#   The full normalized review CSV contains:
#     review_id, likes, dislikes, use_case, recommendations,
#     overall_rating, reviewer_company_size, reviewer_industry,
#     reviewer_title, review_date, source_platform, product_name, ...
#
#   These two datasets need to be joined on review_id to enable:
#     - Sentiment analysis by topic (which topics have low overall_rating?)
#     - Segment analysis (which company sizes mention which topics?)
#     - Temporal trends (are certain topics growing over time?)
#     - Displacement intelligence (switching signals by topic)
#     - ABSA input (topic-tagged review text for aspect sentiment scoring)
#
# OUTPUT:
#   data/bertopic/v3/reviews_with_topics.csv
#     One row per (review × text_field). A review with all 4 fields populated
#     will appear 4 times — once per field — each with its own topic assignment.
#     This is intentional: likes and dislikes of the same review can belong to
#     different macro-groups.
#
#   data/bertopic/v3/reviews_wide.csv
#     One row per review. Each field's topic_id and macro_group as separate columns:
#     likes_topic_id, likes_macro_group, dislikes_topic_id, dislikes_macro_group, etc.
#     Easier to use for per-review analysis and dashboard filters.
#
# USAGE:
#   python nlp_topic_join.py
#   python nlp_topic_join.py --reviews data/g2/normalized/all_products_normalized.csv
#   python nlp_topic_join.py --bertopic-dir data/bertopic/v3
# =============================================================================

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("topic_join")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

REVIEWS_CSV   = "data/g2/normalized/all_products_normalized.csv"
BERTOPIC_DIR  = Path("data/bertopic/v3")
TEXT_FIELDS   = ["likes", "dislikes", "use_case", "recommendations"]

# Columns to pull from the normalized reviews CSV into the joined output.
# review_id is the join key — always included.
REVIEW_META_COLS = [
    "review_id",
    "product_id",
    "product_name",
    "source_platform",
    "review_date",
    "overall_rating",
    "ease_of_use",
    "support_rating",
    "value_rating",
    "likely_to_recommend",
    "reviewer_title",
    "reviewer_company_size",
    "reviewer_industry",
    "reviewer_location",
    "time_using_product",
    "is_verified",
    "switched_from",          # displacement intelligence
    "likes",
    "dislikes",
    "use_case",
    "recommendations",
]


# ---------------------------------------------------------------------------
# LONG FORMAT: one row per (review × field)
# ---------------------------------------------------------------------------

def build_long(reviews: pd.DataFrame, bertopic_dir: Path) -> pd.DataFrame:
    """
    Build a long-format table: one row per (review × text_field).

    For each text field, loads the doc-level topic CSV and joins it to the
    reviews. Stacks all fields vertically so the result has columns:
        review_id, text_field, topic_id, macro_group, group_label,
        probability, topic_keywords, [all review meta cols]

    A review with likes, dislikes, and use_case populated will appear 3 times.

    Use this for:
      - Topic frequency analysis per product
      - Sentiment (overall_rating) by macro-group
      - Segment breakdown (company_size × macro_group)
      - Temporal trend (review_date × macro_group)
    """
    frames = []

    for field in TEXT_FIELDS:
        doc_path = bertopic_dir / f"{field}_topics.csv"
        if not doc_path.exists():
            log.warning("Doc-level file not found: %s — skipping", doc_path)
            continue

        doc_df = pd.read_csv(doc_path)
        log.info("Loaded %s: %d rows", doc_path.name, len(doc_df))

        # Columns we want from the doc-level file
        topic_cols = ["review_id", "topic_id", "probability"]
        if "macro_group" in doc_df.columns:
            topic_cols += ["macro_group", "group_label"]
        if "top_keywords" in doc_df.columns:
            topic_cols.append("top_keywords")

        doc_df = doc_df[topic_cols].copy()
        doc_df["text_field"] = field

        doc_df["review_id"] = doc_df["review_id"].astype(str)

        # Join to full review metadata
        # Left join: keep all topic assignments, bring in review metadata where available
        merged = doc_df.merge(
            reviews[[c for c in REVIEW_META_COLS if c in reviews.columns]],
            on="review_id",
            how="left",
        )

        frames.append(merged)
        log.info("  → %d rows after join (field=%s)", len(merged), field)

    if not frames:
        raise RuntimeError("No doc-level files found in %s" % bertopic_dir)

    long_df = pd.concat(frames, ignore_index=True)
    log.info("Long format: %d rows total across %d fields", len(long_df), len(frames))
    return long_df


# ---------------------------------------------------------------------------
# WIDE FORMAT: one row per review
# ---------------------------------------------------------------------------

def build_wide(long_df: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Build a wide-format table: one row per review.

    Pivots the long format so each field's topic assignment becomes a
    separate column:
        review_id,
        likes_topic_id, likes_macro_group, likes_probability,
        dislikes_topic_id, dislikes_macro_group, dislikes_probability,
        use_case_topic_id, use_case_macro_group, use_case_probability,
        recommendations_topic_id, recommendations_macro_group, recommendations_probability,
        [all review meta cols]

    Use this for:
      - Per-review analysis (a review's likes AND dislikes topic side by side)
      - Finding reviews where likes=ease_of_use but dislikes=pricing
      - Switching signal analysis (switched_from + dislike topic)
      - ABSA input (all text fields in one row)
    """
    # Pivot topic assignments per field
    pivot_frames = []
    for field in TEXT_FIELDS:
        field_df = long_df[long_df["text_field"] == field][
            ["review_id", "topic_id", "probability"]
            + (["macro_group"] if "macro_group" in long_df.columns else [])
            + (["group_label"] if "group_label" in long_df.columns else [])
        ].copy()

        field_df = field_df.rename(columns={
            "topic_id":    f"{field}_topic_id",
            "probability": f"{field}_probability",
            "macro_group": f"{field}_macro_group",
            "group_label": f"{field}_group_label",
        })
        pivot_frames.append(field_df)

    if not pivot_frames:
        raise ValueError("No text fields available to build wide format")

    # Merge all field pivots on review_id
    wide_df = pivot_frames[0]
    for pf in pivot_frames[1:]:
        wide_df = wide_df.merge(pf, on="review_id", how="outer")

    wide_df["review_id"] = wide_df["review_id"].astype(str)
    reviews["review_id"] = reviews["review_id"].astype(str)
    
    # Join full review metadata — one row per review_id
    review_meta = reviews[[c for c in REVIEW_META_COLS if c in reviews.columns]].drop_duplicates("review_id")
    wide_df = wide_df.merge(review_meta, on="review_id", how="left")

    log.info("Wide format: %d rows (one per review)", len(wide_df))
    return wide_df


# ---------------------------------------------------------------------------
# SUMMARY STATS
# ---------------------------------------------------------------------------

def print_summary(long_df: pd.DataFrame) -> None:
    """Print coverage and group distribution stats after joining."""

    print(f"\n{'='*65}")
    print("  TOPIC → REVIEW JOIN SUMMARY")
    print(f"{'='*65}")
    print(f"  Total (review × field) rows: {len(long_df):,}")
    print(f"  Unique reviews:              {long_df['review_id'].nunique():,}")
    print(f"  Fields covered:              {long_df['text_field'].nunique()}")

    if "macro_group" not in long_df.columns:
        print("  (macro_group not present — run nlp_topic_grouping.py first)")
        return

    print(f"\n  MACRO-GROUP DISTRIBUTION (all fields combined):")
    dist = (
        long_df[~long_df["macro_group"].isin(["noise", "other", "-1"])]
        .groupby(["macro_group", "group_label"])
        .agg(
            doc_count=("review_id", "count"),
            avg_rating=("overall_rating", "mean"),
            pct_enterprise=(
                "reviewer_company_size",
                lambda x: (x == "enterprise").mean() * 100,
            ),
        )
        .reset_index()
        .sort_values("doc_count", ascending=False)
    )
    dist["avg_rating"] = dist["avg_rating"].round(2)
    dist["pct_enterprise"] = dist["pct_enterprise"].round(1)
    dist["pct"] = (dist["doc_count"] / dist["doc_count"].sum() * 100).round(1)

    print(f"\n  {'Group':<35} {'Docs':>6}  {'Pct':>5}  {'AvgRating':>9}  {'%Enterprise':>11}")
    print(f"  {'-'*70}")
    for _, row in dist.iterrows():
        print(
            f"  {row['group_label']:<35} {row['doc_count']:>6}  "
            f"{row['pct']:>4.1f}%  {row['avg_rating']:>9}  {row['pct_enterprise']:>10.1f}%"
        )

    print(f"\n  TOP 5 GROUPS BY FIELD:")
    for field in long_df["text_field"].unique():
        field_df = long_df[
            (long_df["text_field"] == field) &
            (~long_df["macro_group"].isin(["noise", "other"]))
        ]
        top = (
            field_df.groupby("group_label")["review_id"]
            .count()
            .sort_values(ascending=False)
            .head(5)
        )
        print(f"\n  {field.upper()}:")
        for group, count in top.items():
            print(f"    {group:<35} {count:>5} docs")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(reviews_csv: str, bertopic_dir: Path) -> None:
    log.info("Loading reviews from %s ...", reviews_csv)
    reviews = pd.read_csv(reviews_csv, low_memory=False)
    log.info("Reviews loaded: %d rows", len(reviews))

    # Ensure review_id is string for clean joins
    reviews["review_id"] = reviews["review_id"].astype(str)

    # Build long format
    log.info("Building long format ...")
    long_df = build_long(reviews, bertopic_dir)
    long_df["review_id"] = long_df["review_id"].astype(str)

    long_path = bertopic_dir / "reviews_with_topics.csv"
    long_df.to_csv(long_path, index=False)
    log.info("Saved long format → %s (%d rows)", long_path, len(long_df))

    # Build wide format
    log.info("Building wide format ...")
    wide_df = build_wide(long_df, reviews)

    wide_path = bertopic_dir / "reviews_wide.csv"
    wide_df.to_csv(wide_path, index=False)
    log.info("Saved wide format → %s (%d rows)", wide_path, len(wide_df))

    # Print summary
    print_summary(long_df)

    print(f"\n  Output files:")
    print(f"    {long_path}  ({len(long_df):,} rows — one per review×field)")
    print(f"    {wide_path}  ({len(wide_df):,} rows — one per review)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Join BERTopic topic assignments to the full review dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nlp_topic_join.py

  python nlp_topic_join.py \\
    --reviews data/g2/normalized/all_products_normalized.csv \\
    --bertopic-dir data/bertopic/v3
        """
    )
    parser.add_argument(
        "--reviews",
        default=REVIEWS_CSV,
        help=f"Path to normalized reviews CSV (default: {REVIEWS_CSV})",
    )
    parser.add_argument(
        "--bertopic-dir",
        default=str(BERTOPIC_DIR),
        help=f"Directory containing BERTopic output files (default: {BERTOPIC_DIR})",
    )
    args = parser.parse_args()

    main(
        reviews_csv=args.reviews,
        bertopic_dir=Path(args.bertopic_dir),
    )
