# nlp_absa.py
# =============================================================================
# Aspect-Based Sentiment Analysis (ABSA) pipeline — ERP Market Intelligence
#
# PURPOSE:
#   Score sentiment for each review's text against its LLM-assigned aspect
#   (macro_group), then aggregate to per-product × per-aspect ratings (1-5).
#
# MODEL:
#   yangheng/deberta-v3-base-absa-v1.1
#   Purpose-built ABSA model — takes (text, aspect_term) as input pair.
#   Returns positive/neutral/negative probabilities *for that specific aspect*.
#   This is fundamentally different from Cardiff which scores text alone:
#     Cardiff on "reporting is great but support takes forever":
#       → mixed (averaging both aspects into one score)
#     DeBERTa on ("reporting is great but support takes forever", "reporting"):
#       → positive 0.94
#     DeBERTa on ("reporting is great but support takes forever", "customer support"):
#       → negative 0.89
#   (Pontiki et al. 2014 — SemEval ABSA Task)
#
# PIPELINE:
#   1. Load reviews_text_classified.csv
#   2. Exclude noise topics (is_noise=True) and unassigned rows
#   3. Split each text into sentences (spaCy sentencizer)
#   4. Filter sentences relevant to the aspect via keyword overlap
#   5. Score each sentence with DeBERTa: (sentence, aspect_term) → pos/neu/neg
#   6. Convert scores to 1-5 rating: score = pos*5 + neu*3 + neg*1
#   7. Aggregate to review-level, then product-aspect level
#   8. Produce net score: (likes_rating + dislikes_rating) / 2
#   9. Output 4 CSVs + heatmap-ready pivot
#
# SENTIMENT → RATING CONVERSION:
#   rating = (positive_prob * 5) + (neutral_prob * 3) + (negative_prob * 1)
#   Range: 1.0 (fully negative) to 5.0 (fully positive)
#   Mirrors the G2 star rating scale.
#
# OUTPUT FILES (data/absa/):
#   absa_sentences.csv          — sentence-level scores (debug/audit)
#   absa_reviews.csv            — review-level aggregated scores
#   absa_product_aspect.csv     — product × aspect × field ratings
#   absa_net_scores.csv         — likes vs dislikes side by side per aspect
#   absa_heatmap.csv            — pivot: products as rows, aspects as columns
#
# INSTALL:
#   pip install pyabsa torch transformers sentencepiece
#   pip install spacy && python -m spacy download en_core_web_sm
#
# USAGE:
#   python nlp_absa.py
#   python nlp_absa.py --input data/enriched/reviews_text_classified.csv
#   python nlp_absa.py --fields likes dislikes
#   python nlp_absa.py --batch-size 32
#   python nlp_absa.py --no-sentences   # review-level only, skip sentence split
#
# RUNTIME:
#   ~2-4 hours CPU for ~54K sentences
#   ~20-40 min with GPU
#
# PAPERS:
#   Pontiki et al. 2014  — SemEval ABSA Task
#   He et al. 2022       — DeBERTa-v3 (arXiv:2111.09543)
#   Yang & Li 2023       — PyABSA (arXiv:2208.01368)
# =============================================================================

from __future__ import annotations

import argparse
import logging
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nlp_absa")


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

INPUT_CSV    = "data/enriched/reviews_text_classified.csv"
OUTPUT_DIR   = Path("data/absa")
TEXT_FIELDS  = ["likes", "dislikes"]   # use_case excluded — aspect signal weaker

# DeBERTa ABSA model — purpose-built for (text, aspect) pair scoring
ABSA_MODEL   = "yangheng/deberta-v3-base-absa-v1.1"

# Minimum sentence length to score — skip very short fragments
MIN_SENT_CHARS = 15

# Minimum keyword overlap fraction to consider a sentence relevant to aspect
# 0.0 = score all sentences regardless (slower but more complete)
# 0.1 = at least 1 keyword from top_keywords must appear in sentence
RELEVANCE_THRESHOLD = 0.0   # start with 0 — filter can be enabled via flag

# Batch size for DeBERTa inference — reduce if OOM on CPU
DEFAULT_BATCH_SIZE = 128

# Rating conversion weights: pos*HIGH + neu*MID + neg*LOW → 1-5 scale
RATING_WEIGHTS = {"positive": 5.0, "neutral": 3.0, "negative": 1.0}


# ---------------------------------------------------------------------------
# ASPECT TERM CLEANING
# ---------------------------------------------------------------------------

def clean_aspect_term(group_label: str) -> str:
    """
    Convert LLM group_label to a clean aspect term for DeBERTa.

    Examples:
      "Financial Reporting & Business Intelligence" → "financial reporting business intelligence"
      "Implementation, Setup & Cost Management"     → "implementation setup cost management"
      "Customer Support & Service Quality"          → "customer support service quality"
      "User Experience & System Usability"          → "user experience system usability"

    DeBERTa handles multi-word aspect terms well. Removing punctuation
    and lowercasing is sufficient — no further truncation needed.
    """
    if not isinstance(group_label, str):
        return "general"

    cleaned = group_label.lower()
    cleaned = re.sub(r"[&,/\\|]+", " ", cleaned)   # replace punctuation with space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()  # collapse whitespace
    return cleaned


# ---------------------------------------------------------------------------
# STEP 1: LOAD AND FILTER
# ---------------------------------------------------------------------------

def load_and_filter(input_csv: str, fields: list[str]) -> pd.DataFrame:
    """
    Load text classified dataset, filter to scoreable rows.

    Filters:
      - text_field in fields (likes, dislikes)
      - text not null
      - is_noise != True  (meta-commentary, no feature content)
      - macro_group not null and not 'unassigned'
      - macro_group not 'noise' or 'outlier'

    Adds:
      - aspect_term: cleaned group_label for DeBERTa input
    """
    log.info("Loading %s ...", input_csv)
    df = pd.read_csv(input_csv, low_memory=False)
    log.info("  Loaded: %d rows", len(df))

    # Filter to target fields
    df = df[df["text_field"].isin(fields)].copy()
    log.info("  After field filter (%s): %d rows", fields, len(df))

    # Filter null text
    df = df[df["text"].notna() & (df["text"].str.strip() != "")].copy()

    # Filter noise and unassigned
    if "is_noise" in df.columns:
        df = df[df["is_noise"].ne(True)].copy()
        log.info("  After noise filter: %d rows", len(df))

    if "macro_group" in df.columns:
        exclude_groups = {"noise", "outlier", "unassigned", None, ""}
        df = df[~df["macro_group"].isin(exclude_groups)].copy()
        df = df[df["macro_group"].notna()].copy()
        log.info("  After macro_group filter: %d rows", len(df))

    # Add clean aspect term
    if "group_label" in df.columns:
        df["aspect_term"] = df["group_label"].apply(clean_aspect_term)
    else:
        df["aspect_term"] = "general"

    log.info("  Final scoreable rows: %d", len(df))
    log.info("  Fields: %s", df["text_field"].value_counts().to_dict())
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# STEP 2: SENTENCE SPLITTING
# ---------------------------------------------------------------------------

def split_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split each review text into individual sentences.

    Uses a regex-based sentence splitter — avoids spaCy model download
    while handling most ERP review sentence patterns correctly.

    Returns long-format DataFrame: one row per (review_id, text_field, sentence).
    Original row metadata is carried through.
    """
    log.info("Splitting texts into sentences ...")

    # Sentence boundary pattern:
    # Split on . ! ? followed by whitespace and capital letter
    # Handles: "Great reports. Support is slow." → ["Great reports.", "Support is slow."]
    # Keeps: "U.S.A.", "v2.0", decimal numbers intact (requires capital after split)
    SENT_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    # Columns to carry from review level to sentence level
    carry_cols = [
        "review_id", "product_id", "product_name",
        "reviewer_company_size", "reviewer_industry",
        "overall_rating", "is_incentivized",
        "review_date", "switched_from_product",
        "text_field", "topic_id", "probability",
        "top_keywords", "label",
        "macro_group", "group_label", "aspect_term",
        "is_noise", "is_product_cluster",
    ]
    carry_cols = [c for c in carry_cols if c in df.columns]

    rows = []
    for _, review in tqdm(df.iterrows(), total=len(df), desc='Splitting sentences', unit='review'):
        text = str(review["text"]).strip()

        # Split into sentences
        sentences = SENT_PATTERN.split(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) >= MIN_SENT_CHARS]

        if not sentences:
            # Fallback: use full text as one sentence
            sentences = [text] if len(text) >= MIN_SENT_CHARS else []

        for sent_idx, sent in enumerate(sentences):
            row = {c: review[c] for c in carry_cols}
            row["sentence"]     = sent
            row["sentence_idx"] = sent_idx
            row["n_sentences"]  = len(sentences)
            rows.append(row)

    sent_df = pd.DataFrame(rows)
    log.info("  Sentences: %d (from %d reviews, avg %.1f per review)",
             len(sent_df),
             df["review_id"].nunique(),
             len(sent_df) / max(df["review_id"].nunique(), 1))
    return sent_df


# ---------------------------------------------------------------------------
# STEP 3: RELEVANCE FILTERING (optional)
# ---------------------------------------------------------------------------

def filter_relevant_sentences(
    sent_df: pd.DataFrame,
    threshold: float = RELEVANCE_THRESHOLD,
) -> pd.DataFrame:
    """
    Filter sentences to those relevant to their aspect.

    Uses keyword overlap: checks if any of the topic's top_keywords
    appear in the sentence. This ensures we only score sentences that
    actually discuss the aspect, not off-topic sentences in the same review.

    If threshold=0.0, all sentences are kept (default — more complete coverage).

    Args:
        sent_df:   Sentence-level DataFrame from split_sentences()
        threshold: Minimum fraction of keywords that must appear in sentence.
                   0.0 = no filtering, 0.1 = at least 1 of 10 keywords present.
    """
    if threshold <= 0.0:
        log.info("Relevance filtering disabled — scoring all %d sentences", len(sent_df))
        return sent_df

    log.info("Filtering sentences by keyword relevance (threshold=%.2f) ...", threshold)

    def is_relevant(row) -> bool:
        kws = str(row.get("top_keywords", "") or "")
        if not kws:
            return True   # no keywords available — keep sentence

        sent    = str(row["sentence"]).lower()
        kw_list = [k.strip().lower() for k in kws.split(",") if k.strip()]
        if not kw_list:
            return True

        matches = sum(1 for kw in kw_list if kw in sent)
        return (matches / len(kw_list)) >= threshold

    mask    = sent_df.apply(is_relevant, axis=1)
    filtered = sent_df[mask].copy()
    log.info("  Kept %d / %d sentences after relevance filter", len(filtered), len(sent_df))
    return filtered


# ---------------------------------------------------------------------------
# STEP 4: ABSA SCORING
# ---------------------------------------------------------------------------

def load_absa_model():
    """
    Load DeBERTa ABSA via transformers (PyTorch CPU).

    No ONNX, no PyABSA — pure transformers for reliability.
    Uses all available CPU threads automatically via PyTorch's
    OpenMP threading (set by torch.set_num_threads).
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    log.info("Loading DeBERTa ABSA model: %s", ABSA_MODEL)
    log.info("  (First run ~180MB download — cached after that)")

    tokenizer = AutoTokenizer.from_pretrained(ABSA_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(ABSA_MODEL)
    model.eval()

    # Use all available CPU threads
    n_threads = torch.get_num_threads()
    log.info("  CPU threads available: %d", n_threads)
    log.info("  Labels: %s", model.config.id2label)

    id2label = {k: v.lower() for k, v in model.config.id2label.items()}
    return (model, tokenizer, id2label), "transformers"



def score_batch_transformers(
    sentences: list[str],
    aspects: list[str],
    model_tokenizer: tuple,
    batch_size: int,
) -> list[dict]:
    """
    Score (sentence, aspect) pairs with DeBERTa ABSA.

    Input format: "sentence [SEP] aspect_term"
    DeBERTa was fine-tuned on this format — it scores sentiment specifically
    about the aspect_term within the sentence context.

    Returns list of dicts per sentence:
      positive, neutral, negative — probabilities (sum to 1.0)
      sentiment_label             — dominant sentiment
      absa_rating                 — 1.0-5.0 weighted score
    """
    import torch
    import torch.nn.functional as F

    model, tokenizer, id2label = model_tokenizer
    results = []

    n_batches = (len(sentences) + batch_size - 1) // batch_size
    pbar = tqdm(
        range(0, len(sentences), batch_size),
        total=n_batches,
        desc="Scoring",
        unit="batch",
        ncols=90,
    )

    for i in pbar:
        batch_sents   = sentences[i:i + batch_size]
        batch_aspects = aspects[i:i + batch_size]

        # Format: "sentence [SEP] aspect_term"
        # This is the exact format DeBERTa ABSA was trained on
        pairs = [f"{s} [SEP] {a}" for s, a in zip(batch_sents, batch_aspects)]

        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs   = F.softmax(outputs.logits, dim=-1).numpy()

        for p in probs:
            scores = {id2label[idx]: float(p[idx]) for idx in range(len(p))}
            pos    = scores.get("positive", 0.33)
            neu    = scores.get("neutral",  0.34)
            neg    = scores.get("negative", 0.33)
            rating = (pos * RATING_WEIGHTS["positive"] +
                      neu * RATING_WEIGHTS["neutral"]  +
                      neg * RATING_WEIGHTS["negative"])
            label  = max(scores, key=scores.get)
            results.append({
                "positive":        round(pos,    4),
                "neutral":         round(neu,    4),
                "negative":        round(neg,    4),
                "sentiment_label": label,
                "absa_rating":     round(rating, 3),
            })

        # Live sentiment distribution in progress bar
        n_pos = sum(1 for r in results if r["sentiment_label"] == "positive")
        n_neg = sum(1 for r in results if r["sentiment_label"] == "negative")
        pbar.set_postfix({
            "pos": f"{n_pos/len(results)*100:.0f}%",
            "neg": f"{n_neg/len(results)*100:.0f}%",
        })

    return results



def score_sentences(
    sent_df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Score all sentences and return DataFrame with sentiment columns added.
    """
    log.info("Loading ABSA model ...")
    model_obj, backend = load_absa_model()

    sentences = sent_df["sentence"].tolist()
    aspects   = sent_df["aspect_term"].tolist()

    log.info("Scoring %d (sentence, aspect) pairs | batch_size=%d",
             len(sentences), batch_size)

    results   = score_batch_transformers(sentences, aspects, model_obj, batch_size)
    scores_df = pd.DataFrame(results)
    result_df = pd.concat([sent_df.reset_index(drop=True), scores_df], axis=1)

    n = len(result_df)
    log.info("Scoring complete.")
    log.info("  Positive: %d (%.1f%%)",
             (result_df["sentiment_label"]=="positive").sum(),
             (result_df["sentiment_label"]=="positive").mean()*100)
    log.info("  Neutral:  %d (%.1f%%)",
             (result_df["sentiment_label"]=="neutral").sum(),
             (result_df["sentiment_label"]=="neutral").mean()*100)
    log.info("  Negative: %d (%.1f%%)",
             (result_df["sentiment_label"]=="negative").sum(),
             (result_df["sentiment_label"]=="negative").mean()*100)
    log.info("  Avg rating: %.2f / 5.0", result_df["absa_rating"].mean())

    return result_df


# ---------------------------------------------------------------------------
# STEP 5: AGGREGATION
# ---------------------------------------------------------------------------

def aggregate_to_reviews(sent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentence-level scores to review level.

    Per review × field: average absa_rating across all scored sentences,
    majority-vote sentiment_label, min/max sentence scores.
    """
    agg = (
        sent_df
        .groupby(["review_id", "product_id", "product_name", "text_field",
                  "macro_group", "group_label", "aspect_term",
                  "reviewer_company_size", "overall_rating", "is_incentivized",
                  "review_date", "switched_from_product"])
        .agg(
            absa_rating       = ("absa_rating",      "mean"),
            absa_rating_min   = ("absa_rating",      "min"),
            absa_rating_max   = ("absa_rating",      "max"),
            positive_avg      = ("positive",          "mean"),
            neutral_avg       = ("neutral",           "mean"),
            negative_avg      = ("negative",          "mean"),
            n_sentences       = ("sentence",          "count"),
            sentiment_label   = ("sentiment_label",   lambda x: x.value_counts().index[0]),
        )
        .reset_index()
    )
    agg["absa_rating"] = agg["absa_rating"].round(3)
    log.info("Review-level: %d rows", len(agg))
    return agg


def aggregate_to_product_aspect(review_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate review-level scores to product × aspect × field level.

    This is the primary analytical output:
      product_name × macro_group × text_field →
        avg_rating (1-5), n_reviews, n_sentences, sentiment distribution,
        avg_overall_rating (the G2 star rating for cross-validation)
    """
    agg = (
        review_df
        .groupby(["product_name", "macro_group", "group_label",
                  "aspect_term", "text_field"])
        .agg(
            avg_rating        = ("absa_rating",    "mean"),
            n_reviews         = ("review_id",      "count"),
            n_sentences       = ("n_sentences",    "sum"),
            pct_positive      = ("sentiment_label",
                                 lambda x: (x == "positive").mean() * 100),
            pct_neutral       = ("sentiment_label",
                                 lambda x: (x == "neutral").mean() * 100),
            pct_negative      = ("sentiment_label",
                                 lambda x: (x == "negative").mean() * 100),
            avg_overall_rating= ("overall_rating", "mean"),
            pct_incentivized  = ("is_incentivized",
                                 lambda x: pd.to_numeric(x, errors="coerce").mean() * 100),
        )
        .reset_index()
    )

    agg["avg_rating"]         = agg["avg_rating"].round(2)
    agg["avg_overall_rating"] = agg["avg_overall_rating"].round(2)
    agg["pct_positive"]       = agg["pct_positive"].round(1)
    agg["pct_neutral"]        = agg["pct_neutral"].round(1)
    agg["pct_negative"]       = agg["pct_negative"].round(1)
    agg["pct_incentivized"]   = agg["pct_incentivized"].round(1)

    log.info("Product-aspect: %d rows", len(agg))
    return agg


def build_net_scores(product_aspect_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build net score table: likes_rating vs dislikes_rating side by side.

    For each product × aspect:
      likes_rating    — ABSA rating from likes text (higher = more praised)
      dislikes_rating — ABSA rating from dislikes text (higher = less complained about)
      net_rating      — (likes + dislikes) / 2 — overall aspect health score
      likes_n         — number of reviews in likes
      dislikes_n      — number of reviews in dislikes

    Interpretation:
      net_rating > 4.0 = strong aspect (praised more than complained)
      net_rating 3-4   = neutral
      net_rating < 3.0 = weak aspect (significant complaints)
    """
    # Pivot likes and dislikes into columns
    likes_df = product_aspect_df[
        product_aspect_df["text_field"] == "likes"
    ][["product_name", "macro_group", "group_label", "avg_rating",
       "n_reviews", "pct_positive", "pct_negative", "avg_overall_rating"]].copy()

    dislikes_df = product_aspect_df[
        product_aspect_df["text_field"] == "dislikes"
    ][["product_name", "macro_group", "avg_rating", "n_reviews",
       "pct_positive", "pct_negative"]].copy()

    likes_df    = likes_df.rename(columns={
        "avg_rating":   "likes_rating",
        "n_reviews":    "likes_n_reviews",
        "pct_positive": "likes_pct_positive",
        "pct_negative": "likes_pct_negative",
    })
    dislikes_df = dislikes_df.rename(columns={
        "avg_rating":   "dislikes_rating",
        "n_reviews":    "dislikes_n_reviews",
        "pct_positive": "dislikes_pct_positive",
        "pct_negative": "dislikes_pct_negative",
    })

    net = likes_df.merge(dislikes_df, on=["product_name", "macro_group"], how="outer")

    # Net score: average of both directions
    net["net_rating"] = (
        (net["likes_rating"].fillna(3.0) + net["dislikes_rating"].fillna(3.0)) / 2
    ).round(2)

    # Strength signal: gap between likes praise and dislikes complaints
    net["sentiment_gap"] = (
        net["likes_rating"].fillna(3.0) - (6 - net["dislikes_rating"].fillna(3.0))
    ).round(2)
    # positive gap = praised more than complained about
    # negative gap = complained about more than praised

    net = net.sort_values(["product_name", "net_rating"], ascending=[True, False])
    log.info("Net scores: %d rows", len(net))
    return net


def build_heatmap(net_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build pivot table for dashboard heatmap.

    Rows = products, Columns = aspects, Values = net_rating (1-5)

    This is the competitive intelligence heatmap:
      - Which product wins on each aspect?
      - Where are the gaps vs Intuit products?
    """
    heatmap = net_df.pivot_table(
        index="product_name",
        columns="group_label",
        values="net_rating",
        aggfunc="mean",
    ).round(2)

    heatmap = heatmap.reset_index()
    log.info("Heatmap: %d products × %d aspects",
             len(heatmap), len(heatmap.columns) - 1)
    return heatmap


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def print_summary(net_df: pd.DataFrame, heatmap_df: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("  ABSA RESULTS SUMMARY")
    print(f"{'='*70}")

    if "group_label" in net_df.columns:
        print(f"\n  TOP ASPECTS BY NET RATING (all products avg):")
        aspect_avg = (
            net_df.groupby("group_label")["net_rating"]
            .mean()
            .sort_values(ascending=False)
        )
        for aspect, score in aspect_avg.items():
            bar = "█" * int(score * 4) + "░" * (20 - int(score * 4))
            print(f"    {aspect:<45} {score:.2f}/5  {bar}")

    if "product_name" in net_df.columns:
        print(f"\n  PRODUCT RANKINGS BY AVERAGE NET RATING:")
        prod_avg = (
            net_df.groupby("product_name")["net_rating"]
            .mean()
            .sort_values(ascending=False)
        )
        for prod, score in prod_avg.items():
            bar = "█" * int(score * 4) + "░" * (20 - int(score * 4))
            print(f"    {prod:<35} {score:.2f}/5  {bar}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(
    input_csv: str,
    fields: list[str],
    batch_size: int,
    use_sentences: bool,
    relevance_threshold: float,
) -> None:
    # Use field-specific subdirectory when running a single field
    # so likes and dislikes outputs don't overwrite each other
    global OUTPUT_DIR
    if len(fields) == 1:
        OUTPUT_DIR = Path("data/absa") / fields[0]
        log.info("Single-field run — outputs in: %s", OUTPUT_DIR)
    else:
        OUTPUT_DIR = Path("data/absa") / "combined"
        log.info("Multi-field run — outputs in: %s", OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    t_total = datetime.now()

    # Step 1: load and filter
    log.info("Step 1/7 — Load and filter")
    df = load_and_filter(input_csv, fields)

    # Step 2: sentence split (or use full text)
    log.info("Step 2/7 — Sentence splitting")
    if use_sentences:
        score_input = split_sentences(df)
        score_input = filter_relevant_sentences(score_input, relevance_threshold)
    else:
        log.info("Sentence splitting disabled — scoring full review texts")
        score_input = df.copy()
        score_input["sentence"]     = score_input["text"]
        score_input["sentence_idx"] = 0
        score_input["n_sentences"]  = 1

    # Step 3: score with DeBERTa ABSA
    log.info("Step 3/7 — ABSA scoring (~54K sentences)")
    scored = score_sentences(score_input, batch_size=batch_size)

    # Save sentence-level output
    sent_path = OUTPUT_DIR / "absa_sentences.csv"
    scored.to_csv(sent_path, index=False)
    log.info("Saved -> %s", sent_path)

    # Step 4: aggregate to review level
    log.info("Step 4/7 — Aggregate to review level")
    reviews_agg = aggregate_to_reviews(scored)
    rev_path = OUTPUT_DIR / "absa_reviews.csv"
    reviews_agg.to_csv(rev_path, index=False)
    log.info("Saved -> %s", rev_path)

    # Step 5: aggregate to product × aspect
    log.info("Step 5/7 — Aggregate to product x aspect")
    product_aspect = aggregate_to_product_aspect(reviews_agg)
    pa_path = OUTPUT_DIR / "absa_product_aspect.csv"
    product_aspect.to_csv(pa_path, index=False)
    log.info("Saved -> %s", pa_path)

    # Step 6: net scores (likes vs dislikes)
    log.info("Step 6/7 — Build net scores")
    net_scores = build_net_scores(product_aspect)
    net_path = OUTPUT_DIR / "absa_net_scores.csv"
    net_scores.to_csv(net_path, index=False)
    log.info("Saved -> %s", net_path)

    # Step 7: heatmap pivot
    log.info("Step 7/7 — Build heatmap")
    heatmap = build_heatmap(net_scores)
    hm_path = OUTPUT_DIR / "absa_heatmap.csv"
    heatmap.to_csv(hm_path, index=False)
    log.info("Saved -> %s", hm_path)

    print_summary(net_scores, heatmap)

    elapsed = (datetime.now() - t_total).total_seconds() / 60
    log.info("Total runtime: %.1f minutes", elapsed)
    log.info("All outputs in: %s", OUTPUT_DIR.resolve())
    log.info("Output files:")
    log.info("  absa_sentences.csv      — sentence-level scores (audit trail)")
    log.info("  absa_reviews.csv        — review-level aggregated scores")
    log.info("  absa_product_aspect.csv — product x aspect x field ratings")
    log.info("  absa_net_scores.csv     — likes vs dislikes side by side")
    log.info("  absa_heatmap.csv        — dashboard heatmap pivot")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="ABSA pipeline — DeBERTa aspect sentiment → 1-5 ratings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Install:
  pip install pyabsa transformers torch sentencepiece

Examples:
  python nlp_absa.py
  python nlp_absa.py --fields likes dislikes
  python nlp_absa.py --batch-size 32
  python nlp_absa.py --no-sentences     # skip sentence split, score full text
  python nlp_absa.py --relevance 0.1    # only score sentences with keyword match
        """
    )
    p.add_argument("--input",        default=INPUT_CSV)
    p.add_argument("--fields",       nargs="+", default=TEXT_FIELDS,
                   choices=["likes", "dislikes", "use_case"])
    p.add_argument("--batch-size",   type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--no-sentences", action="store_true",
                   help="Score full review text instead of splitting into sentences")
    p.add_argument("--relevance",    type=float, default=RELEVANCE_THRESHOLD,
                   help="Keyword relevance threshold 0.0-1.0 (0=off)")
    args = p.parse_args()

    main(
        input_csv=args.input,
        fields=args.fields,
        batch_size=args.batch_size,
        use_sentences=not args.no_sentences,
        relevance_threshold=args.relevance,
    )