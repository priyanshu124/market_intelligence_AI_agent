# nlp_bertopic.py
# =============================================================================
# BERTopic pipeline — ERP Market Intelligence (v5)
#
# WHAT CHANGED FROM v4 (based on oracle_netsuite_normalized.json inspection):
#
#   1. BETTER EMBEDDING MODEL — BAAI/bge-base-en-v1.5
#      all-MiniLM-L6-v2 is general-purpose English. ERP reviews contain dense
#      domain jargon (ap/ar, gl, mrp, rev rec, job costing) that MiniLM rarely
#      saw during training — producing loose clusters. bge-base-en-v1.5 ranks
#      top on MTEB semantic similarity benchmarks, ~110MB, still CPU-viable.
#
#   2. ERP ABBREVIATION EXPANSION BEFORE EMBEDDING
#      Reviewers write "ap", "ar", "gl", "po", "mrp", "wms" without expansion.
#      The model treats these as rare tokens with weak representations.
#      Expanding to full forms ("accounts payable", "general ledger") before
#      embedding puts concepts into well-represented semantic neighbourhoods.
#
#   3. KEYBERT FOR TOPIC LABELLING (replaces c-TF-IDF labels only)
#      c-TF-IDF splits score across token variants: "invoice", "invoicing",
#      "invoices" = 3 separate signals. KeyBERT finds keyphrases closest to
#      the topic centroid — picks "invoice processing" as one concept.
#      c-TF-IDF still drives clustering. KeyBERT only runs for labelling.
#
#   4. HDBSCAN leaf CLUSTER SELECTION (was eom)
#      ERP datasets have uneven product densities (QuickBooks 3000 docs vs
#      Certinia 437). "eom" favours large dense regions, fragments small ones.
#      "leaf" gives more uniform cluster sizes across all products.
#
#   5. USE_CASE TITLE LEAKAGE FIX
#      37.8% of use_case records end with a truncated G2 review title:
#        "...seeing what is happening. Powerful Yet Pricey ERP with"
#      Pattern confirmed from data: last real sentence ends with "." then
#      title-case fragment with no terminal punctuation. Stripped before
#      embedding.
#
#   6. SWITCHED_FROM EXTRACTION FROM BONUS_DATA
#      716/2000 records have switched_from as a nested dict:
#        {"products": [{"name": "QuickBooks Online"}], "reason": "..."}
#      Extracted as switched_from_product (pipe-joined if multiple) and
#      switched_from_reason for displacement intelligence use case.
#
#   7. INCENTIVIZED REVIEW FLAGGING
#      27.6% of records are incentivized (g2gives + g2_incentivized).
#      These have positive rating bias. Flagged as is_incentivized so
#      downstream ABSA can weight accordingly.
#
#   8. RECOMMENDATIONS REMOVED FROM TEXT_FIELDS
#      99.4% null in jupri output. Running BERTopic on ~12 docs = garbage.
#      Only 3 fields now: likes, dislikes, use_case.
#
# INPUT:  data/g2/normalized/all_products_normalized.csv
# OUTPUT: data/bertopic/v5/{field}_topics.csv
#         data/bertopic/v5/topic_info_{field}.csv
#         data/bertopic/v5/topic_dist_{field}.csv
#         data/bertopic/v5/model_{field}/
#
# INSTALL:
#   pip install bertopic sentence-transformers hdbscan umap-learn keybert
#
# USAGE:
#   python nlp_bertopic.py
#   python nlp_bertopic.py --fields likes dislikes
#   python nlp_bertopic.py --no-keybert       # faster, c-TF-IDF labels only
#   python nlp_bertopic.py --no-save-model
#
# RUNTIME:
#   With KeyBERT:    ~60-75 min CPU for 3 fields
#   Without KeyBERT: ~35-45 min CPU for 3 fields
#
# PAPERS:
#   Grootendorst 2022 — BERTopic (arXiv:2203.05794)
#   Xiao et al. 2023  — BGE embeddings (arXiv:2309.07597)
#   Egger & Yu 2022   — KeyBERT (doi:10.3390/fi14110330)
# =============================================================================

from __future__ import annotations

import argparse
import json
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
OUTPUT_DIR = Path("data/bertopic/v5")

# BAAI/bge-base-en-v1.5: top MTEB semantic similarity benchmark
# ~110MB download, cached after first run
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Only 3 fields — recommendations is 99.4% null in jupri output
FIELD_SETTINGS: dict[str, dict] = {
    "likes":    {"min_topic_size": 10, "nr_topics": 60},
    "dislikes": {"min_topic_size": 10, "nr_topics": 60},
    "use_case": {"min_topic_size": 10, "nr_topics": 60},
}

TEXT_FIELDS = list(FIELD_SETTINGS.keys())

# Raised from 30 → 50: cuts more filler ("It's great, no complaints" = 30 chars)
MIN_CHARS = 50

# Incentivized review types from bonus_data.type — have positive rating bias
INCENTIVIZED_TYPES = {"g2gives", "g2_incentivized"}

META_COLS = [
    "review_id",
    "product_id",
    "product_name",
    "reviewer_company_size",
    "reviewer_industry",
    "source_platform",
    "review_date",
    "overall_rating",
    "is_incentivized",
    "switched_from_product",
    "switched_from_reason",
]


# ---------------------------------------------------------------------------
# ERP ABBREVIATION EXPANSION
# ---------------------------------------------------------------------------
# Applied BEFORE embedding so model encodes semantically rich representations.
# Patterns use word boundaries (\b) for precision.
# Listed longest/most specific first to avoid partial matches.

ERP_ABBREV_MAP: list[tuple[str, str]] = [
    # Multi-word first (most specific)
    (r"\bap aging\b",        "accounts payable aging report"),
    (r"\bar aging\b",        "accounts receivable aging report"),
    (r"\brev rec\b",         "revenue recognition"),
    (r"\basc 606\b",         "revenue recognition standard"),
    (r"\bfp&a\b",            "financial planning and analysis"),
    (r"\bfpa\b",             "financial planning and analysis"),
    # Accounting
    (r"\bap\b",              "accounts payable"),
    (r"\bar\b",              "accounts receivable"),
    (r"\bgl\b",              "general ledger"),
    (r"\bcoa\b",             "chart of accounts"),
    (r"\bp&l\b",             "profit and loss"),
    (r"\bpnl\b",             "profit and loss"),
    (r"\btb\b",              "trial balance"),
    # Procurement / Supply chain
    (r"\bpo\b",              "purchase order"),
    (r"\bpos\b",             "point of sale"),
    (r"\bso\b",              "sales order"),
    (r"\bwms\b",             "warehouse management system"),
    (r"\bmrp\b",             "material requirements planning"),
    (r"\bwip\b",             "work in progress"),
    (r"\bskus?\b",           "stock keeping unit"),
    (r"\bbom\b",             "bill of materials"),
    (r"\b3pl\b",             "third party logistics"),
    # Finance / Compliance
    (r"\bkpis?\b",           "key performance indicator"),
    (r"\broi\b",             "return on investment"),
    (r"\bgaap\b",            "generally accepted accounting principles"),
    (r"\bifrs\b",            "international financial reporting standards"),
    (r"\bsox\b",             "sarbanes oxley compliance"),
    (r"\b1099\b",            "1099 tax form"),
    (r"\bw-?2\b",            "w2 payroll form"),
    # HR
    (r"\bpto\b",             "paid time off"),
    (r"\bhris\b",            "human resources information system"),
    (r"\bhcm\b",             "human capital management"),
    # Projects / Services
    (r"\bpsa\b",             "professional services automation"),
    (r"\bsow\b",             "statement of work"),
    (r"\bcrm\b",             "customer relationship management"),
    # Tech
    (r"\bapi\b",             "application programming interface"),
    (r"\bsso\b",             "single sign on"),
    (r"\b2fa\b",             "two factor authentication"),
    (r"\bui\b",              "user interface"),
    (r"\bux\b",              "user experience"),
    (r"\bsla\b",             "service level agreement"),
    (r"\betl\b",             "data extraction transformation loading"),
]

# Pre-compile for performance (applied once per document)
_ABBREV_RE = [(re.compile(p, re.IGNORECASE), r) for p, r in ERP_ABBREV_MAP]


# ---------------------------------------------------------------------------
# STOPWORD SETS
# ---------------------------------------------------------------------------

PRODUCT_STOP: set[str] = {
    "netsuite", "net suite", "oracle",
    "sap", "s4", "s4hana", "hana", "4hana", "s/4hana", "ecc", "brf", "fiori",
    "microsoft", "dynamics", "dynamics 365", "business central", "bc",
    "gp", "navision", "nav", "d365", "365", "central",
    "quickbooks", "quickbooks online", "quickbooks enterprise", "quickbooks desktop",
    "qb", "qbo", "qbe", "qbd", "quick books", "quickbook",
    "online", "desktop", "pro", "desktop pro",
    "intuit", "intuit enterprise", "ies",
    "sage", "intacct", "sage intacct", "intaact",
    "workday", "workday financial",
    "acumatica", "aia", "var", "vars",
    "xero",
    "certinia", "financialforce", "financial force", "ffa", "ff",
    "erp", "software", "system", "platform", "tool", "product",
    "application", "solution", "program", "saas", "cloud", "cloud based",
    "books", "book", "likes", "dislikes", "dislike",
}

META_STOP: set[str] = {
    "downsides", "downside", "complaints", "complain", "negatives", "negative",
    "honestly", "nothing", "anything", "haven", "moment", "head",
    "think", "comes", "mind", "change", "thing", "great", "good",
    "perfectly", "enjoyed", "deserves", "relevant", "works",
}

ALL_STOP: set[str] = PRODUCT_STOP | META_STOP


# ---------------------------------------------------------------------------
# PREPROCESSING HELPERS
# ---------------------------------------------------------------------------

def extract_switched_from(bonus_data_val) -> tuple[str | None, str | None]:
    """
    Extract switched_from product names and reason from bonus_data.

    jupri structure:
      bonus_data.switched_from = {
        "products": [{"id": "394", "name": "QuickBooks Online", "slug": "..."}],
        "reason": "The board made us switch."
      }

    Returns:
        (pipe-joined product names, reason text)
        e.g. ("QuickBooks Online | Fishbowl Inventory", "Needed better reporting")
    """
    try:
        bd = bonus_data_val if isinstance(bonus_data_val, dict) \
             else json.loads(str(bonus_data_val))
        sf = bd.get("switched_from")
        if not sf or not isinstance(sf, dict):
            return None, None
        products = sf.get("products", [])
        names    = [p["name"] for p in products if isinstance(p, dict) and "name" in p]
        return (" | ".join(names) if names else None), (sf.get("reason") or None)
    except Exception:
        return None, None


def extract_incentivized(bonus_data_val) -> bool:
    """Return True if review was incentivized (positive rating bias)."""
    try:
        bd = bonus_data_val if isinstance(bonus_data_val, dict) \
             else json.loads(str(bonus_data_val))
        return bd.get("type", "") in INCENTIVIZED_TYPES
    except Exception:
        return False


def strip_title_leakage(text: str) -> str:
    """
    Remove G2 review title fragment appended to end of use_case text.

    Confirmed pattern from data (37.8% of use_case records affected):
      "...seeing what is happening. Powerful Yet Pricey ERP with"
    The leaked fragment: follows ". ", starts title-case, no terminal punct.
    """
    if not isinstance(text, str):
        return text
    return re.sub(r'\.\s+[A-Z][^.!?]{10,120}[^.!?\s]$', '.', text.strip())


def expand_abbreviations(text: str) -> str:
    """Expand ERP abbreviations to full forms before embedding."""
    if not isinstance(text, str):
        return text
    for pattern, replacement in _ABBREV_RE:
        text = pattern.sub(replacement, text)
    return text


def load_and_preprocess(input_csv: str) -> pd.DataFrame:
    """
    Load CSV, extract bonus_data fields, clean and expand text.

    Per record:
      1. Parse bonus_data → switched_from_product, switched_from_reason
      2. Extract is_incentivized from bonus_data.type
      3. Normalize review_date to date-only
      4. Clean text: unescape newlines, strip whitespace
      5. Strip title leakage from use_case (37.8% affected)
      6. Expand ERP abbreviations
      7. Apply MIN_CHARS length filter
    """
    log.info("Loading %s ...", input_csv)
    df = pd.read_csv(input_csv, low_memory=False)
    log.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # --- Extract from bonus_data ---
    log.info("Parsing bonus_data fields ...")
    sf_products, sf_reasons, incentivized = [], [], []
    bd_col = df.get("bonus_data", pd.Series([None] * len(df)))

    for bd in bd_col:
        p, r = extract_switched_from(bd)
        sf_products.append(p)
        sf_reasons.append(r)
        incentivized.append(extract_incentivized(bd))

    df["switched_from_product"] = sf_products
    df["switched_from_reason"]  = sf_reasons
    df["is_incentivized"]       = incentivized

    log.info("  switched_from: %d records", df["switched_from_product"].notna().sum())
    log.info("  incentivized:  %d (%.1f%%)", df["is_incentivized"].sum(),
             df["is_incentivized"].mean() * 100)

    # --- Normalize review_date ---
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(
            df["review_date"], utc=True, errors="coerce"
        ).dt.date.astype(str)

    # --- Clean and prepare each text field ---
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
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        )

        # Strip title leakage — only systematically present in use_case
        if col == "use_case":
            cleaned = cleaned.apply(
                lambda x: strip_title_leakage(x) if pd.notna(x) else x
            )

        # Expand abbreviations (expand first, then apply length filter)
        cleaned = cleaned.apply(
            lambda x: expand_abbreviations(x) if pd.notna(x) else x
        )

        # Minimum length filter
        df[f"{col}_clean"] = cleaned.where(
            cleaned.str.len().fillna(0) >= MIN_CHARS, other=np.nan
        )

        n = df[f"{col}_clean"].notna().sum()
        log.info("  %-15s usable: %d / %d (%.1f%%)",
                 col, n, len(df), n / len(df) * 100)

    return df


# ---------------------------------------------------------------------------
# VECTORIZER
# ---------------------------------------------------------------------------

def build_vectorizer():
    """
    Build CountVectorizer with combined stopwords.
    Must be passed explicitly to update_topics() — omitting it causes
    BERTopic to silently rebuild keywords with a no-stopword default.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
    combined = list(ENGLISH_STOP_WORDS) + list(PRODUCT_STOP) + list(META_STOP)
    return CountVectorizer(
        ngram_range=(1, 2),
        stop_words=combined,
        min_df=5,
        max_df=0.85,
    )


# ---------------------------------------------------------------------------
# KEYWORD CLEANING
# ---------------------------------------------------------------------------

def clean_keywords(model, topic_id: int) -> str:
    """Get top-10 clean keywords for a topic, removing residual stopwords."""
    if topic_id == -1:
        return "outlier — no coherent topic"
    clean = []
    for word, _ in (model.get_topic(topic_id) or []):
        w = word.lower().strip()
        is_stop = any(
            w == s or w.startswith(s + " ") or w.endswith(" " + s)
            for s in ALL_STOP
        )
        if not is_stop and len(w) > 2 and not w.isdigit():
            clean.append(word)
        if len(clean) == 10:
            break
    return ", ".join(clean) if clean else "unlabelled"


def make_label(keywords: str) -> str:
    """'word1 | word2 | word3' from top-3 keywords."""
    if not keywords or keywords in ("outlier — no coherent topic", "unlabelled"):
        return keywords
    return " | ".join(k.strip() for k in keywords.split(",")[:3])


# ---------------------------------------------------------------------------
# FIT MODEL
# ---------------------------------------------------------------------------

def fit_model(docs: list[str], min_topic_size: int, nr_topics: int, use_keybert: bool):
    """
    Fit BERTopic: BGE embeddings → UMAP → HDBSCAN (leaf) → c-TF-IDF
    then optionally KeyBERT for labelling.

    Two-stage:
      Stage 1: fit_transform() — clustering
      Stage 2: reduce_outliers() + update_topics(vectorizer_model=vectorizer)
               The explicit vectorizer pass is the critical v4 fix —
               without it, stopwords are silently discarded on keyword regen.
    """
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from umap import UMAP

    vectorizer     = build_vectorizer()
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=10,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    # leaf selection: more uniform cluster sizes across uneven product densities
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="leaf",
        prediction_data=True,
    )

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    # KeyBERT representation — concept-level labels, not token frequency
    representation_model = None
    if use_keybert:
        try:
            from bertopic.representation import KeyBERTInspired
            representation_model = KeyBERTInspired()
            log.info("KeyBERT representation enabled")
        except ImportError:
            log.warning("keybert not installed — pip install keybert. Falling back to c-TF-IDF.")

    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        nr_topics=nr_topics,
        calculate_probabilities=True,
        verbose=True,
    )

    # Stage 1
    log.info("Stage 1 — fit_transform (%d docs) ...", len(docs))
    topics, probs = model.fit_transform(docs)
    before = list(topics).count(-1)
    log.info("Clustering done: %d topics | %d outliers (%.1f%%)",
             len(set(topics)) - 1, before, before / len(docs) * 100)

    # Stage 2 — reduce outliers + regenerate keywords with stopword vectorizer
    log.info("Stage 2 — reduce_outliers + update_topics ...")
    try:
        new_topics = model.reduce_outliers(docs, topics, strategy="embeddings")
        # Critical: explicit vectorizer_model= so stopwords are applied
        model.update_topics(docs, topics=new_topics, vectorizer_model=vectorizer)
        topics = new_topics
        after = list(topics).count(-1)
        log.info("After reduce_outliers: %d outliers (%.1f%%)",
                 after, after / len(docs) * 100)
    except Exception as exc:
        log.warning("reduce_outliers failed (%s) — keeping original topics", exc)

    return model, topics, probs


# ---------------------------------------------------------------------------
# FIT AND EXTRACT
# ---------------------------------------------------------------------------

def fit_and_extract(
    df: pd.DataFrame,
    field: str,
    output_dir: Path,
    save_model: bool,
    use_keybert: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit BERTopic on one field, save three output CSVs."""
    settings = FIELD_SETTINGS[field]
    text_col = f"{field}_clean"
    sub      = df[df[text_col].notna()].copy().reset_index(drop=True)
    docs     = sub[text_col].tolist()

    if not docs:
        log.warning("No usable docs for '%s' — skipping", field)
        empty = pd.DataFrame()
        return empty, empty, empty

    log.info("Field [%s]: %d docs | min_topic_size=%d | nr_topics=%d",
             field, len(docs), settings["min_topic_size"], settings["nr_topics"])

    model, topics, probs = fit_model(
        docs=docs,
        min_topic_size=settings["min_topic_size"],
        nr_topics=settings["nr_topics"],
        use_keybert=use_keybert,
    )

    # Document-level output
    avail = [c for c in META_COLS if c in sub.columns]
    doc_df = sub[avail].copy()
    doc_df["topic_id"]    = topics
    doc_df["probability"] = np.round(
        probs.max(axis=1) if hasattr(probs, "ndim") and probs.ndim > 1 else probs, 4
    )
    doc_df["text_field"] = field
    doc_df["text"]       = sub[text_col].values

    # Topic info
    rows = []
    for _, row in model.get_topic_info().iterrows():
        tid   = row["Topic"]
        kw    = clean_keywords(model, tid)
        label = make_label(kw)
        top3  = [k.strip().lower() for k in kw.split(",")[:3]]
        is_prod = (tid != -1) and all(
            any(p in t for p in PRODUCT_STOP) for t in top3
        )
        rows.append({
            "topic_id":           tid,
            "doc_count":          row["Count"],
            "label":              label,
            "top_keywords":       kw,
            "is_product_cluster": is_prod,
        })
    info_df = pd.DataFrame(rows)

    # Product × topic distribution
    valid = info_df[(info_df["topic_id"] != -1) & (~info_df["is_product_cluster"])]["topic_id"].tolist()
    dist_df = (
        doc_df[doc_df["topic_id"].isin(valid)]
        .groupby(["product_name", "topic_id"]).size()
        .reset_index(name="doc_count")
        .merge(info_df[["topic_id", "label", "top_keywords"]], on="topic_id", how="left")
        .sort_values(["product_name", "doc_count"], ascending=[True, False])
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_df.to_csv(output_dir / f"{field}_topics.csv",      index=False)
    info_df.to_csv(output_dir / f"topic_info_{field}.csv", index=False)
    dist_df.to_csv(output_dir / f"topic_dist_{field}.csv", index=False)
    log.info("Saved → %s", output_dir)

    if save_model:
        model.save(str(output_dir / f"model_{field}"),
                   serialization="safetensors", save_ctfidf=True,
                   save_embedding_model=EMBEDDING_MODEL)
        log.info("Model saved → %s", output_dir / f"model_{field}")

    return doc_df, info_df, dist_df


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------

def print_summary(field: str, info_df: pd.DataFrame, dist_df: pd.DataFrame) -> None:
    if info_df.empty:
        return
    total    = info_df["doc_count"].sum()
    outliers = info_df[info_df["topic_id"] == -1]["doc_count"].sum() if -1 in info_df["topic_id"].values else 0
    prod_cl  = info_df[info_df["is_product_cluster"] & (info_df["topic_id"] != -1)]
    clean    = info_df[(info_df["topic_id"] != -1) & (~info_df["is_product_cluster"])]

    print(f"\n{'='*70}")
    print(f"  BERTopic v5 — {field.upper()}")
    print(f"{'='*70}")
    print(f"  Feature topics:        {len(clean)}")
    print(f"  Product-name clusters: {len(prod_cl)} ({prod_cl['doc_count'].sum()} docs)")
    print(f"  Outliers:              {outliers} ({outliers/total*100:.1f}%)")
    print(f"  Total docs:            {total}")
    print(f"\n  TOP 15 FEATURE TOPICS:")
    for _, row in clean.nlargest(15, "doc_count").iterrows():
        print(f"    [{row['topic_id']:>3}] {row['doc_count']:>5} docs  "
              f"{row['label']:<30}  {str(row['top_keywords'])[:45]}")
    if not dist_df.empty:
        print(f"\n  DOMINANT TOPIC PER PRODUCT:")
        top_per = dist_df.loc[dist_df.groupby("product_name")["doc_count"].idxmax()]
        for _, row in top_per.sort_values("product_name").iterrows():
            print(f"    {row['product_name']:<35} [{row['topic_id']:>3}] {row['label']}")


# ---------------------------------------------------------------------------
# MAIN / CLI
# ---------------------------------------------------------------------------

def main(input_csv, fields, save_model, use_keybert):
    t0 = datetime.now()
    log.info("BERTopic v5 | fields=%s | keybert=%s | model=%s", fields, use_keybert, EMBEDDING_MODEL)
    df = load_and_preprocess(input_csv)
    for field in fields:
        log.info("\n%s\n--- %s ---\n%s", "="*55, field.upper(), "="*55)
        doc_df, info_df, dist_df = fit_and_extract(df, field, OUTPUT_DIR, save_model, use_keybert)
        print_summary(field, info_df, dist_df)
    log.info("Done in %.1f min → %s", (datetime.now()-t0).total_seconds()/60, OUTPUT_DIR.resolve())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BERTopic v5 — BGE, KeyBERT, ERP abbrev expansion")
    p.add_argument("--input",         default=INPUT_CSV)
    p.add_argument("--fields",        nargs="+", default=TEXT_FIELDS, choices=TEXT_FIELDS)
    p.add_argument("--no-keybert",    action="store_true")
    p.add_argument("--no-save-model", action="store_true")
    args = p.parse_args()
    main(args.input, args.fields, not args.no_save_model, not args.no_keybert)
