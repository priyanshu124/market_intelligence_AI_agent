# g2_config.py
# =============================================================================
# Single source of truth for all Apify actors, ERP products, input schemas,
# output field mappings, and scrape strategy.
#
# DESIGN PRINCIPLE:
#   Nothing is hardcoded in the scraper or normalizer. Every actor difference
#   (input param names, output field names, pricing model) lives here.
#   To add a new actor: add one block to ACTOR_CONFIGS.
#   To add a new product: add one block to PRODUCTS.
#   The scraper and normalizer read this config at runtime — they never need
#   to know which actor they're talking to.
#
# CONFIRMED ACTORS (April 2026):
#   jupri         — PRIMARY G2 review scraper. Compute-only pricing. Proven.
#   samstorm      — G2 + Capterra dual-source. Per-result pricing. Sub-ratings.
#   focused_vanguard — Multi-platform (G2, Capterra, Gartner). Domain-based.
#
# SECTION MAP:
#   1. ERP Products
#   2. Actor Configs
#   3. Scrape Strategy
#   4. Unified Schema
#   5. Output Paths
#   6. Apify Settings
# =============================================================================

from __future__ import annotations
from typing import Any


# ---------------------------------------------------------------------------
# 1. ERP PRODUCTS
# ---------------------------------------------------------------------------
# Internal product keys are stable IDs — never change them once data is written.
# g2_slug:   used as actor input (jupri `query`, samstorm URL component)
# capterra_slug: used by samstorm for Capterra URL construction
# domain:    used by focused_vanguard auto-discovery mode
# source_platform is injected at normalisation time — not stored here.

PRODUCTS: dict[str, dict[str, Any]] = {

    "oracle_netsuite": {
        "product_id":      "ORC-NS-001",
        "name":            "Oracle NetSuite",
        "vendor":          "Oracle",
        "tier":            "enterprise",
        "g2_slug":         "netsuite",
        "capterra_slug":   "netsuite",
        "domain":          "netsuite.com",
        "g2_url":          "https://www.g2.com/products/netsuite/reviews",
        "capterra_url":    "https://www.capterra.com/p/22580/NetSuite/",
    },

    "sap_cloud_erp": {
        "product_id":      "SAP-S4-001",
        "name":            "SAP S/4HANA Cloud",
        "vendor":          "SAP",
        "tier":            "enterprise",
        "g2_slug":         "sap-cloud-erp-sap-s-4hana-cloud",
        "capterra_slug":   "sap-s4hana",
        "domain":          "sap.com",
        "g2_url":          "https://www.g2.com/products/sap-cloud-erp-sap-s-4hana-cloud/reviews",
        "capterra_url":    "https://www.capterra.com/p/152293/SAP-S-4HANA/",
    },

    "microsoft_dynamics_365": {
        "product_id":      "MSFT-D365-001",
        "name":            "Microsoft Dynamics 365",
        "vendor":          "Microsoft",
        "tier":            "enterprise",
        "g2_slug":         "microsoft-microsoft-dynamics-365-business-central",
        "capterra_slug":   "microsoft-dynamics-365",
        "domain":          "microsoft.com",
        "g2_url":          "https://www.g2.com/products/microsoft-microsoft-dynamics-365-business-central/reviews",
        "capterra_url":    "https://www.capterra.com/p/55711/Microsoft-Dynamics-365/",
    },

    "quickbooks_enterprise": {
        "product_id":      "QB-ENT-001",
        "name":            "QuickBooks Enterprise",
        "vendor":          "Intuit",
        "tier":            "smb",
        "g2_slug":         "quickbooks-desktop-enterprise",
        "capterra_slug":   "quickbooks-enterprise",
        "domain":          "quickbooks.intuit.com",
        "g2_url":          "https://www.g2.com/products/quickbooks-desktop-enterprise/reviews",
        "capterra_url":    "https://www.capterra.com/p/14445/QuickBooks-Enterprise/",
    },

    "quickbooks_online": {
        "product_id":      "QB-ONLINE-001",
        "name":            "QuickBooks Online",
        "vendor":          "Intuit",
        "tier":            "smb",
        "g2_slug":         "quickbooks-online",
        "capterra_slug":   "quickbooks-online",
        "domain":          "quickbooks.intuit.com",
        "g2_url":          "https://www.g2.com/products/quickbooks-online/reviews",
        "capterra_url":    "https://www.capterra.com/p/146255/QuickBooks-Online/",
    },

    "quickbooks_desktop_pro": {
        "product_id":      "QB-DESK-PRO-001",
        "name":            "QuickBooks Desktop Pro",
        "vendor":          "Intuit",
        "tier":            "smb",
        "g2_slug":         "quickbooks-desktop-pro",
        "capterra_slug":   "quickbooks-desktop-pro",
        "domain":          "quickbooks.intuit.com",
        "g2_url":          "https://www.g2.com/products/quickbooks-desktop-pro/reviews",
        "capterra_url":    "https://www.capterra.com/p/140791/QuickBooks-Desktop-Pro/",
    },

    "intuit_enterprise_suite": {
        "product_id":      "INT-IES-001",
        "name":            "Intuit Enterprise Suite",
        "vendor":          "Intuit",
        "tier":            "smb",
        "g2_slug":         "intuit-intuit-enterprise-suite",
        "capterra_slug":   "intuit-enterprise-suite",
        "domain":          "intuit.com",
        "g2_url":          "https://www.g2.com/products/intuit-intuit-enterprise-suite/reviews",
        "capterra_url":    "https://www.capterra.com/p/intuit-enterprise-suite/",
    },

    "sage_intacct": {
        "product_id":      "SGE-INT-001",
        "name":            "Sage Intacct",
        "vendor":          "Sage",
        "tier":            "smb",
        "g2_slug":         "sage-intacct",
        "capterra_slug":   "sage-intacct",
        "domain":          "sageintacct.com",
        "g2_url":          "https://www.g2.com/products/sage-intacct/reviews",
        "capterra_url":    "https://www.capterra.com/p/19165/Sage-Intacct/",
    },

    "workday_financial": {
        "product_id":      "WD-FIN-001",
        "name":            "Workday Financial Management",
        "vendor":          "Workday",
        "tier":            "enterprise",
        "g2_slug":         "workday-financial-management",
        "capterra_slug":   "workday-financial-management",
        "domain":          "workday.com",
        "g2_url":          "https://www.g2.com/products/workday-financial-management/reviews",
        "capterra_url":    "https://www.capterra.com/p/64796/Workday-Financial-Management/",
    },

    "acumatica": {
        "product_id":      "ACU-ERP-001",
        "name":            "Acumatica",
        "vendor":          "Acumatica",
        "tier":            "smb",
        "g2_slug":         "acumatica",
        "capterra_slug":   "acumatica",
        "domain":          "acumatica.com",
        "g2_url":          "https://www.g2.com/products/acumatica/reviews",
        "capterra_url":    "https://www.capterra.com/p/112156/Acumatica/",
    },

    "xero": {
        "product_id":      "XRO-ACC-001",
        "name":            "Xero",
        "vendor":          "Xero",
        "tier":            "smb",
        "g2_slug":         "xero",
        "capterra_slug":   "xero",
        "domain":          "xero.com",
        "g2_url":          "https://www.g2.com/products/xero/reviews",
        "capterra_url":    "https://www.capterra.com/p/60849/Xero/",
    },

    "certinia": {
        "product_id":      "CRT-ERP-001",
        "name":            "Certinia",
        "vendor":          "Certinia",
        "tier":            "enterprise",
        "g2_slug":         "certinia-financial-management-cloud",
        "capterra_slug":   "certinia",
        "domain":          "certinia.com",
        "g2_url":          "https://www.g2.com/products/certinia-financial-management-cloud/reviews",
        "capterra_url":    "https://www.capterra.com/p/certinia/",
    },
}


# ---------------------------------------------------------------------------
# 2. ACTOR CONFIGS
# ---------------------------------------------------------------------------
# Each actor entry has four sections:
#   actor_id     — Apify actor identifier (user/actor-name)
#   pricing      — cost model for budget awareness
#   input_schema — exact param names this actor accepts (used by build_input())
#   field_map    — {unified_field: actor_output_field} (used by normalizer)
#
# Unified field names are defined in UNIFIED_SCHEMA (section 4).
# If an actor doesn't produce a field, map it to None.
# Dot notation = nested path:  "location.country" → raw["location"]["country"]
# Integer parts = array index: "answers.0"        → raw["answers"][0]

ACTOR_CONFIGS: dict[str, dict[str, Any]] = {

    # -------------------------------------------------------------------------
    # ACTOR 1: jupri/g2-explorer
    # Role: PRIMARY G2 review scraper.
    # Compute-only pricing — fits within Apify free tier.
    # 3 operating modes: "review" | "product" | "product_info"
    # Confirmed real output schema from live run (March 2026).
    # Limitations: no verified/incentivized flags, no sub-ratings.
    # -------------------------------------------------------------------------
    "jupri": {
        "actor_id": "jupri/g2-explorer",
        "label":    "jupri/g2-explorer",
        "source_platform": "g2",                   # this actor only scrapes G2

        "pricing": {
            "model":            "compute_only",
            "cost_per_1k":      None,
            "estimated_per_1k": 1.50,              # rough $ based on compute units
            "free_tier_viable": True,
        },

        # Input params confirmed from official docs (March 2026).
        # query accepts the G2 product slug directly e.g. "netsuite".
        # limit: max reviews to return — if multiple slugs passed, limit is per slug.
        # No proxy injection — jupri manages its own residential proxy pool internally.
        "input_schema": {
            "query": "{g2_slug}",
            "limit": 2000,
        },

        # Field map confirmed from real jupri output (March 2026):
        # {
        #   "id": "6996026",
        #   "answers": ["likes text", "dislikes text", "recommendations", "use_case"],
        #   "score": 5,
        #   "name": "Lucjan K.",
        #   "role": "User",
        #   "segment": "small-business",
        #   "industry": "Computer Software",
        #   "location": {"country": "Poland", "primary": "EMEA", "region": "Europe"},
        #   "date": {"published": "2022-08-18T09:05:00.496-05:00", ...},
        #   "product_id": 74085,
        #   "product_slug": "apify",
        #   "switched_from": null,
        #   "helpful": 0,
        #   "url": "https://...",
        #   "type": "text"
        # }
        # NOTE: answers is an ARRAY of 4 strings — indexed by position.
        # The normalizer uses dot-integer notation to index into arrays.
        "field_map": {
            "review_id":              "id",
            "likes":                  "answers.0",         # What do you like best?
            "dislikes":               "answers.1",         # What do you dislike?
            "recommendations":        "answers.2",         # Recommendations to others
            "use_case":               "answers.3",         # What problems is it solving?
            "overall_rating":         "score",
            "ease_of_use":            None,                # jupri doesn't extract sub-ratings
            "support_rating":         None,
            "value_rating":           None,
            "likely_to_recommend":    None,
            "reviewer_name":          "name",
            "reviewer_title":         "role",
            "reviewer_company_size":  "segment",
            "reviewer_industry":      "industry",
            "reviewer_location":      "location.country",
            "time_using_product":     None,
            "review_date":            "date.published",
            "is_verified":            None,
            "is_incentivized":        None,
            "product_overall_rating": None,
            "product_review_count":   None,
        },

        # Fields that don't fit the unified schema but are worth preserving.
        # The normalizer collects these into a "bonus_data" dict on each record.
        "bonus_fields": [
            "title",
            "date.submitted",
            "date.updated",
            "helpful",
            "url",
            "source.type",
            "switched_from",
            "product_slug",
            "product_id",
            "location.region",
            "location.primary",
        ],
    },

    # -------------------------------------------------------------------------
    # ACTOR 2: samstorm/g2-capterra-review-scraper
    # Role: DUAL-SOURCE scraper — runs G2 and Capterra in one actor call.
    # Key advantage: Capterra sub-ratings (Value, Support, Features, Ease, LTR).
    # These are the numerics the Capterra agent uses for quantitative analysis.
    # Pricing: per-result. More expensive than jupri but provides sub-ratings.
    # Use for Capterra pass; use jupri for G2 pass (cheaper + more G2 fields).
    # -------------------------------------------------------------------------
    "samstorm": {
        "actor_id": "samstorm/g2-capterra-review-scraper",
        "label":    "samstorm/g2-capterra",
        "source_platform": "capterra",             # primary use case is Capterra sub-ratings

        "pricing": {
            "model":            "pay_per_result",
            "cost_per_1k":      5.00,              # approximate
            "estimated_per_1k": 5.00,
            "free_tier_viable": False,
        },

        # startUrls: list of G2 or Capterra review page URLs.
        # maxReviews: cap per URL.
        # platform is auto-detected from URL — no explicit param needed.
        "input_schema": {
            "startUrls": ["{capterra_url}"],        # pass Capterra URL for sub-ratings
            "maxReviews": 500,
        },

        # Capterra output schema (approximate — verify against live output):
        # {
        #   "reviewId": "abc123",
        #   "pros": "What I liked...",
        #   "cons": "What I didn't...",
        #   "overallRating": 4.5,
        #   "easeOfUse": 4.0,
        #   "customerService": 3.5,
        #   "valueForMoney": 4.0,
        #   "likelihood_to_recommend": 9,
        #   "reviewerName": "Jane D.",
        #   "reviewerTitle": "CFO",
        #   "companySize": "51-200 employees",
        #   "publishedDate": "2024-01-15"
        # }
        "field_map": {
            "review_id":              "reviewId",
            "likes":                  "pros",
            "dislikes":               "cons",
            "recommendations":        None,
            "use_case":               None,
            "overall_rating":         "overallRating",
            "ease_of_use":            "easeOfUse",          # Capterra sub-rating ✓
            "support_rating":         "customerService",    # Capterra sub-rating ✓
            "value_rating":           "valueForMoney",      # Capterra sub-rating ✓
            "likely_to_recommend":    "likelihood_to_recommend",  # Capterra sub-rating ✓
            "reviewer_name":          "reviewerName",
            "reviewer_title":         "reviewerTitle",
            "reviewer_company_size":  "companySize",
            "reviewer_industry":      None,
            "reviewer_location":      None,
            "time_using_product":     None,
            "review_date":            "publishedDate",
            "is_verified":            None,
            "is_incentivized":        None,
            "product_overall_rating": None,
            "product_review_count":   None,
        },

        "bonus_fields": [
            "sourceUrl",
            "featuresRating",          # Capterra "Features" sub-rating
            "reviewTitle",
        ],
    },

    # -------------------------------------------------------------------------
    # ACTOR 3: focused_vanguard/multi-platform-reviews-scraper
    # Role: CROSS-PLATFORM — G2, Capterra, Gartner in one run via domain lookup.
    # Key advantage: lookbackDays for temporal filtering (freshness queries).
    # Pricing: per-result. More expensive than jupri.
    # Use case: periodic freshness updates, not full historical scrapes.
    # -------------------------------------------------------------------------
    "focused_vanguard": {
        "actor_id": "focused_vanguard/multi-platform-reviews-scraper",
        "label":    "focused_vanguard/multi-platform",
        "source_platform": "multi",                # platform detected per-record via bonus_fields

        "pricing": {
            "model":            "pay_per_result",
            "cost_per_1k":      7.25,              # mid-point of $6.49–$7.99 range
            "estimated_per_1k": 7.25,
            "free_tier_viable": False,
        },

        # domain: company domain for auto-discovery (e.g. "netsuite.com")
        # platforms: omit to scrape all; pass list to restrict
        # lookbackDays: only reviews in last N days — None = all time
        "input_schema": {
            "domain":       "{domain}",
            "platforms":    ["g2", "capterra"],    # Gartner has anti-scraping blocks
            "limit":        200,
            "lookbackDays": None,
        },

        "field_map": {
            "review_id":              "id",
            "likes":                  "pros",
            "dislikes":               "cons",
            "recommendations":        None,
            "use_case":               "body",      # full review text fallback
            "overall_rating":         "rating",
            "ease_of_use":            None,
            "support_rating":         None,
            "value_rating":           None,
            "likely_to_recommend":    None,
            "reviewer_name":          "author",
            "reviewer_title":         "authorTitle",
            "reviewer_company_size":  "companySize",
            "reviewer_industry":      None,
            "reviewer_location":      None,
            "time_using_product":     None,
            "review_date":            "publishedAt",
            "is_verified":            None,
            "is_incentivized":        None,
            "product_overall_rating": None,
            "product_review_count":   None,
        },

        # sourcePlatform must be read from bonus_data to set source_platform
        # correctly on each record — the normalizer handles this special case.
        "bonus_fields": [
            "sourcePlatform",          # "g2" | "capterra" — must be promoted to source_platform
            "sourceUrl",
        ],
    },
}


# ---------------------------------------------------------------------------
# 3. SCRAPE STRATEGY
# ---------------------------------------------------------------------------
# Maps each scraping purpose to the best actor + config.
# This is the only place you change to swap actors — scraper code is unchanged.
#
# primary_reviews:   Full historical review corpus for G2 (jupri)
# capterra_reviews:  Capterra corpus with sub-ratings (samstorm)
# freshness_update:  Recent reviews only — temporal delta (focused_vanguard)

SCRAPE_STRATEGY: dict[str, dict[str, Any]] = {

    "primary_reviews": {
        "actor":       "jupri",
        "products":    list(PRODUCTS.keys()),
        "limit":       2000,                       # per product
        "description": "Full G2 review corpus. Run once then update via freshness_update.",
    },

    "capterra_reviews": {
        "actor":       "samstorm",
        "products":    list(PRODUCTS.keys()),
        "limit":       2000,
        "description": "Capterra reviews with sub-ratings. Provides SMB buyer signals.",
    },

    "freshness_update": {
        "actor":       "focused_vanguard",
        "products":    list(PRODUCTS.keys()),
        "limit":       2000,
        "lookback_days": 90,
        "description": "Recent reviews only. Run monthly to keep corpus fresh.",
    },
}


# ---------------------------------------------------------------------------
# 4. UNIFIED SCHEMA
# ---------------------------------------------------------------------------
# Canonical field definitions for every record that flows through the pipeline.
# The normalizer outputs exactly these fields (plus bonus_data for actor extras).
# All downstream components (DuckDB, ChromaDB, NLP) read from this schema.
#
# source_platform is set by the normalizer — not the actor config — because
# focused_vanguard outputs multiple platforms in one run and the value must
# be read from each record's bonus_data["sourcePlatform"].

UNIFIED_SCHEMA: dict[str, dict[str, Any]] = {

    # --- Pipeline provenance (injected by normalizer, not from actor output) ---
    "review_id": {
        "type": str, "required": True,
        "description": "Stable unique ID. Preferably the actor's native ID; fallback to SHA-256 hash.",
    },
    "product_id": {
        "type": str, "required": True,
        "description": "Internal product key e.g. 'ORC-NS-001'. Injected from PRODUCTS config.",
    },
    "product_name": {
        "type": str, "required": True,
        "description": "Human-readable product name. Injected from PRODUCTS config.",
    },
    "source_platform": {
        "type": str, "required": True,
        "description": "Data origin: 'g2' | 'capterra'. First-class field — every downstream filter uses this.",
    },
    "source_actor": {
        "type": str, "required": True,
        "description": "Apify actor key used e.g. 'jupri', 'samstorm'. For reproducibility.",
    },
    "scraped_at": {
        "type": str, "required": True,
        "description": "ISO 8601 UTC timestamp of the scrape run. Injected by normalizer.",
    },

    # --- Review text fields (the primary NLP inputs) ---
    "likes": {
        "type": str, "required": False,
        "description": "Free-text: what the reviewer liked. G2: answers[0]. Capterra: pros.",
    },
    "dislikes": {
        "type": str, "required": False,
        "description": "Free-text: what the reviewer disliked. G2: answers[1]. Capterra: cons.",
    },
    "recommendations": {
        "type": str, "required": False,
        "description": "Free-text: recommendations to others. G2: answers[2]. Capterra: not available.",
    },
    "use_case": {
        "type": str, "required": False,
        "description": "Free-text: problem the product solves. G2: answers[3]. Capterra: not available.",
    },

    # --- Star ratings ---
    "overall_rating": {
        "type": float, "required": False,
        "description": "Overall star rating (1–5). G2: score. Capterra: overallRating.",
    },
    "ease_of_use": {
        "type": float, "required": False,
        "description": "Ease of use sub-rating (1–5). Capterra only — None for G2 records.",
    },
    "support_rating": {
        "type": float, "required": False,
        "description": "Customer support sub-rating (1–5). Capterra only — None for G2 records.",
    },
    "value_rating": {
        "type": float, "required": False,
        "description": "Value for money sub-rating (1–5). Capterra only — None for G2 records.",
    },
    "likely_to_recommend": {
        "type": float, "required": False,
        "description": "Likelihood to recommend (0–10). Capterra only — None for G2 records.",
    },

    # --- Reviewer metadata ---
    "reviewer_name": {
        "type": str, "required": False,
        "description": "Reviewer display name (may be anonymised).",
    },
    "reviewer_title": {
        "type": str, "required": False,
        "description": "Job title / role as shown on the review.",
    },
    "reviewer_company_size": {
        "type": str, "required": False,
        "description": "Company size segment e.g. 'small-business', 'mid-market', 'enterprise'.",
    },
    "reviewer_industry": {
        "type": str, "required": False,
        "description": "Industry vertical e.g. 'Computer Software', 'Financial Services'.",
    },
    "reviewer_location": {
        "type": str, "required": False,
        "description": "Country of the reviewer.",
    },
    "time_using_product": {
        "type": str, "required": False,
        "description": "Self-reported usage duration e.g. '2+ years'.",
    },

    # --- Review metadata ---
    "review_date": {
        "type": str, "required": False,
        "description": "ISO 8601 date of review publication.",
    },
    "is_verified": {
        "type": bool, "required": False,
        "description": "G2 verified reviewer badge. None if actor doesn't provide this.",
    },
    "is_incentivized": {
        "type": bool, "required": False,
        "description": "Review was incentivized. Reserved — currently no actor extracts this.",
    },

    # --- Product-level aggregates ---
    "product_overall_rating": {
        "type": float, "required": False,
        "description": "Aggregate product rating from platform header. Optional enrichment.",
    },
    "product_review_count": {
        "type": int, "required": False,
        "description": "Total reviews shown on the product page. Optional enrichment.",
    },

    # --- Pipeline quality fields ---
    "extraction_confidence": {
        "type": float, "required": False,
        "description": "0.0–1.0: fraction of core NLP-critical fields successfully extracted.",
    },
    "bonus_data": {
        "type": dict, "required": False,
        "description": "Actor-specific fields that don't fit the unified schema. Preserved for debugging.",
    },
}


# ---------------------------------------------------------------------------
# 5. OUTPUT PATHS
# ---------------------------------------------------------------------------
# All paths relative to the project root.
# Each actor run writes to its own subdirectory — runs don't overwrite each other.

OUTPUT_PATHS: dict[str, str] = {
    "raw_base":        "data/g2/raw",              # unmodified actor JSON output
    "normalized_base": "data/g2/normalized",       # unified schema records per actor/product
    "processed":       "data/g2/processed",        # merged outputs ready for NLP
    "reviews_csv":     "data/g2/processed/reviews.csv",
    "ground_truth":    "data/g2/processed/ground_truth.json",
    "topic_matrix":    "data/g2/processed/topic_matrix.csv",
    "field_stats":     "data/g2/processed/field_stats.csv",
    "run_log":         "data/g2/run_log.jsonl",    # append-only log of all scrape runs
}


# ---------------------------------------------------------------------------
# 6. APIFY SETTINGS
# ---------------------------------------------------------------------------

APIFY_SETTINGS: dict[str, Any] = {
    "token_env_var":        "APIFY_API_TOKEN",     # loaded from .env — never hardcode

    "default_memory_mb":    1024,
    "default_timeout_secs": 300,                   # 5 min; increase for large runs

    # Retry config — actors can fail on first attempt due to anti-bot measures
    "max_retries":          3,
    "retry_delay_secs":     30,

    # jupri manages its own proxy internally — do NOT inject proxy config for jupri.
    # samstorm and focused_vanguard benefit from Apify's residential pool.
    "proxy_config": {
        "useApifyProxy":    True,
        "apifyProxyGroups": ["RESIDENTIAL"],
    },

    # Actors that should NOT receive proxy config (they manage it internally)
    "no_proxy_actors": ["jupri"],
}


# ---------------------------------------------------------------------------
# VALIDATION — run this file directly to verify config integrity
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("G2 CONFIG VALIDATION")
    print("=" * 60)

    print(f"\n📦 Products: {len(PRODUCTS)}")
    for key, p in PRODUCTS.items():
        print(f"   {key}: {p['name']} ({p['tier']}) — G2 slug: {p['g2_slug']}")

    print(f"\n🤖 Actors: {len(ACTOR_CONFIGS)}")
    for key, a in ACTOR_CONFIGS.items():
        viable = "✓ free tier" if a["pricing"]["free_tier_viable"] else f"${a['pricing']['estimated_per_1k']}/1k"
        print(f"   {key}: {a['actor_id']} — {viable}")

    print(f"\n📐 Unified schema fields: {len(UNIFIED_SCHEMA)}")
    required = [f for f, v in UNIFIED_SCHEMA.items() if v["required"]]
    print(f"   Required: {required}")

    print(f"\n📁 Output paths: {len(OUTPUT_PATHS)}")
    for k, v in OUTPUT_PATHS.items():
        print(f"   {k}: {v}")

    # Verify every actor's field_map uses valid unified schema keys
    print("\n🔍 Field map validation:")
    for actor_key, actor in ACTOR_CONFIGS.items():
        invalid = [f for f in actor["field_map"] if f not in UNIFIED_SCHEMA]
        if invalid:
            print(f"   ❌ {actor_key}: unknown fields {invalid}")
        else:
            print(f"   ✅ {actor_key}: all field_map keys valid")

    print("\n✅ Config loaded successfully!")
