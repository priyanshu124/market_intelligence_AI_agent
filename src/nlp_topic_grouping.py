# nlp_topic_grouping.py
# =============================================================================
# Post-processing: assign macro-groups to BERTopic output topics.
#
# WHY A SEPARATE FILE:
#   BERTopic produces fine-grained clusters (50-80 per field). Many of these
#   are semantically related and should be grouped into actionable categories
#   for the dashboard and ABSA pipeline. Rather than re-running BERTopic with
#   fewer topics (which loses granularity), we keep the fine-grained clusters
#   and add a macro_group label on top.
#
# APPROACH:
#   1. Keyword scoring — each topic's keywords are scored against each group's
#      term list. Multi-word phrases score higher than single words to avoid
#      false matches (e.g. "cost" alone shouldn't trigger pricing_licensing
#      but "pricing, expensive, license" should).
#   2. Manual overrides — topics with ambiguous or wrong auto-assignment are
#      corrected via MANUAL_OVERRIDES dict. These were identified by inspecting
#      the group assignment output.
#   3. Noise/filler flagging — topics with no clear feature content (e.g.
#      "say, issues, dont, disliked" or "regret, try, worth, buy") are flagged
#      as is_noise=True and assigned to group "noise".
#
# MACRO-GROUPS (20 total — maps to ABSA aspects and dashboard panels):
#   reporting_analytics     — reports, dashboards, export, BI, insights
#   ease_of_use             — navigation, interface, intuitive, layout
#   pricing_licensing       — cost, expensive, subscription, licensing
#   implementation_setup    — setup, onboarding, go-live, configuration
#   customer_support        — support, helpdesk, response time, chat
#   integrations_api        — integrations, API, third-party, Salesforce
#   invoicing_payments      — invoices, billing, payments, AR
#   bank_reconciliation     — bank feeds, reconciliation, transactions
#   inventory_orders        — inventory, purchase orders, stock, supply chain
#   payroll_hr              — payroll, employees, taxes, timesheets
#   customization           — custom fields, workflows, flexibility
#   performance_reliability — slow, crashes, bugs, errors, loading
#   multi_entity            — consolidation, subsidiaries, intercompany
#   mobile_access           — mobile app, remote access, cloud access
#   accounting_bookkeeping  — journal entries, ledger, bookkeeping, accounts
#   automation_workflow     — automation, manual processes, workflows
#   project_budgeting       — project accounting, budgeting, cost tracking
#   learning_training       — learning curve, training, onboarding
#   updates_upgrades        — version updates, upgrade cycles, bugs
#   security_access         — roles, permissions, login, audit trail
#
# INPUT:  data/bertopic/v3/topic_info_{field}.csv  (existing BERTopic output)
# OUTPUT: data/bertopic/v3/topic_info_{field}.csv  (same files, macro_group added)
#         data/bertopic/v3/{field}_topics.csv       (macro_group joined to doc level)
#         data/bertopic/v3/macro_summary_{field}.csv (group-level rollup)
#
# USAGE:
#   python nlp_topic_grouping.py
#   python nlp_topic_grouping.py --input-dir data/bertopic/v3
#   python nlp_topic_grouping.py --fields likes dislikes
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
log = logging.getLogger("topic_grouping")


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

INPUT_DIR  = Path("data/bertopic/v3")
TEXT_FIELDS = ["likes", "dislikes", "use_case", "recommendations"]


# ---------------------------------------------------------------------------
# MACRO-GROUP DEFINITIONS
# ---------------------------------------------------------------------------
# Each group has a term list scored against topic keywords.
# Multi-word phrases (bigrams) count double — they're more precise signals.
# Order within a group doesn't matter; scoring picks the best-matching group.

MACRO_GROUPS: dict[str, dict] = {

    "reporting_analytics": {
        "label": "Reporting & Analytics",
        "terms": [
            # High-signal bigrams (weight 2)
            "custom reports", "financial reporting", "report writer", "real time",
            "real time data", "decision making", "power bi", "data analytics",
            "export excel", "access data", "data management", "reporting dashboards",
            "run reports", "generate reports", "saved searches",
            # Single terms (weight 1)
            "reports", "reporting", "report", "dashboard", "dashboards",
            "analytics", "insights", "excel", "export", "import",
            "search", "searches", "data", "visibility", "kpi",
        ],
        "bigrams": [  # terms that MUST match as a phrase (exact or near)
            "custom reports", "real time", "power bi", "saved searches",
            "financial reporting", "data analytics",
        ],
    },

    "ease_of_use": {
        "label": "Ease of Use & UX",
        "terms": [
            "navigate", "navigation", "intuitive", "user friendly", "friendly",
            "interface", "ui", "layout", "simple", "easy navigate",
            "easy use", "easy learn", "learn", "organized", "clear",
            "straightforward", "screens", "menus", "clicks", "steps",
        ],
        "bigrams": [
            "user friendly", "easy navigate", "easy use", "easy learn",
            "learning curve",
        ],
    },

    "pricing_licensing": {
        "label": "Pricing & Licensing",
        "terms": [
            "cost", "price", "pricing", "expensive", "affordable", "cheap",
            "license", "licensing", "subscription", "monthly", "annual",
            "unlimited users", "per user", "fee", "fees", "value",
            "worth", "budget", "add ons", "additional cost",
        ],
        "bigrams": [
            "unlimited users", "per user", "additional cost", "add ons",
            "subscription model", "licensing model",
        ],
    },

    "implementation_setup": {
        "label": "Implementation & Setup",
        "terms": [
            "implementation", "setup", "initial setup", "onboarding",
            "deploy", "configuration", "going live", "configure",
            "install", "rollout", "migration", "migrating", "partner",
            "consultant", "project manager", "go live",
        ],
        "bigrams": [
            "initial setup", "going live", "go live", "implementation partner",
            "implementation process",
        ],
    },

    "customer_support": {
        "label": "Customer Support",
        "terms": [
            "support", "customer service", "customer support", "helpdesk",
            "help desk", "response", "chat", "phone", "ticket",
            "account manager", "agent", "documentation", "help center",
            "knowledgebase", "community",
        ],
        "bigrams": [
            "customer service", "customer support", "help center",
            "response time", "account manager",
        ],
    },

    "integrations_api": {
        "label": "Integrations & API",
        "terms": [
            "integration", "integrations", "api", "connect", "connector",
            "salesforce", "third party", "party", "sync", "platforms",
            "ecosystem", "webhook", "rest", "endpoint", "crm integration",
            "seamless integration",
        ],
        "bigrams": [
            "third party", "seamless integration", "crm integration",
            "rest api", "api documentation",
        ],
    },

    "invoicing_payments": {
        "label": "Invoicing & Payments",
        "terms": [
            "invoice", "invoices", "invoicing", "payment", "payments",
            "billing", "bills", "credit", "send invoice", "recurring",
            "accounts receivable", "ar", "collections", "reminders",
            "estimates", "quotes",
        ],
        "bigrams": [
            "accounts receivable", "send invoice", "recurring invoice",
            "ap automation", "payment processing",
        ],
    },

    "bank_reconciliation": {
        "label": "Bank & Reconciliation",
        "terms": [
            "bank", "reconciliation", "bank reconciliation", "bank feed",
            "bank feeds", "transactions", "reconcile", "banking",
            "bank accounts", "bank statement", "cash flow",
        ],
        "bigrams": [
            "bank reconciliation", "bank feed", "bank feeds", "cash flow",
            "bank statement",
        ],
    },

    "inventory_orders": {
        "label": "Inventory & Orders",
        "terms": [
            "inventory", "orders", "order", "inventory management",
            "purchase order", "sales order", "stock", "supply chain",
            "warehouse", "fulfillment", "items", "purchasing",
            "bom", "manufacturing", "production",
        ],
        "bigrams": [
            "inventory management", "purchase order", "sales order",
            "supply chain", "order fulfillment",
        ],
    },

    "payroll_hr": {
        "label": "Payroll & HR",
        "terms": [
            "payroll", "employee", "employees", "hr", "taxes", "tax",
            "direct deposit", "pay", "salary", "benefits", "w2",
            "1099", "compliance", "deductions",
        ],
        "bigrams": [
            "direct deposit", "payroll processing", "payroll taxes",
            "employee expense",
        ],
    },

    "customization": {
        "label": "Customization & Flexibility",
        "terms": [
            "customize", "customization", "customizable", "flexibility",
            "custom fields", "workflows", "workflow", "configure",
            "tailor", "flexible", "custom", "fields", "forms",
            "templates", "custom reports",
        ],
        "bigrams": [
            "custom fields", "custom reports", "highly customizable",
            "ability customize",
        ],
    },

    "performance_reliability": {
        "label": "Performance & Reliability",
        "terms": [
            "slow", "load", "loading", "speed", "lag", "lags",
            "crash", "crashes", "glitch", "glitches", "error", "errors",
            "bugs", "performance", "freeze", "freezes", "timeout",
            "downtime", "outage", "unstable",
        ],
        "bigrams": [
            "bit slow", "slow load", "loading time", "error messages",
        ],
    },

    "multi_entity": {
        "label": "Multi-Entity & Consolidation",
        "terms": [
            "entity", "multi entity", "entities", "consolidation",
            "subsidiaries", "subsidiary", "intercompany", "multi",
            "multiple entities", "parent company", "branches",
            "consolidated", "group reporting",
        ],
        "bigrams": [
            "multi entity", "multiple entities", "intercompany",
            "consolidated financial", "group reporting",
        ],
    },

    "mobile_access": {
        "label": "Mobile & Remote Access",
        "terms": [
            "mobile", "app", "remote", "access", "anywhere",
            "device", "log", "internet", "web based", "browser",
            "offline", "tablet", "smartphone", "iphone", "android",
        ],
        "bigrams": [
            "mobile app", "remote access", "web based", "work remotely",
            "access anywhere",
        ],
    },

    "accounting_bookkeeping": {
        "label": "Accounting & Bookkeeping",
        "terms": [
            "accounting", "bookkeeping", "accountant", "journal",
            "ledger", "general ledger", "accounts", "financial",
            "journal entries", "chart of accounts", "double entry",
            "accrual", "cash basis", "trial balance",
        ],
        "bigrams": [
            "journal entries", "general ledger", "chart of accounts",
            "financial statements",
        ],
    },

    "automation_workflow": {
        "label": "Automation & Workflow",
        "terms": [
            "automation", "automate", "process", "workflow", "manual",
            "processes", "automated", "trigger", "rule", "scheduled",
            "batch", "recurring", "streamline", "efficiency",
        ],
        "bigrams": [
            "manual processes", "business processes", "automate processes",
            "workflow automation",
        ],
    },

    "project_budgeting": {
        "label": "Project & Budgeting",
        "terms": [
            "project", "budget", "budgeting", "cost tracking",
            "project management", "costs", "forecasting", "planning",
            "profitability", "job costing", "cost center",
        ],
        "bigrams": [
            "project management", "cost tracking", "job costing",
            "project accounting", "budget planning",
        ],
    },

    "learning_training": {
        "label": "Learning & Training",
        "terms": [
            "training", "learn", "learning curve", "curve", "steep",
            "tutorials", "videos", "onboarding", "guide", "courses",
            "certification", "documentation", "knowledge base",
        ],
        "bigrams": [
            "learning curve", "steep learning", "training videos",
            "training resources",
        ],
    },

    "updates_upgrades": {
        "label": "Updates & Upgrades",
        "terms": [
            "update", "updates", "upgrade", "upgrades", "version",
            "release", "changes", "bugs", "patch", "rollout",
            "new features", "changelog",
        ],
        "bigrams": [
            "new features", "version update", "upgrade process",
            "frequent updates",
        ],
    },

    "security_access": {
        "label": "Security & Access Control",
        "terms": [
            "security", "login", "password", "roles", "user mode",
            "permission", "permissions", "audit", "multi user",
            "access control", "authentication", "sso", "two factor",
            "log in", "logout",
        ],
        "bigrams": [
            "audit trail", "access control", "multi user", "two factor",
            "user permissions",
        ],
    },
}

# ---------------------------------------------------------------------------
# MANUAL OVERRIDES
# ---------------------------------------------------------------------------
# Topics that the keyword scorer assigns incorrectly, identified by inspecting
# the group assignment output. Format: {field: {topic_id: correct_group}}
# "noise" = no feature content, should be excluded from analysis.

MANUAL_OVERRIDES: dict[str, dict[int, str]] = {
    "likes": {
        29: "accounting_bookkeeping",  # "transactions, record, cash, batch" → bookkeeping not ease_of_use
        32: "customization",           # "module, department, interface" → customization
        37: "reporting_analytics",     # "machine learning, ai integration" → analytics/AI
        48: "reporting_analytics",     # "single source of truth" → data/reporting
        50: "ease_of_use",             # "fix mistakes, easy fix, correcting" → UX (correct, keep)
        52: "integrations_api",        # "sync, syncing, easy sync" → integrations not reporting
        25: "ease_of_use",             # "saved searches, search bar" → UX feature not reporting
    },
    "dislikes": {
        2:  "noise",                   # "say, issues, dont, disliked, isn" → meta noise
        23: "ease_of_use",             # "love, enjoy, easy" → actually UX positive (keep but note)
        24: "implementation_setup",    # "implementation, complex, complexity, process" → setup
        53: "noise",                   # "pos, aging, fifo" → too niche / fragmented
        12: "implementation_setup",    # "complex, implementation, complexity" → setup not pricing
        51: "noise",                   # "wms, robust, box" → warehouse niche, too small
        45: "accounting_bookkeeping",  # "ap ar, check, aging, bills" → AP/accounting
        54: "inventory_orders",        # "mrp, planning" → manufacturing/orders
        56: "noise",                   # "say, opinion, dont, criticism" → meta noise
    },
    "use_case": {
        1:  "inventory_orders",        # "inventory, manufacturing, project" → not reporting
        2:  "inventory_orders",        # "inventory, orders, management" → not reporting
        14: "noise",                   # "helps, allows, helping, easier" → generic filler
        17: "accounting_bookkeeping",  # "ap automation, ap ar" → AP/accounting
        18: "noise",                   # "recommend, trial, free trial" → meta noise
        19: "multi_entity",            # "suite, entities, multiple entities" → multi-entity
        35: "noise",                   # "regret, try, worth, buy" → meta/sentiment noise
        37: "noise",                   # "problems, solving, solve" → meta noise
        52: "integrations_api",        # "crm, sales team, customer data" → CRM/integrations
    },
    "recommendations": {
        7:  "noise",                   # "it, for, to, and, use" → generic filler
        11: "noise",                   # "benefits, time, more" → too generic
        14: "noise",                   # "problems, haven, issues, resolve" → meta
        35: "noise",                   # "regret, try, worth" → meta sentiment
    },
}


# ---------------------------------------------------------------------------
# SCORING FUNCTION
# ---------------------------------------------------------------------------

def assign_group(keywords: str, field: str, topic_id: int) -> str:
    """
    Assign a macro-group to a topic based on keyword scoring.

    Scoring logic:
      - Exact phrase match (bigram) in keywords → score +2
      - Single term match → score +1
      - Best-scoring group wins
      - If no group scores > 0 → "other"
      - Manual overrides always win

    Args:
        keywords: Comma-separated keyword string from topic_info CSV
        field:    Text field name ("likes", "dislikes", etc.)
        topic_id: Topic ID — checked against MANUAL_OVERRIDES first

    Returns:
        Macro-group key string
    """
    # Manual override always wins
    if field in MANUAL_OVERRIDES and topic_id in MANUAL_OVERRIDES[field]:
        return MANUAL_OVERRIDES[field][topic_id]

    if not isinstance(keywords, str) or not keywords.strip():
        return "other"

    kw_lower = keywords.lower()
    scores: dict[str, float] = {}

    for group, cfg in MACRO_GROUPS.items():
        score = 0.0

        # Bigram matches — score 2 each (more specific signal)
        for bigram in cfg.get("bigrams", []):
            if bigram in kw_lower:
                score += 2.0

        # Single term matches — score 1 each
        for term in cfg["terms"]:
            # Only count single-word terms here to avoid double-counting bigrams
            if " " not in term and term in kw_lower.split():
                score += 1.0
            elif " " in term and term in kw_lower:
                # Multi-word term not in bigrams list — score 1.5
                score += 1.5

        if score > 0:
            scores[group] = score

    if not scores:
        return "other"

    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# MAIN PROCESSING
# ---------------------------------------------------------------------------

def process_field(field: str, input_dir: Path) -> None:
    """
    Load topic_info CSV, assign macro-groups, save updated files.

    Produces:
      topic_info_{field}.csv  — original file + macro_group + group_label columns
      macro_summary_{field}.csv — one row per group: total docs, n_topics, top keywords
    """
    info_path = input_dir / f"topic_info_{field}.csv"
    if not info_path.exists():
        log.warning("Not found: %s — skipping", info_path)
        return

    info_df = pd.read_csv(info_path)
    log.info("Processing %s: %d topics", field, len(info_df))

    # Assign macro-group per topic
    info_df["macro_group"] = info_df.apply(
        lambda row: (
            "noise" if row["topic_id"] == -1
            else assign_group(
                row.get("top_keywords", ""),
                field,
                row["topic_id"],
            )
        ),
        axis=1,
    )

    # Add human-readable label
    group_labels = {k: v["label"] for k, v in MACRO_GROUPS.items()}
    group_labels["noise"] = "Noise / Filler"
    group_labels["other"] = "Other"
    info_df["group_label"] = info_df["macro_group"].map(group_labels).fillna("Other")

    # Save updated topic_info
    info_df.to_csv(info_path, index=False)
    log.info("Updated %s", info_path)

    # Build macro-level summary
    clean = info_df[~info_df["macro_group"].isin(["noise", "other"])].copy()

    summary = (
        clean.groupby(["macro_group", "group_label"])
        .agg(
            total_docs=("doc_count", "sum"),
            n_topics=("topic_id", "count"),
            top_topic_keywords=(
                "top_keywords",
                lambda x: " | ".join(
                    x.iloc[:3].str[:40].tolist()  # top 3 topic keyword strings
                ),
            ),
        )
        .reset_index()
        .sort_values("total_docs", ascending=False)
    )
    summary["pct_of_docs"] = (
        summary["total_docs"] / summary["total_docs"].sum() * 100
    ).round(1)

    summary_path = input_dir / f"macro_summary_{field}.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Saved %s", summary_path)

    # Update doc-level CSV if it exists
    doc_path = input_dir / f"{field}_topics.csv"
    if doc_path.exists():
        doc_df = pd.read_csv(doc_path)
        topic_to_group = info_df.set_index("topic_id")[["macro_group", "group_label"]].to_dict("index")
        doc_df["macro_group"] = doc_df["topic_id"].map(
            lambda tid: topic_to_group.get(tid, {}).get("macro_group", "other")
        )
        doc_df["group_label"] = doc_df["topic_id"].map(
            lambda tid: topic_to_group.get(tid, {}).get("group_label", "Other")
        )
        doc_df.to_csv(doc_path, index=False)
        log.info("Updated %s", doc_path)

    # Print console summary
    print(f"\n{'='*65}")
    print(f"  MACRO-GROUPS — {field.upper()}")
    print(f"{'='*65}")
    print(f"  {'Group':<35} {'Docs':>6}  {'Topics':>7}  {'Pct':>5}")
    print(f"  {'-'*60}")
    for _, row in summary.iterrows():
        print(
            f"  {row['group_label']:<35} {row['total_docs']:>6}  "
            f"{row['n_topics']:>7}  {row['pct_of_docs']:>4.1f}%"
        )
    noise_rows = info_df[info_df["macro_group"] == "noise"]
    other_rows = info_df[info_df["macro_group"] == "other"]
    print(f"  {'Noise / Filler':<35} {noise_rows['doc_count'].sum():>6}  {len(noise_rows):>7}")
    print(f"  {'Other (unassigned)':<35} {other_rows['doc_count'].sum():>6}  {len(other_rows):>7}")


def main(input_dir: Path, fields: list[str]) -> None:
    log.info("Topic grouping starting | input_dir=%s | fields=%s", input_dir, fields)
    for field in fields:
        log.info("\n--- %s ---", field.upper())
        process_field(field, input_dir)
    log.info("\nDone. Updated files in %s", input_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign macro-groups to BERTopic output topics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nlp_topic_grouping.py
  python nlp_topic_grouping.py --input-dir data/bertopic/v3
  python nlp_topic_grouping.py --fields likes dislikes
        """
    )
    parser.add_argument("--input-dir", default=str(INPUT_DIR))
    parser.add_argument(
        "--fields", nargs="+", default=TEXT_FIELDS, choices=TEXT_FIELDS
    )
    args = parser.parse_args()

    main(
        input_dir=Path(args.input_dir),
        fields=args.fields,
    )
