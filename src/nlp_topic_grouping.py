# nlp_topic_grouping.py
# =============================================================================
# LLM-driven topic grouping — ERP Market Intelligence
#
# REPLACES the hardcoded keyword-scoring approach (v1) with a three-stage
# LLM pipeline that understands domain abbreviations, context, and nuance.
#
# WHY LLM INSTEAD OF KEYWORD MATCHING:
#   Keyword matching produced false positives because:
#     - "cost" in "job cost tracking" → wrongly triggered pricing_licensing
#     - "easy" in "easy export" → wrongly triggered ease_of_use
#     - "bc" → matched bank reconciliation AND Business Central
#   An LLM understands these distinctions through context and reasoning.
#
# THREE-STAGE PIPELINE:
#
#   STAGE 1 — Per-topic enrichment (parallelized, cheap)
#     For each BERTopic fine-grained topic:
#       - Resolve abbreviations (ap → accounts payable, gl → general ledger)
#       - Write a plain English description of what reviewers in this cluster say
#       - Suggest 2-3 candidate functional categories
#     Input: topic_id, doc_count, top_keywords
#     Output: enriched_description, candidate_categories (per topic)
#
#   STAGE 2 — Global clustering reasoning (one call, all topics in context)
#     Send all enriched topics to LLM in one prompt:
#       - Reason about semantic relationships between topics
#       - Propose macro-groups that emerge from the data (no preset list)
#       - Assign topics to groups with justification
#       - Name each group descriptively
#     Output: {group_name: [topic_ids], group_description}
#
#   STAGE 3 — Self-critique and refinement (one call)
#     Send proposed groups back to LLM:
#       - Identify any misplacements or oversized catch-all groups
#       - Merge groups that are too similar
#       - Flag topics that belong in "noise" (meta-commentary, filler)
#     Output: final group assignments with confidence scores
#
# COST ESTIMATE:
#   Stage 1: ~60 topics × 3 fields × ~300 tokens = ~54K tokens → ~$0.08
#   Stage 2: 3 calls × ~5K tokens                = ~15K tokens → ~$0.02
#   Stage 3: 3 calls × ~4K tokens                = ~12K tokens → ~$0.02
#   Total: ~$0.12 for all 3 fields
#
# INPUT:  data/bertopic/v5/topic_info_{field}.csv
# OUTPUT: data/bertopic/v5/topic_info_{field}.csv  (macro_group column added)
#         data/bertopic/v5/{field}_topics.csv       (macro_group joined to docs)
#         data/bertopic/v5/macro_summary_{field}.csv
#         data/bertopic/v5/grouping_reasoning_{field}.json  (full LLM reasoning)
#
# USAGE:
#   python nlp_topic_grouping.py
#   python nlp_topic_grouping.py --fields likes dislikes
#   python nlp_topic_grouping.py --input-dir data/bertopic/v5
#   python nlp_topic_grouping.py --stage1-only   # enrich topics, don't group yet
#   python nlp_topic_grouping.py --skip-stage1   # use cached enrichments
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("topic_grouping")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

INPUT_DIR   = Path("data/bertopic/v5")
TEXT_FIELDS = ["likes", "dislikes", "use_case"]
MODEL       = "claude-sonnet-4-20250514"

# Stage 1: max concurrent API calls (stay within rate limits)
STAGE1_BATCH_SIZE = 10   # process 10 topics at a time
STAGE1_DELAY_SECS = 0.5  # delay between batches


# ---------------------------------------------------------------------------
# ANTHROPIC CLIENT
# ---------------------------------------------------------------------------

def get_client() -> anthropic.Anthropic:
    token = os.getenv("ANTHROPIC_API_KEY") or os.getenv("AI_API_KEY")
    if not token:
        raise EnvironmentError(
            "Anthropic API key not found.\n"
            "Set ANTHROPIC_API_KEY or AI_API_KEY in your .env file."
        )
    return anthropic.Anthropic(api_key=token)


def call_claude(client: anthropic.Anthropic, prompt: str, system: str, max_tokens: int = 2048) -> str:
    """Call Claude with retry on rate limit."""
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            wait = 30 * (attempt + 1)
            log.warning("Rate limit hit — waiting %ds (attempt %d/3)", wait, attempt + 1)
            time.sleep(wait)
        except Exception as exc:
            log.error("API call failed: %s", exc)
            raise
    raise RuntimeError("All 3 API attempts failed")


def parse_json_from_response(text: str) -> dict | list:
    """
    Extract and parse JSON from LLM response.
    Handles both raw JSON and JSON wrapped in markdown code blocks.
    """
    # Try raw parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Extract from ```json ... ``` blocks
    match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Extract from <output> tags (reasoning-then-JSON pattern)
    match = re.search(r'<output>([\s\S]+?)</output>', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response:\n{text[:500]}")


# ---------------------------------------------------------------------------
# STAGE 1: PER-TOPIC ENRICHMENT
# ---------------------------------------------------------------------------

STAGE1_SYSTEM = """You are an expert ERP software analyst. You understand accounting, 
finance, supply chain, and enterprise software terminology deeply.

You will receive a BERTopic cluster from reviews of ERP products (NetSuite, SAP, 
Microsoft Dynamics, QuickBooks, Sage Intacct, Workday, Acumatica, Xero, Certinia).

Your job: enrich each topic by resolving abbreviations and writing a clear description
of what reviewers in this cluster are actually talking about.

Respond ONLY with a JSON object. No preamble, no explanation outside the JSON."""

STAGE1_PROMPT = """Enrich this BERTopic cluster from ERP software reviews.

Topic ID: {topic_id}
Document count: {doc_count}
Top keywords: {top_keywords}
Text field: {field} (what reviewers wrote about what they {field_verb})

Return a JSON object with exactly these keys:
{{
  "topic_id": {topic_id},
  "resolved_keywords": "keywords with abbreviations expanded to full ERP terminology",
  "plain_description": "1-2 sentence plain English description of what reviewers in this cluster are discussing. Be specific about the ERP feature or pain point.",
  "candidate_categories": ["category1", "category2"],
  "is_noise": false,
  "noise_reason": null
}}

For is_noise: set true only if the topic is meta-commentary (e.g. "I can't think of downsides", 
"nothing to dislike", "no complaints") with no specific feature content.
For candidate_categories: use specific ERP functional area names, not generic terms.
Examples of good categories: "accounts payable automation", "financial reporting", 
"implementation complexity", "user interface navigation", "bank reconciliation",
"inventory management", "payroll processing", "multi-entity consolidation"."""

FIELD_VERBS = {
    "likes":    "liked",
    "dislikes": "disliked",
    "use_case": "use the product for",
}


def run_stage1(
    client: anthropic.Anthropic,
    info_df: pd.DataFrame,
    field: str,
    cache_path: Path,
) -> dict[int, dict]:
    """
    Enrich each topic individually with LLM understanding.

    Processes in batches to stay within rate limits.
    Caches results to disk so Stage 1 doesn't re-run if Stage 2/3 fail.

    Returns: {topic_id: enrichment_dict}
    """
    # Load cache if exists
    if cache_path.exists():
        log.info("Loading Stage 1 cache from %s", cache_path)
        return json.loads(cache_path.read_text())

    field_verb    = FIELD_VERBS.get(field, "talked about")
    enrichments: dict[int, dict] = {}
    clean_topics  = info_df[info_df["topic_id"] != -1]
    total         = len(clean_topics)

    log.info("Stage 1: enriching %d topics for field '%s' ...", total, field)

    for i, (_, row) in enumerate(clean_topics.iterrows()):
        tid = int(row["topic_id"])
        kw  = str(row.get("top_keywords", ""))

        if not kw or kw == "unlabelled":
            enrichments[tid] = {
                "topic_id": tid,
                "resolved_keywords": kw,
                "plain_description": "Unlabelled topic",
                "candidate_categories": ["other"],
                "is_noise": True,
                "noise_reason": "No keywords available",
            }
            continue

        prompt = STAGE1_PROMPT.format(
            topic_id=tid,
            doc_count=int(row["doc_count"]),
            top_keywords=kw,
            field=field,
            field_verb=field_verb,
        )

        try:
            response = call_claude(client, prompt, STAGE1_SYSTEM, max_tokens=512)
            enrichment = parse_json_from_response(response)
            enrichments[tid] = enrichment
            log.info("  [%d/%d] topic %d enriched: %s",
                     i+1, total, tid,
                     enrichment.get("plain_description", "")[:60])
        except Exception as exc:
            log.warning("  topic %d enrichment failed: %s", tid, exc)
            enrichments[tid] = {
                "topic_id": tid,
                "resolved_keywords": kw,
                "plain_description": f"Keywords: {kw}",
                "candidate_categories": ["other"],
                "is_noise": False,
                "noise_reason": None,
            }

        # Batch delay
        if (i + 1) % STAGE1_BATCH_SIZE == 0:
            time.sleep(STAGE1_DELAY_SECS)

    # Cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(enrichments, indent=2))
    log.info("Stage 1 complete. Cache saved → %s", cache_path)

    return enrichments


# ---------------------------------------------------------------------------
# STAGE 2: GLOBAL CLUSTERING REASONING
# ---------------------------------------------------------------------------

STAGE2_SYSTEM = """You are an expert ERP software product analyst tasked with organizing 
customer review topics into meaningful, actionable macro-groups for product stakeholders.

You will receive enriched BERTopic clusters from reviews of ERP products. Your job is to:
1. Reason about semantic relationships between topics
2. Propose macro-groups that emerge naturally from the data (do NOT use a preset list)
3. Assign every non-noise topic to exactly one macro-group
4. Name each group descriptively using ERP domain language

The groups should be useful for: competitive intelligence, feature gap analysis, 
sentiment monitoring, and executive briefings."""

STAGE2_PROMPT = """Here are {n_topics} enriched BERTopic topics from the "{field}" field 
of ERP software reviews. Each topic represents a cluster of reviews discussing related themes.

ENRICHED TOPICS:
{topics_json}

TASK:
First, think through the semantic relationships between these topics in <reasoning> tags.
Consider:
- Which topics are discussing the same ERP functional area?
- Which topics are too similar and should be in the same group?
- What natural groupings emerge from the actual content?
- What would be most useful for Intuit product stakeholders?

Then output your grouping in <output> tags as a JSON object:
{{
  "macro_groups": [
    {{
      "group_key": "snake_case_key",
      "group_label": "Human Readable Label",
      "group_description": "1 sentence: what functional area or theme this covers",
      "topic_ids": [list of topic_ids assigned to this group],
      "reasoning": "Why these topics belong together"
    }}
  ],
  "noise_topic_ids": [topic_ids that are meta-commentary with no feature content],
  "ungrouped_topic_ids": [topic_ids you are uncertain about]
}}

Rules:
- Every non-noise topic must appear in exactly one macro_group OR ungrouped_topic_ids
- Aim for 10-20 groups (not too fine, not too coarse)
- Group names should be specific ERP functional areas, not generic ("Reporting & Analytics" 
  not "Features", "Implementation Complexity" not "Problems")
- Do NOT create a large catch-all "Other" group — if a topic doesn't fit, put it in ungrouped"""

def run_stage2(
    client: anthropic.Anthropic,
    enrichments: dict[int, dict],
    field: str,
) -> dict:
    """
    Send all enriched topics to LLM for global clustering reasoning.

    Returns the raw Stage 2 response dict including macro_groups.
    """
    # Filter noise topics from Stage 1
    non_noise = {
        tid: e for tid, e in enrichments.items()
        if not e.get("is_noise", False)
    }

    # Build compact topic list for the prompt
    topics_for_prompt = []
    for tid, e in sorted(non_noise.items()):
        topics_for_prompt.append({
            "topic_id":          tid,
            "doc_count":         enrichments[tid].get("doc_count",
                                 e.get("doc_count", "?")),
            "keywords":          e.get("resolved_keywords", ""),
            "description":       e.get("plain_description", ""),
            "candidate_cats":    e.get("candidate_categories", []),
        })

    prompt = STAGE2_PROMPT.format(
        n_topics=len(topics_for_prompt),
        field=field,
        topics_json=json.dumps(topics_for_prompt, indent=2),
    )

    log.info("Stage 2: global clustering (%d non-noise topics) ...", len(topics_for_prompt))
    response = call_claude(client, prompt, STAGE2_SYSTEM, max_tokens=4096)

    # Extract reasoning for audit log
    reasoning_match = re.search(r'<reasoning>([\s\S]+?)</reasoning>', response)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    result = parse_json_from_response(response)
    result["_stage2_reasoning"] = reasoning

    log.info("Stage 2 complete: %d macro-groups proposed",
             len(result.get("macro_groups", [])))
    for g in result.get("macro_groups", []):
        log.info("  %-35s %d topics", g["group_label"], len(g["topic_ids"]))

    return result


# ---------------------------------------------------------------------------
# STAGE 3: SELF-CRITIQUE AND REFINEMENT
# ---------------------------------------------------------------------------

STAGE3_SYSTEM = """You are a senior ERP product analyst reviewing a topic grouping for quality.
Be critical. Identify problems and fix them."""

STAGE3_PROMPT = """Review this proposed grouping of ERP review topics and improve it.

PROPOSED GROUPING:
{grouping_json}

UNGROUPED TOPICS (need assignment):
{ungrouped_json}

Check for and fix:
1. OVERSIZED GROUPS: Any group with >8 topics likely contains distinct themes — split it
2. UNDERSIZED GROUPS: Any group with 1-2 topics — can it merge with a similar group?
3. MISPLACEMENTS: Topics where the keywords don't match the group label
4. MISSING NOISE: Topics that are meta-commentary ("no complaints", "nothing to dislike") 
   not already in noise_topic_ids
5. UNGROUPED: Assign all ungrouped topics to the best matching group

Think through fixes in <reasoning> tags, then output the corrected grouping in <output> tags
using the same JSON structure as the input but with your improvements applied.

The output must be complete — include ALL groups and ALL topic assignments."""


def run_stage3(
    client: anthropic.Anthropic,
    stage2_result: dict,
    enrichments: dict[int, dict],
) -> dict:
    """
    Self-critique pass: fix oversized groups, misplacements, ungrouped topics.

    Returns the final refined grouping dict.
    """
    ungrouped_ids = stage2_result.get("ungrouped_topic_ids", [])
    ungrouped_enriched = [
        enrichments.get(str(tid), enrichments.get(int(tid), {"topic_id": tid}))
        for tid in ungrouped_ids
    ]

    prompt = STAGE3_PROMPT.format(
        grouping_json=json.dumps({
            "macro_groups":     stage2_result.get("macro_groups", []),
            "noise_topic_ids":  stage2_result.get("noise_topic_ids", []),
        }, indent=2),
        ungrouped_json=json.dumps(ungrouped_enriched, indent=2),
    )

    log.info("Stage 3: self-critique and refinement ...")
    response = call_claude(client, prompt, STAGE3_SYSTEM, max_tokens=4096)

    reasoning_match = re.search(r'<reasoning>([\s\S]+?)</reasoning>', response)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    result = parse_json_from_response(response)
    result["_stage3_reasoning"] = reasoning

    log.info("Stage 3 complete: %d final macro-groups",
             len(result.get("macro_groups", [])))

    return result


# ---------------------------------------------------------------------------
# APPLY GROUPS TO DATAFRAMES
# ---------------------------------------------------------------------------

def apply_groups(
    info_df: pd.DataFrame,
    doc_df: pd.DataFrame,
    final_grouping: dict,
    enrichments: dict[int, dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply final group assignments back to topic_info and doc-level DataFrames.

    Also adds is_noise flag (from Stage 1 enrichments + Stage 3 noise list).
    Builds macro_summary DataFrame for dashboard.

    Returns: (info_df_enriched, doc_df_enriched, macro_summary_df)
    """
    # Build topic_id → group mapping
    noise_ids = set(final_grouping.get("noise_topic_ids", []))
    topic_to_group: dict[int, dict] = {}

    for group in final_grouping.get("macro_groups", []):
        for tid in group.get("topic_ids", []):
            topic_to_group[int(tid)] = {
                "macro_group": group["group_key"],
                "group_label": group["group_label"],
            }

    # Mark noise topics
    for tid in noise_ids:
        topic_to_group[int(tid)] = {
            "macro_group": "noise",
            "group_label": "Noise / Filler",
        }

    # Enrich topic_info
    info_df = info_df.copy()
    info_df["macro_group"] = info_df["topic_id"].apply(
        lambda tid: topic_to_group.get(int(tid), {}).get("macro_group", "unassigned")
        if tid != -1 else "outlier"
    )
    info_df["group_label"] = info_df["topic_id"].apply(
        lambda tid: topic_to_group.get(int(tid), {}).get("group_label", "Unassigned")
        if tid != -1 else "Outlier"
    )

    # Add Stage 1 enrichment columns
    info_df["plain_description"] = info_df["topic_id"].apply(
        lambda tid: enrichments.get(str(tid), enrichments.get(int(tid), {})).get("plain_description", "")
    )
    info_df["is_noise"] = info_df["macro_group"].isin(["noise", "outlier"])

    # Enrich doc-level
    doc_df = doc_df.copy()
    doc_df["macro_group"] = doc_df["topic_id"].apply(
        lambda tid: topic_to_group.get(int(tid), {}).get("macro_group", "unassigned")
        if tid != -1 else "outlier"
    )
    doc_df["group_label"] = doc_df["topic_id"].apply(
        lambda tid: topic_to_group.get(int(tid), {}).get("group_label", "Unassigned")
        if tid != -1 else "Outlier"
    )

    # Build macro summary
    clean = info_df[~info_df["is_noise"] & (info_df["topic_id"] != -1)]
    summary_rows = []
    for group in final_grouping.get("macro_groups", []):
        group_topics = clean[clean["macro_group"] == group["group_key"]]
        summary_rows.append({
            "macro_group":       group["group_key"],
            "group_label":       group["group_label"],
            "group_description": group.get("group_description", ""),
            "n_topics":          len(group_topics),
            "total_docs":        group_topics["doc_count"].sum(),
            "top_keywords":      " | ".join(
                group_topics.nlargest(3, "doc_count")["top_keywords"]
                .str[:40].tolist()
            ),
        })

    macro_df = pd.DataFrame(summary_rows).sort_values("total_docs", ascending=False)
    if not macro_df.empty:
        macro_df["pct_docs"] = (
            macro_df["total_docs"] / macro_df["total_docs"].sum() * 100
        ).round(1)

    return info_df, doc_df, macro_df


# ---------------------------------------------------------------------------
# MAIN PROCESSING PER FIELD
# ---------------------------------------------------------------------------

def process_field(
    field: str,
    input_dir: Path,
    client: anthropic.Anthropic,
    stage1_only: bool,
    skip_stage1: bool,
) -> None:
    """Run the full 3-stage LLM grouping pipeline for one field."""

    info_path = input_dir / f"topic_info_{field}.csv"
    doc_path  = input_dir / f"{field}_topics.csv"

    if not info_path.exists():
        log.warning("Not found: %s — skipping", info_path)
        return

    info_df = pd.read_csv(info_path)
    doc_df  = pd.read_csv(doc_path) if doc_path.exists() else pd.DataFrame()

    log.info("Field '%s': %d topics", field, len(info_df))

    # Stage 1 cache path
    cache_path = input_dir / f"stage1_cache_{field}.json"

    # --- Stage 1 ---
    if skip_stage1 and cache_path.exists():
        log.info("Skipping Stage 1 — loading cache from %s", cache_path)
        enrichments = json.loads(cache_path.read_text())
    else:
        enrichments = run_stage1(client, info_df, field, cache_path)

    # Add doc_count to enrichments from info_df
    for _, row in info_df.iterrows():
        tid = str(int(row["topic_id"]))
        if tid in enrichments:
            enrichments[tid]["doc_count"] = int(row["doc_count"])

    if stage1_only:
        log.info("--stage1-only: stopping after Stage 1")
        return

    # --- Stage 2 ---
    stage2_result = run_stage2(client, enrichments, field)

    # --- Stage 3 ---
    final_grouping = run_stage3(client, stage2_result, enrichments)

    # --- Apply and save ---
    info_enriched, doc_enriched, macro_df = apply_groups(
        info_df, doc_df, final_grouping, enrichments
    )

    info_enriched.to_csv(info_path, index=False)
    log.info("Updated → %s", info_path)

    if not doc_df.empty:
        doc_enriched.to_csv(doc_path, index=False)
        log.info("Updated → %s", doc_path)

    macro_path = input_dir / f"macro_summary_{field}.csv"
    macro_df.to_csv(macro_path, index=False)
    log.info("Saved → %s", macro_path)

    # Save full reasoning audit log
    reasoning_log = {
        "field":              field,
        "stage1_enrichments": enrichments,
        "stage2_result":      stage2_result,
        "stage3_result":      final_grouping,
    }
    reasoning_path = input_dir / f"grouping_reasoning_{field}.json"
    reasoning_path.write_text(json.dumps(reasoning_log, indent=2, default=str))
    log.info("Reasoning log → %s", reasoning_path)

    # Console summary
    print(f"\n{'='*65}")
    print(f"  LLM GROUPING — {field.upper()}")
    print(f"{'='*65}")
    print(f"  {'Group':<35} {'Docs':>6}  {'Topics':>7}  {'Pct':>5}")
    print(f"  {'-'*58}")
    for _, row in macro_df.iterrows():
        print(f"  {row['group_label']:<35} {row['total_docs']:>6}  "
              f"{row['n_topics']:>7}  {row['pct_docs']:>4.1f}%")


# ---------------------------------------------------------------------------
# MAIN / CLI
# ---------------------------------------------------------------------------

def main(
    input_dir: Path,
    fields: list[str],
    stage1_only: bool,
    skip_stage1: bool,
) -> None:
    client = get_client()
    log.info("LLM topic grouping | fields=%s | model=%s", fields, MODEL)

    for field in fields:
        log.info("\n%s\n--- FIELD: %s ---\n%s", "="*55, field.upper(), "="*55)
        process_field(
            field=field,
            input_dir=input_dir,
            client=client,
            stage1_only=stage1_only,
            skip_stage1=skip_stage1,
        )

    log.info("All fields complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="LLM-driven topic grouping — 3-stage Claude pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nlp_topic_grouping.py
  python nlp_topic_grouping.py --fields likes dislikes
  python nlp_topic_grouping.py --stage1-only      # enrich topics, save cache
  python nlp_topic_grouping.py --skip-stage1      # use cached enrichments
  python nlp_topic_grouping.py --input-dir data/bertopic/v5
        """
    )
    p.add_argument("--input-dir",   default=str(INPUT_DIR))
    p.add_argument("--fields",      nargs="+", default=TEXT_FIELDS, choices=TEXT_FIELDS)
    p.add_argument("--stage1-only", action="store_true",
                   help="Run Stage 1 enrichment only, save cache, stop")
    p.add_argument("--skip-stage1", action="store_true",
                   help="Skip Stage 1, load from cache, run Stages 2+3 only")

    args = p.parse_args()
    main(
        input_dir=Path(args.input_dir),
        fields=args.fields,
        stage1_only=args.stage1_only,
        skip_stage1=args.skip_stage1,
    )
