# apify_loader.py
# =============================================================================
# Calls the Apify API to run actors, collects raw output, and saves it to disk.
# This is the entry point for all data extraction — run this file to scrape.
#
# WHAT THIS FILE DOES:
#   1. Reads actor + product config from g2_config.py
#   2. Builds the correct input payload for the chosen actor
#   3. Calls the Apify API, waits for the run to complete
#   4. Streams all dataset items back
#   5. Saves raw JSON to data/g2/raw/{actor}/{product}_{timestamp}.json
#   6. Appends a run summary to data/g2/run_log.jsonl
#   7. Optionally triggers g2_normalizer.py on the raw output
#
# USAGE:
#   # Scrape all products using the primary_reviews strategy (jupri actor)
#   python apify_loader.py --strategy primary_reviews
#
#   # Scrape one product with one actor
#   python apify_loader.py --actor jupri --product oracle_netsuite --limit 100
#
#   # Scrape Capterra for QuickBooks only
#   python apify_loader.py --actor samstorm --product quickbooks_enterprise
#
#   # Dry run — print input payload, don't call Apify
#   python apify_loader.py --strategy primary_reviews --dry-run
#
#   # Run and immediately normalize output
#   python apify_loader.py --actor jupri --product oracle_netsuite --normalize
#
# ENVIRONMENT:
#   APIFY_API_TOKEN — required. Set in .env file.
#
# OUTPUT FILES (per run):
#   data/g2/raw/{actor_key}/{product_key}_{YYYYMMDD_HHMMSS}.json
#   data/g2/run_log.jsonl   (append-only log of every run)
#
# PAPERS:
#   Data provenance tracking follows:
#   Buneman et al. 2001 — "Why and Where: A Characterization of Data Provenance"
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .g2_config import (
    ACTOR_CONFIGS,
    APIFY_SETTINGS,
    OUTPUT_PATHS,
    PRODUCTS,
    SCRAPE_STRATEGY,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("apify_loader")


# =============================================================================
# APIFY CLIENT
# =============================================================================

def _get_client():
    """
    Initialise and return an ApifyClient.

    Raises clearly if:
        - apify-client is not installed
        - APIFY_API_TOKEN is missing from environment

    Why fail early: a missing token causes a silent empty result set
    rather than an error if we don't check upfront.
    """
    try:
        from apify_client import ApifyClient
    except ImportError as exc:
        raise ImportError(
            "apify-client not installed.\n"
            "Run: pip install apify-client"
        ) from exc

    token = os.getenv(APIFY_SETTINGS["token_env_var"])
    if not token:
        raise EnvironmentError(
            f"Apify token not found in environment.\n"
            f"Expected env var: {APIFY_SETTINGS['token_env_var']}\n"
            f"Add it to your .env file: APIFY_API_TOKEN=apify_api_..."
        )
    return ApifyClient(token)


# =============================================================================
# INPUT BUILDER
# =============================================================================

def build_input(
    actor_key: str,
    product_key: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the actor input payload for a given (actor, product) pair.

    Reads the actor's input_schema from g2_config.py and substitutes
    product-specific values at runtime using template placeholders:
        {g2_slug}       → product["g2_slug"]
        {capterra_url}  → product["capterra_url"]
        {domain}        → product["domain"]

    For samstorm, startUrls is a list — wrapped correctly here.
    For focused_vanguard, platforms and lookbackDays are included.

    Args:
        actor_key:   Key in ACTOR_CONFIGS (e.g. "jupri")
        product_key: Key in PRODUCTS (e.g. "oracle_netsuite")
        overrides:   Dict of param overrides applied on top of input_schema.
                     Useful for per-run limit changes without editing config.

    Returns:
        Dict ready to pass as run_input to the Apify actor.
    """
    actor_cfg   = ACTOR_CONFIGS[actor_key]
    product_cfg = PRODUCTS[product_key]
    schema      = actor_cfg["input_schema"]

    # Template substitution map — all placeholders the schema can use
    template_vars = {
        "g2_slug":       product_cfg["g2_slug"],
        "capterra_url":  product_cfg["capterra_url"],
        "capterra_slug": product_cfg["capterra_slug"],
        "domain":        product_cfg["domain"],
        "g2_url":        product_cfg["g2_url"],
    }

    run_input: dict[str, Any] = {}
    for param, value in schema.items():
        if value is None:
            # Omit None values — don't send null params to actors
            continue

        if isinstance(value, str) and "{" in value:
            # Substitute template placeholder
            for placeholder, replacement in template_vars.items():
                value = value.replace(f"{{{placeholder}}}", replacement)

        if isinstance(value, list):
            # Lists (e.g. startUrls) — substitute placeholders in each element
            resolved = []
            for item in value:
                if isinstance(item, str) and "{" in item:
                    for placeholder, replacement in template_vars.items():
                        item = item.replace(f"{{{placeholder}}}", replacement)
                resolved.append(item)
            run_input[param] = resolved
        else:
            run_input[param] = value

    # samstorm special case: startUrls must be a list of dicts with "url" key
    # if they were passed as plain strings above, wrap them now
    if actor_key == "samstorm" and "startUrls" in run_input:
        urls = run_input["startUrls"]
        if urls and isinstance(urls[0], str):
            run_input["startUrls"] = [{"url": u} for u in urls]

    # Apply proxy config — skip for actors that manage proxy internally (e.g. jupri)
    if actor_key not in APIFY_SETTINGS.get("no_proxy_actors", []):
        run_input["proxyConfiguration"] = APIFY_SETTINGS["proxy_config"]

    # Apply any caller-supplied overrides last so they win
    if overrides:
        run_input.update(overrides)

    log.debug("Built input for %s/%s: %s", actor_key, product_key, run_input)
    return run_input


# =============================================================================
# SINGLE ACTOR RUN
# =============================================================================

def run_actor(
    actor_key: str,
    product_key: str,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """
    Run one Apify actor for one product and return the raw output items.

    Handles:
        - Input construction via build_input()
        - Actor execution with retry logic (G2 blocks cause transient failures)
        - Dataset item collection via Apify's paginated dataset API
        - Raw output saved to disk
        - Run metadata appended to run_log.jsonl

    Args:
        actor_key:   Key in ACTOR_CONFIGS
        product_key: Key in PRODUCTS
        overrides:   Optional param overrides (limit, sortBy, etc.)
        dry_run:     If True, print the input payload and return [] without
                     calling the Apify API. Safe to use in CI or testing.

    Returns:
        List of raw item dicts exactly as returned by the actor.
        Also saves them to disk as a side effect.
    """
    actor_cfg   = ACTOR_CONFIGS[actor_key]
    product_cfg = PRODUCTS[product_key]
    run_input   = build_input(actor_key, product_key, overrides)

    product_name = product_cfg["name"]
    actor_id     = actor_cfg["actor_id"]

    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("Actor:   %s", actor_id)
    log.info("Product: %s (%s)", product_name, product_key)
    log.info("Input:   %s", json.dumps(run_input, indent=None))

    if dry_run:
        log.info("DRY RUN — Apify API not called. Input above is what would be sent.")
        return []

    client    = _get_client()
    run_start = datetime.now(timezone.utc)

    # Retry loop — transient G2 blocks can cause actors to fail on first attempt
    last_error: Exception | None = None
    for attempt in range(1, APIFY_SETTINGS["max_retries"] + 1):
        try:
            log.info("Starting actor run (attempt %d/%d)...", attempt, APIFY_SETTINGS["max_retries"])
            run = client.actor(actor_id).call(
                run_input=run_input,
                memory_mbytes=APIFY_SETTINGS["default_memory_mb"],
                timeout_secs=APIFY_SETTINGS["default_timeout_secs"],
            )
            break  # success
        except Exception as exc:
            last_error = exc
            log.warning("Attempt %d failed: %s", attempt, exc)
            if attempt < APIFY_SETTINGS["max_retries"]:
                log.info("Retrying in %ds...", APIFY_SETTINGS["retry_delay_secs"])
                time.sleep(APIFY_SETTINGS["retry_delay_secs"])
    else:
        raise RuntimeError(
            f"All {APIFY_SETTINGS['max_retries']} attempts failed "
            f"for {actor_key}/{product_key}. Last error: {last_error}"
        )

    run_end       = datetime.now(timezone.utc)
    duration_secs = (run_end - run_start).total_seconds()

    # Collect all items from the actor's output dataset
    dataset_id = run["defaultDatasetId"]
    log.info("Collecting dataset items (dataset_id=%s)...", dataset_id)
    items: list[dict[str, Any]] = list(
        client.dataset(dataset_id).iterate_items()
    )

    log.info(
        "✓ %d items collected in %.1fs (actor=%s product=%s)",
        len(items), duration_secs, actor_key, product_key,
    )

    if not items:
        log.warning(
            "Actor returned 0 items. Possible causes: "
            "wrong slug, G2 block, actor quota exceeded."
        )

    # Save raw output to disk
    raw_path = _save_raw(items, actor_key, product_key, run_start)

    # Append run metadata to the append-only run log
    _log_run(
        actor_key=actor_key,
        product_key=product_key,
        actor_id=actor_id,
        run_id=run.get("id", "unknown"),
        dataset_id=dataset_id,
        item_count=len(items),
        duration_secs=duration_secs,
        run_start=run_start,
        raw_path=str(raw_path),
        input_used=run_input,
    )

    return items


# =============================================================================
# STRATEGY RUNNER (multi-product)
# =============================================================================

def run_strategy(
    strategy_key: str,
    dry_run: bool = False,
    normalize: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """
    Run a full scrape strategy across all configured products.

    A strategy (defined in SCRAPE_STRATEGY in g2_config.py) maps a scraping
    purpose to an actor + settings. This is the recommended entry point for
    production scraping — it handles all products in sequence with logging.

    Args:
        strategy_key: Key in SCRAPE_STRATEGY e.g. "primary_reviews"
        dry_run:      If True, print inputs without calling Apify
        normalize:    If True, run g2_normalizer on each product's raw output

    Returns:
        Dict mapping product_key → list of raw items
    """
    if strategy_key not in SCRAPE_STRATEGY:
        raise ValueError(
            f"Unknown strategy '{strategy_key}'. "
            f"Valid options: {list(SCRAPE_STRATEGY.keys())}"
        )

    strategy    = SCRAPE_STRATEGY[strategy_key]
    actor_key   = strategy["actor"]
    products    = strategy.get("products", list(PRODUCTS.keys()))
    description = strategy.get("description", "")

    log.info("Strategy: %s", strategy_key)
    log.info("Actor:    %s", ACTOR_CONFIGS[actor_key]["actor_id"])
    log.info("Products: %d — %s", len(products), products)
    log.info("Desc:     %s", description)

    # Build overrides from strategy-level settings
    overrides: dict[str, Any] = {}
    if "limit" in strategy:
        overrides["limit"]      = strategy["limit"]
        overrides["maxReviews"] = strategy["limit"]   # samstorm uses maxReviews
    if "lookback_days" in strategy and strategy["lookback_days"]:
        overrides["lookbackDays"] = strategy["lookback_days"]

    results: dict[str, list[dict]] = {}
    for product_key in products:
        log.info("--- %s (%s) ---", product_key, PRODUCTS[product_key]["name"])
        try:
            items = run_actor(actor_key, product_key, overrides, dry_run=dry_run)
            results[product_key] = items

            # Optionally normalize immediately after each product scrape
            if normalize and items and not dry_run:
                _normalize_product(actor_key, product_key)

        except Exception as exc:
            log.error("FAILED %s: %s", product_key, exc)
            results[product_key] = []

    total = sum(len(v) for v in results.values())
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("Strategy '%s' complete: %d total items across %d products",
             strategy_key, total, len(products))

    return results


# =============================================================================
# I/O HELPERS
# =============================================================================

def _save_raw(
    items: list[dict[str, Any]],
    actor_key: str,
    product_key: str,
    run_time: datetime,
) -> Path:
    """
    Save raw actor output to disk. Never modify these files — they are the
    permanent source of truth that all downstream stages read from.

    Output path: data/g2/raw/{actor_key}/{product_key}_{YYYYMMDD_HHMMSS}.json

    Args:
        items:       Raw items from the Apify dataset.
        actor_key:   Actor key used in the run.
        product_key: Product key scraped.
        run_time:    UTC datetime of the run start (used in filename).

    Returns:
        Path to the saved file.
    """
    timestamp = run_time.strftime("%Y%m%d_%H%M%S")
    out_dir   = Path(OUTPUT_PATHS["raw_base"]) / actor_key
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{product_key}_{timestamp}.json"
    out_path.write_text(
        json.dumps(items, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log.info("Raw output saved → %s (%d items)", out_path, len(items))
    return out_path


def _log_run(
    actor_key: str,
    product_key: str,
    actor_id: str,
    run_id: str,
    dataset_id: str,
    item_count: int,
    duration_secs: float,
    run_start: datetime,
    raw_path: str,
    input_used: dict[str, Any],
) -> None:
    """
    Append one line to the append-only run log (JSONL format).

    The run log is the audit trail for all scrape activity — it records
    every actor run with its inputs, outputs, timing, and file location.
    Used for: cost tracking, reproducibility, debugging, freshness checking.

    Output path: data/g2/run_log.jsonl
    """
    log_entry = {
        "run_start":    run_start.isoformat(),
        "actor_key":    actor_key,
        "product_key":  product_key,
        "actor_id":     actor_id,
        "apify_run_id": run_id,
        "dataset_id":   dataset_id,
        "item_count":   item_count,
        "duration_secs": round(duration_secs, 1),
        "raw_path":     raw_path,
        "input_used":   input_used,
    }

    log_path = Path(OUTPUT_PATHS["run_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_entry, ensure_ascii=False, default=str) + "\n")

    log.debug("Run logged → %s", log_path)


def _normalize_product(actor_key: str, product_key: str) -> None:
    """
    Find the most recent raw file for a product and normalize it.

    Called when --normalize flag is passed. Looks for the newest file in
    data/g2/raw/{actor_key}/{product_key}_*.json and passes it to
    g2_normalizer.normalize_file().

    Args:
        actor_key:   Actor key used in the run.
        product_key: Product key to normalize.
    """
    from .g2_normalizer import normalize_file

    raw_dir = Path(OUTPUT_PATHS["raw_base"]) / actor_key
    pattern = f"{product_key}_*.json"
    candidates = sorted(raw_dir.glob(pattern), reverse=True)

    if not candidates:
        log.warning("No raw files found matching %s/%s — skipping normalize.", actor_key, pattern)
        return

    latest = candidates[0]
    log.info("Normalizing %s...", latest)
    normalize_file(
        input_path=latest,
        actor_key=actor_key,
        product_key=product_key,
    )


# =============================================================================
# RAW FILE LOADER (for normalizing previously saved files)
# =============================================================================

def load_raw_file(path: str | Path) -> list[dict[str, Any]]:
    """
    Load a previously saved raw JSON file from disk.

    Use this to re-normalize or inspect a past scrape run without
    calling the Apify API again.

    Args:
        path: Path to a raw JSON file (e.g. data/g2/raw/jupri/oracle_netsuite_20260401_120000.json)

    Returns:
        List of raw item dicts.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")

    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(items)}")

    log.info("Loaded %d items from %s", len(items), path)
    return items


def list_raw_files(actor_key: str | None = None, product_key: str | None = None) -> list[Path]:
    """
    List all raw JSON files saved to disk, optionally filtered by actor or product.

    Args:
        actor_key:   Filter to one actor's output directory. None = all actors.
        product_key: Filter to one product's files. None = all products.

    Returns:
        List of Paths sorted newest first.
    """
    base = Path(OUTPUT_PATHS["raw_base"])

    if actor_key:
        search_dirs = [base / actor_key]
    else:
        search_dirs = [d for d in base.iterdir() if d.is_dir()] if base.exists() else []

    files: list[Path] = []
    for d in search_dirs:
        pattern = f"{product_key}_*.json" if product_key else "*.json"
        files.extend(d.glob(pattern))

    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apify loader — scrape G2/Capterra reviews via Apify actors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full primary scrape (jupri, all 5 products, 500 reviews each)
  python apify_loader.py --strategy primary_reviews

  # Capterra scrape for one product
  python apify_loader.py --actor samstorm --product quickbooks_enterprise

  # G2 scrape for NetSuite, limit 50, normalize immediately
  python apify_loader.py --actor jupri --product oracle_netsuite --limit 50 --normalize

  # Preview what would be sent to Apify without calling it
  python apify_loader.py --strategy primary_reviews --dry-run

  # Normalize a previously saved raw file without scraping again
  python apify_loader.py --normalize-file data/g2/raw/jupri/oracle_netsuite_20260401.json \\
                         --actor jupri --product oracle_netsuite
        """
    )

    # Mutually exclusive: strategy OR actor+product
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--strategy",  metavar="KEY",
                      help="Run a full strategy from SCRAPE_STRATEGY (e.g. primary_reviews)")
    mode.add_argument("--actor",     metavar="KEY",
                      help="Run one actor (requires --product)")
    mode.add_argument("--normalize-file", metavar="PATH",
                      help="Normalize an existing raw file (requires --actor --product)")

    parser.add_argument("--product",   metavar="KEY",
                        help="Product key e.g. oracle_netsuite (required with --actor)")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Override max reviews per product")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Print inputs without calling Apify")
    parser.add_argument("--normalize", action="store_true",
                        help="Run normalizer immediately after scraping")
    parser.add_argument("--list-files", action="store_true",
                        help="List all raw files saved to disk and exit")

    args = parser.parse_args()

    # List files mode
    if args.list_files:
        files = list_raw_files()
        if not files:
            print("No raw files found in", OUTPUT_PATHS["raw_base"])
        else:
            print(f"Raw files ({len(files)} total):")
            for f in files:
                size_kb = f.stat().st_size / 1024
                print(f"  {f}  ({size_kb:.1f} KB)")
        exit(0)

    # Normalize existing file mode
    if args.normalize_file:
        if not args.actor or not args.product:
            parser.error("--normalize-file requires --actor and --product")
        from .g2_normalizer import normalize_file
        records = normalize_file(
            input_path=args.normalize_file,
            actor_key=args.actor,
            product_key=args.product,
        )
        print(f"\nDone. {len(records)} records normalized.")
        exit(0)

    # Strategy mode
    if args.strategy:
        overrides: dict[str, Any] = {}
        if args.limit:
            overrides["limit"] = args.limit
            overrides["maxReviews"] = args.limit
        run_strategy(
            strategy_key=args.strategy,
            dry_run=args.dry_run,
            normalize=args.normalize,
        )
        exit(0)

    # Single actor + product mode
    if args.actor:
        if not args.product:
            parser.error("--actor requires --product")
        overrides = {}
        if args.limit:
            overrides["limit"] = args.limit
            overrides["maxReviews"] = args.limit

        items = run_actor(
            actor_key=args.actor,
            product_key=args.product,
            overrides=overrides if overrides else None,
            dry_run=args.dry_run,
        )
        print(f"\nDone. {len(items)} items collected.")

        if args.normalize and items and not args.dry_run:
            _normalize_product(args.actor, args.product)
