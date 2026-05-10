# g2_normalizer.py
# =============================================================================
# Normalize raw Apify actor output into the unified G2/Capterra review schema.
#
# DESIGN:
#   One G2Normalizer instance per (actor, product) pair.
#   All actor differences (field names, nesting, array indexing) are described
#   in g2_config.ACTOR_CONFIGS — this file reads that config at runtime.
#   Adding a new actor requires zero changes here.
#
# KEY BEHAVIOURS:
#   - Dot notation path resolution:  "location.country" → raw["location"]["country"]
#   - Integer index path resolution: "answers.0"        → raw["answers"][0]
#   - source_platform promotion: focused_vanguard emits "sourcePlatform" per record;
#     the normalizer reads that and sets source_platform on each record individually.
#   - extraction_confidence: fraction of NLP-critical fields successfully extracted.
#   - Stable review IDs: uses actor native ID if present; falls back to SHA-256 hash
#     of (product_id + likes + dislikes + review_date) for deduplication.
#   - bonus_data: actor-specific fields that don't fit the unified schema are
#     preserved here for debugging and future schema extension.
#
# USAGE:
#   normalizer = G2Normalizer("jupri", "oracle_netsuite")
#   records    = normalizer.normalize_batch(raw_items)
#   normalizer.save(records)
#
#   # Or normalize one record:
#   record = normalizer.normalize(raw_item)
#
# PAPERS:
#   Unified schema design follows data provenance principles in:
#   Halevy et al. 2006 — "The Unreasonable Effectiveness of Data"
# =============================================================================

from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .g2_config import (
    ACTOR_CONFIGS,
    OUTPUT_PATHS,
    PRODUCTS,
    UNIFIED_SCHEMA,
)

log = logging.getLogger("g2_normalizer")

# Fields used to compute extraction_confidence.
# These are the fields most critical to the downstream NLP pipeline.
# If these are missing, the record has limited analytical value.
CONFIDENCE_FIELDS = [
    "likes",
    "dislikes",
    "overall_rating",
    "reviewer_company_size",
    "review_date",
]

# For focused_vanguard: the bonus field name that carries the source platform.
# Must be promoted to the top-level source_platform field per record.
FOCUSED_VANGUARD_PLATFORM_FIELD = "sourcePlatform"

# Platform value normalisation: actor output values → canonical values.
# All downstream filters use the canonical values ("g2", "capterra").
PLATFORM_CANONICAL: dict[str, str] = {
    "g2":       "g2",
    "G2":       "g2",
    "capterra": "capterra",
    "Capterra": "capterra",
    "gartner":  "gartner",
    "Gartner":  "gartner",
}


# =============================================================================
# CORE UTILITY: _deep_get
# =============================================================================

def _deep_get(obj: Any, path: str) -> Any:
    """
    Resolve a dot-notation path into a nested dict/list structure.

    Supports:
        - Nested dict access:  "location.country" → obj["location"]["country"]
        - Array index access:  "answers.0"        → obj["answers"][0]
        - Mixed:               "items.0.name"     → obj["items"][0]["name"]

    Returns None if:
        - Any key in the path is missing
        - An integer index is out of range
        - A non-dict/list is encountered mid-path
        - path is None (actor field_map maps to None for unavailable fields)

    Args:
        obj:  The raw dict from the Apify actor output.
        path: Dot-separated path string, or None.

    Returns:
        The resolved value, or None if anything goes wrong.

    Examples:
        >>> _deep_get({"date": {"published": "2024-01"}}, "date.published")
        '2024-01'
        >>> _deep_get({"answers": ["likes", "dislikes"]}, "answers.0")
        'likes'
        >>> _deep_get({}, "missing.key")
        None
    """
    if path is None:
        return None

    current = obj
    for part in path.split("."):
        if current is None:
            return None

        # Integer part → list index
        if part.isdigit():
            idx = int(part)
            if isinstance(current, list) and idx < len(current):
                current = current[idx]
            else:
                return None

        # String part → dict key
        elif isinstance(current, dict):
            current = current.get(part)

        else:
            # Can't descend further — path is deeper than the data structure
            return None

    return current


# =============================================================================
# NORMALIZER CLASS
# =============================================================================

class G2Normalizer:
    """
    Normalizes raw Apify actor output into the unified review schema.

    One instance per (actor_key, product_key) pair. Stores context needed for
    field injection (product_id, source_actor, scraped_at) so individual
    normalize() calls are stateless.

    Args:
        actor_key:   Key in ACTOR_CONFIGS e.g. "jupri", "samstorm"
        product_key: Key in PRODUCTS e.g. "oracle_netsuite"
        scraped_at:  ISO 8601 UTC timestamp of the scrape run. Defaults to now.
    """

    def __init__(
        self,
        actor_key: str,
        product_key: str,
        scraped_at: str | None = None,
    ) -> None:
        if actor_key not in ACTOR_CONFIGS:
            raise ValueError(
                f"Unknown actor_key '{actor_key}'. "
                f"Valid options: {list(ACTOR_CONFIGS.keys())}"
            )
        if product_key not in PRODUCTS:
            raise ValueError(
                f"Unknown product_key '{product_key}'. "
                f"Valid options: {list(PRODUCTS.keys())}"
            )

        self.actor_key    = actor_key
        self.product_key  = product_key
        self.actor_cfg    = ACTOR_CONFIGS[actor_key]
        self.product_cfg  = PRODUCTS[product_key]
        self.field_map    = self.actor_cfg["field_map"]
        self.bonus_fields = set(self.actor_cfg.get("bonus_fields", []))
        self.scraped_at   = scraped_at or datetime.now(timezone.utc).isoformat()

        # Default source_platform from actor config.
        # focused_vanguard overrides this per record using bonus_data["sourcePlatform"].
        self._default_source_platform = self.actor_cfg.get("source_platform", "unknown")

        log.info(
            "G2Normalizer ready — actor=%s product=%s platform=%s",
            actor_key, product_key, self._default_source_platform,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Normalize a single raw actor output item into the unified schema.

        Process:
            1. For each unified field, resolve the actor's output path via field_map
            2. Coerce values to correct types (float for ratings, bool for flags)
            3. Inject pipeline metadata (product_id, source_platform, etc.)
            4. Collect bonus fields into bonus_data dict
            5. Compute extraction_confidence
            6. Generate stable review_id (native ID preferred, hash fallback)

        Args:
            raw: One item from the Apify actor's output JSON array.

        Returns:
            Dict matching UNIFIED_SCHEMA keys plus bonus_data.
        """
        record: dict[str, Any] = {}

        # Step 1: extract all unified schema fields using field_map paths
        for unified_field, actor_path in self.field_map.items():
            raw_value = _deep_get(raw, actor_path)
            record[unified_field] = self._coerce(unified_field, raw_value)

        # Step 2: inject pipeline metadata
        record["product_id"]    = self.product_cfg["product_id"]
        record["product_name"]  = self.product_cfg["name"]
        record["source_actor"]  = self.actor_key
        record["scraped_at"]    = self.scraped_at

        # Step 3: collect bonus fields into bonus_data
        record["bonus_data"] = self._collect_bonus(raw)

        # Step 4: resolve source_platform
        # focused_vanguard outputs per-record platform — read from bonus_data.
        # All other actors use the actor-level default from config.
        record["source_platform"] = self._resolve_source_platform(record["bonus_data"])

        # Step 5: generate stable review_id
        # Use actor's native ID if available; hash otherwise.
        record["review_id"] = self._stable_id(record)

        # Step 6: compute extraction_confidence
        record["extraction_confidence"] = self._compute_confidence(record)

        return record

    def normalize_batch(self, raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Normalize a list of raw actor output items.

        Skips records that raise exceptions (logs a warning) so one bad record
        doesn't abort the entire batch.

        Args:
            raw_items: List of raw dicts from the Apify actor output.

        Returns:
            List of normalized records. May be shorter than raw_items if
            any records were skipped.
        """
        records = []
        for i, raw in enumerate(raw_items):
            try:
                records.append(self.normalize(raw))
            except Exception as exc:
                log.warning("Skipping record %d — normalize failed: %s", i, exc)
        log.info(
            "Normalized %d / %d records (actor=%s product=%s)",
            len(records), len(raw_items), self.actor_key, self.product_key,
        )
        return records

    def save(
        self,
        records: list[dict[str, Any]],
        output_dir: str | Path | None = None,
    ) -> dict[str, Path]:
        """
        Save normalized records to JSON and CSV.

        Writes to:
            {output_dir}/{actor_key}/{product_key}_normalized.json
            {output_dir}/{actor_key}/{product_key}_normalized.csv

        Args:
            records:    List of normalized records from normalize_batch().
            output_dir: Override output directory. Defaults to OUTPUT_PATHS["normalized_base"].

        Returns:
            Dict with keys "json" and "csv" pointing to the written file paths.
        """
        base = Path(output_dir or OUTPUT_PATHS["normalized_base"])
        out_dir = base / self.actor_key
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{self.product_key}_normalized"

        # JSON — full fidelity including bonus_data
        json_path = out_dir / f"{stem}.json"
        json_path.write_text(
            json.dumps(records, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        log.info("Saved %d records → %s", len(records), json_path)

        # CSV — unified schema fields only (bonus_data serialised as JSON string)
        csv_path = out_dir / f"{stem}.csv"
        self._save_csv(records, csv_path)
        log.info("Saved CSV → %s", csv_path)

        return {"json": json_path, "csv": csv_path}

    def print_field_coverage(self, records: list[dict[str, Any]]) -> None:
        """
        Print a coverage report showing how many records have each field populated.

        Useful for validating actor output after a scrape run — reveals which
        fields the actor actually provides vs which are always None.
        """
        if not records:
            print("No records to analyse.")
            return

        total = len(records)
        print(f"\n{'Field':<30} {'Coverage':>10} {'Example value'}")
        print("-" * 70)

        for field in UNIFIED_SCHEMA:
            if field == "bonus_data":
                continue
            values = [r[field] for r in records if r.get(field) is not None]
            pct = len(values) / total * 100
            example = str(values[0])[:40] if values else "—"
            print(f"{field:<30} {pct:>9.1f}%  {example}")

        print(f"\nTotal records: {total}")

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _coerce(self, field: str, value: Any) -> Any:
        """
        Coerce a raw value to the type declared in UNIFIED_SCHEMA.

        Handles:
            - float fields (ratings): strips non-numeric characters, converts
            - bool fields (verified, incentivized): maps truthy strings
            - str fields: strips whitespace, returns None for empty strings
            - None values: always returned as None regardless of target type
        """
        if value is None:
            return None

        schema_type = UNIFIED_SCHEMA.get(field, {}).get("type", str)

        try:
            if schema_type is float:
                # Strip anything that's not a digit or decimal point
                cleaned = re.sub(r"[^\d.]", "", str(value))
                return float(cleaned) if cleaned else None

            elif schema_type is int:
                cleaned = re.sub(r"[^\d]", "", str(value))
                return int(cleaned) if cleaned else None

            elif schema_type is bool:
                if isinstance(value, bool):
                    return value
                return str(value).lower() in {"true", "yes", "1", "verified"}

            elif schema_type is str:
                cleaned = str(value).strip()
                return cleaned if cleaned else None

            else:
                return value

        except (ValueError, TypeError) as exc:
            log.debug("Coerce failed for field=%s value=%r: %s", field, value, exc)
            return None

    def _collect_bonus(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Extract actor-specific bonus fields into a flat dict.

        Uses _deep_get for nested paths so bonus fields can use dot notation
        just like the main field_map.

        Returns:
            Dict of {bonus_field_name: value}. Empty dict if no bonus fields
            are configured for this actor.
        """
        bonus: dict[str, Any] = {}
        for path in self.bonus_fields:
            # Use the last segment of the path as the dict key for readability.
            # e.g. "location.region" → key "region"
            key = path.split(".")[-1]
            value = _deep_get(raw, path)
            if value is not None:
                bonus[key] = value
        return bonus

    def _resolve_source_platform(self, bonus_data: dict[str, Any]) -> str:
        """
        Determine the source_platform for this record.

        For most actors, the platform is fixed at the actor level (e.g. jupri
        always produces G2 records). For focused_vanguard, each record has a
        "sourcePlatform" field in the bonus data that must be read individually.

        Args:
            bonus_data: The bonus_data dict already collected for this record.

        Returns:
            Canonical platform string: "g2" | "capterra" | "gartner" | "unknown"
        """
        if self.actor_key == "focused_vanguard":
            # Read from bonus_data — the field name is the last part of the path
            raw_platform = bonus_data.get("sourcePlatform") or bonus_data.get("platform")
            if raw_platform:
                return PLATFORM_CANONICAL.get(str(raw_platform), str(raw_platform).lower())

        return self._default_source_platform

    def _stable_id(self, record: dict[str, Any]) -> str:
        """
        Generate a stable, unique review ID.

        Priority:
            1. Use the actor's native ID if available (already in record["review_id"])
            2. Fall back to SHA-256 hash of key fields for deduplication

        The hash ensures that the same review scraped by two different actors
        or at two different times gets the same ID (assuming text content is stable).

        Args:
            record: The partially-built normalized record.

        Returns:
            String ID suitable for use as a primary key.
        """
        # Native ID already extracted from field_map into record["review_id"]
        native_id = record.get("review_id")
        if native_id:
            return str(native_id)

        # Hash fallback — use fields that are stable across scrape runs
        fingerprint = "|".join([
            str(record.get("product_id", "")),
            str(record.get("reviewer_name", "")),
            str(record.get("review_date", "")),
            str(record.get("likes", ""))[:100],     # first 100 chars to avoid hash sensitivity
            str(record.get("dislikes", ""))[:100],
        ])
        return "hash-" + hashlib.sha256(fingerprint.encode()).hexdigest()[:16]

    def _compute_confidence(self, record: dict[str, Any]) -> float:
        """
        Compute extraction_confidence as the fraction of NLP-critical fields
        that were successfully extracted.

        CONFIDENCE_FIELDS are defined at module level — the fields that matter
        most for the ABSA and BERTopic pipelines. A record with all five fields
        has confidence=1.0; one with none has confidence=0.0.

        Args:
            record: The fully-built normalized record.

        Returns:
            Float in [0.0, 1.0].
        """
        filled = sum(
            1 for f in CONFIDENCE_FIELDS
            if record.get(f) is not None
        )
        return round(filled / len(CONFIDENCE_FIELDS), 2)

    def _save_csv(self, records: list[dict[str, Any]], path: Path) -> None:
        """
        Write records to CSV with one column per unified schema field.

        bonus_data is serialised as a JSON string in a single column so the
        CSV remains flat and importable by any tool.

        Args:
            records: List of normalized records.
            path:    Destination file path.
        """
        if not records:
            log.warning("save_csv called with 0 records — skipping.")
            return

        # Column order: unified schema fields first, then bonus_data
        fieldnames = [f for f in UNIFIED_SCHEMA.keys()] + ["bonus_data"]

        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for record in records:
                row = dict(record)
                # Serialise bonus_data so it fits in a single CSV cell
                row["bonus_data"] = json.dumps(
                    record.get("bonus_data", {}), ensure_ascii=False
                )
                writer.writerow(row)


# =============================================================================
# STANDALONE USAGE
# =============================================================================

def normalize_file(
    input_path: str | Path,
    actor_key: str,
    product_key: str,
    output_dir: str | Path | None = None,
    scraped_at: str | None = None,
) -> list[dict[str, Any]]:
    """
    Load a raw Apify JSON export and normalize it.

    Convenience wrapper for running the normalizer from the command line
    or from apify_loader.py after a scrape completes.

    Args:
        input_path:  Path to the raw Apify output JSON file.
        actor_key:   Key in ACTOR_CONFIGS.
        product_key: Key in PRODUCTS.
        output_dir:  Where to save outputs. Defaults to OUTPUT_PATHS["normalized_base"].
        scraped_at:  ISO 8601 timestamp to inject. Defaults to now.

    Returns:
        List of normalized records.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_items = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw_items, list):
        raise ValueError(f"Expected JSON array at {input_path}, got {type(raw_items)}")

    normalizer = G2Normalizer(actor_key, product_key, scraped_at=scraped_at)
    records    = normalizer.normalize_batch(raw_items)
    normalizer.save(records, output_dir=output_dir)
    normalizer.print_field_coverage(records)

    return records


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize raw Apify G2/Capterra output")
    parser.add_argument("--input",   required=True, help="Path to raw Apify JSON file")
    parser.add_argument("--actor",   required=True, help="Actor key e.g. jupri, samstorm")
    parser.add_argument("--product", required=True, help="Product key e.g. oracle_netsuite")
    parser.add_argument("--output",  default=None,  help="Output directory (optional)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    records = normalize_file(
        input_path=args.input,
        actor_key=args.actor,
        product_key=args.product,
        output_dir=args.output,
    )
    print(f"\nDone. {len(records)} records normalized.")
