"""Aggregate all normalized CSVs under data/g2/normalized into one CSV.

Usage:
    python src/g2/aggregate_normalized.py
    python src/g2/aggregate_normalized.py --root data/g2/normalized --output data/g2/normalized/all_products_normalized.csv

The script does not modify any existing files; it writes a single output CSV.
"""
from pathlib import Path
import csv
import argparse
import sys


def find_csv_files(root: Path):
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def build_union_header(files, out_path: Path):
    seen = []
    seen_set = set()
    for f in files:
        # skip the output file if it lives under the same tree
        try:
            if out_path and f.resolve() == out_path.resolve():
                continue
        except Exception:
            pass
        try:
            with f.open(newline='', encoding='utf-8') as fh:
                reader = csv.DictReader(fh)
                names = reader.fieldnames or []
                for n in names:
                    if n not in seen_set:
                        seen_set.add(n)
                        seen.append(n)
        except Exception:
            # skip unreadable files
            continue
    return seen


def aggregate(files, out_path: Path, header):
    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=header)
        writer.writeheader()
        for f in files:
            try:
                if out_path and f.resolve() == out_path.resolve():
                    continue
            except Exception:
                pass
            try:
                with f.open(newline='', encoding='utf-8') as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        out_row = {k: row.get(k, "") for k in header}
                        writer.writerow(out_row)
                        written += 1
            except Exception:
                # ignore problematic files
                continue
    return written


def main():
    p = argparse.ArgumentParser(description='Aggregate normalized CSV files')
    p.add_argument('--root', default='data/g2/normalized', help='Root folder to scan for normalized CSVs')
    p.add_argument('--output', default='data/g2/normalized/all_products_normalized.csv', help='Output CSV path')
    args = p.parse_args()

    root = Path(args.root)
    out = Path(args.output)

    files = find_csv_files(root)
    if not files:
        print(f'No CSV files found under {root}', file=sys.stderr)
        sys.exit(2)

    header = build_union_header(files, out)
    if not header:
        print('No headers discovered in CSV files; aborting.', file=sys.stderr)
        sys.exit(3)

    written = aggregate(files, out, header)
    print(f'Wrote {written} rows to {out} (from {len(files)} files, {len(header)} columns)')


if __name__ == '__main__':
    main()
