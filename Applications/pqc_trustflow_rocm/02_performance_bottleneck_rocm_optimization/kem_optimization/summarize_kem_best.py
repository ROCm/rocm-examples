#!/usr/bin/env python3
import csv
import sys
from pathlib import Path


if len(sys.argv) != 2:
    print("usage: summarize_kem_best.py <kem_summary.csv>", file=sys.stderr)
    raise SystemExit(2)


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


rows = list(csv.DictReader(Path(sys.argv[1]).open(newline="")))
best = {}

for row in rows:
    if row.get("status") != "PASS":
        continue
    algo = row.get("algorithm", "")
    if not algo:
        continue
    entry = best.setdefault(
        algo,
        {
            "algorithm": algo,
            "best_keygen_ops_s": 0.0,
            "best_keygen_config": "",
            "best_encaps_ops_s": 0.0,
            "best_encaps_config": "",
            "best_decaps_ops_s": 0.0,
            "best_decaps_config": "",
        },
    )
    config = f"batch={row.get('batch','')} mode={row.get('mode','')} streams={row.get('streams','')}"
    for op in ("keygen", "encaps", "decaps"):
        value = to_float(row.get(f"{op}_ops_s", ""))
        key = f"best_{op}_ops_s"
        if value > entry[key]:
            entry[key] = value
            entry[f"best_{op}_config"] = config


fieldnames = [
    "algorithm",
    "best_keygen_ops_s",
    "best_keygen_config",
    "best_encaps_ops_s",
    "best_encaps_config",
    "best_decaps_ops_s",
    "best_decaps_config",
]

writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
writer.writeheader()
for algorithm in sorted(best):
    row = best[algorithm]
    writer.writerow(
        {
            key: (f"{row[key]:.0f}" if key.endswith("_ops_s") else row[key])
            for key in fieldnames
        }
    )
