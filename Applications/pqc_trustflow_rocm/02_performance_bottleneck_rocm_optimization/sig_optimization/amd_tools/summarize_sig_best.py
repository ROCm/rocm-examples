#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("usage: summarize_sig_best.py <sig_summary.csv|->", file=sys.stderr)
    raise SystemExit(2)

path = Path(sys.argv[1])

def target_from_row(row):
    scheme = row.get("scheme", "")
    mode = row.get("mode", "")
    if scheme == "ML-DSA":
        return {"2": "mldsa44", "3": "mldsa65", "5": "mldsa87"}.get(mode, f"mldsa_mode{mode}")
    if scheme == "Aigis-sig":
        return {"1": "aigis1", "2": "aigis2", "3": "aigis3"}.get(mode, f"aigis_mode{mode}")
    return f"{scheme}_mode{mode}"

def bench_mode_from_log(log_name):
    if "_independent_" in log_name:
        return "independent"
    if "_paper_" in log_name:
        return "paper"
    return "default"

rows = []
if sys.argv[1] == "-":
    for row in csv.DictReader(sys.stdin):
        row["target"] = target_from_row(row)
        row["benchmark_mode"] = bench_mode_from_log(row.get("log", ""))
        rows.append(row)
else:
    with path.open(newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            row["target"] = target_from_row(row)
            row["benchmark_mode"] = bench_mode_from_log(row.get("log", ""))
            rows.append(row)

best = {}
for row in rows:
    if row.get("status") != "PASS":
        continue
    key = (row["target"], row["benchmark_mode"])
    for op, field in (
        ("Keygen", "keygen_ops_s"),
        ("Sign", "sign_ops_s"),
        ("Verify", "verify_ops_s"),
    ):
        try:
            ops = float(row.get(field) or 0)
        except ValueError:
            ops = 0.0
        bkey = key + (op,)
        if ops > best.get(bkey, {}).get("ops_s", -1):
            best[bkey] = {
                "target": row["target"],
                "benchmark_mode": row["benchmark_mode"],
                "operation": op,
                "batch": row.get("batch", ""),
                "ms": row.get(f"{op.lower()}_ms", ""),
                "ops_s": ops,
                "path": row.get(f"{op.lower()}_path", ""),
                "log": row.get("log", ""),
            }

fieldnames = ["target", "benchmark_mode", "operation", "batch", "ms", "ops_s", "path", "log"]
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
writer.writeheader()
for key in sorted(best):
    row = best[key]
    out = dict(row)
    out["ops_s"] = f"{row['ops_s']:.0f}"
    writer.writerow(out)
