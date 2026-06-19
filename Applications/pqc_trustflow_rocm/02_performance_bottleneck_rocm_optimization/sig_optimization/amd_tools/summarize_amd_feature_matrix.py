#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path


if len(sys.argv) != 2:
    print("usage: summarize_amd_feature_matrix.py <sig_amd_feature_matrix.csv>", file=sys.stderr)
    raise SystemExit(2)


name_re = re.compile(
    r"^(mldsa44|mldsa65|mldsa87|aigis1|aigis2|aigis3)_"
    r"(.+)_(paper|independent)_b(\d+)\.log$"
)
repeat_name_re = re.compile(
    r"^(mldsa44|mldsa65|mldsa87|aigis1|aigis2|aigis3)_"
    r"(.+)_(paper|independent)_b(\d+)_r(\d+)\.log$"
)


def parse_log_name(log_name):
    m = repeat_name_re.match(log_name)
    if m:
        return m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    m = name_re.match(log_name)
    if not m:
        return "", "", "", "", ""
    return m.group(1), m.group(2), m.group(3), m.group(4), ""


def as_float(value):
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def median(values):
    vals = sorted(v for v in values if v > 0)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def median_row(group):
    row = dict(group[0])
    pass_rows = [
        r for r in group
        if r.get("status") == "PASS"
        and r.get("sign_pass") == "YES"
        and r.get("verify_pass") == "YES"
    ]
    source = pass_rows if pass_rows else group
    med_sign = median([as_float(r.get("sign_ops_s")) for r in source])
    med_sign_ms = median([as_float(r.get("sign_ms")) for r in source])
    med_keygen = median([as_float(r.get("keygen_ops_s")) for r in source])
    med_verify = median([as_float(r.get("verify_ops_s")) for r in source])
    row["status"] = "PASS" if pass_rows else "FAIL"
    row["sign_pass"] = "YES" if pass_rows else "NO"
    row["verify_pass"] = "YES" if pass_rows else "NO"
    row["sign_ops_s"] = f"{med_sign:.0f}" if med_sign else ""
    row["sign_ms"] = f"{med_sign_ms:.3f}" if med_sign_ms else ""
    row["keygen_ops_s"] = f"{med_keygen:.0f}" if med_keygen else ""
    row["verify_ops_s"] = f"{med_verify:.0f}" if med_verify else ""
    row["log"] = ";".join(r.get("log", "") for r in group)
    return row


rows = []
with Path(sys.argv[1]).open(newline="", errors="replace") as f:
    for row in csv.DictReader(f):
        target, variant, bench_mode, batch, repeat = parse_log_name(row.get("log", ""))
        if not target:
            continue
        row["target"] = target
        row["variant"] = variant
        row["benchmark_mode"] = bench_mode
        row["batch"] = batch
        row["repeat"] = repeat
        rows.append(row)

raw_rows = rows

grouped = {}
for row in rows:
    key = (row["target"], row["variant"], row["benchmark_mode"], row["batch"])
    grouped.setdefault(key, []).append(row)
rows = [median_row(group) for key, group in sorted(grouped.items())]

base_repeat_sign = {}
for row in raw_rows:
    if (
        row.get("variant") == "base"
        and row.get("status") == "PASS"
        and row.get("sign_pass") == "YES"
        and row.get("verify_pass") == "YES"
    ):
        base_repeat_sign[
            (row["target"], row["benchmark_mode"], row["batch"], row.get("repeat", ""))
        ] = as_float(row.get("sign_ops_s"))

paired_speedup = {}
for row in raw_rows:
    if (
        row.get("status") != "PASS"
        or row.get("sign_pass") != "YES"
        or row.get("verify_pass") != "YES"
    ):
        continue
    key = (row["target"], row["benchmark_mode"], row["batch"])
    repeat_key = (row["target"], row["benchmark_mode"], row["batch"], row.get("repeat", ""))
    base_ops = base_repeat_sign.get(repeat_key, 0.0)
    ops = as_float(row.get("sign_ops_s"))
    if base_ops > 0 and ops > 0:
        paired_speedup.setdefault((key, row["variant"]), []).append(ops / base_ops)

base_sign = {}
for row in rows:
    if row.get("status") != "PASS":
        continue
    try:
        ops = float(row.get("sign_ops_s") or 0)
    except ValueError:
        ops = 0.0
    if row["variant"] == "base":
        base_sign[(row["target"], row["benchmark_mode"], row["batch"])] = ops

out_rows = []
for row in rows:
    try:
        sign_ops = float(row.get("sign_ops_s") or 0)
    except ValueError:
        sign_ops = 0.0
    key = (row["target"], row["benchmark_mode"], row["batch"])
    base_ops = base_sign.get(key, 0.0)
    speedup = sign_ops / base_ops if base_ops > 0 and sign_ops > 0 else 0.0
    if (key, row["variant"]) in paired_speedup:
        speedup = median(paired_speedup[(key, row["variant"])])
    out_rows.append({
        "target": row["target"],
        "benchmark_mode": row["benchmark_mode"],
        "batch": row["batch"],
        "variant": row["variant"],
        "status": row.get("status", ""),
        "sign_pass": row.get("sign_pass", ""),
        "verify_pass": row.get("verify_pass", ""),
        "sign_ms": row.get("sign_ms", ""),
        "sign_ops_s": f"{sign_ops:.0f}" if sign_ops else row.get("sign_ops_s", ""),
        "speedup_vs_base": f"{speedup:.4f}" if speedup else "",
        "keygen_ops_s": row.get("keygen_ops_s", ""),
        "verify_ops_s": row.get("verify_ops_s", ""),
        "sign_path": row.get("sign_path", ""),
        "log": row.get("log", ""),
    })

out_rows.sort(key=lambda r: (
    r["target"],
    r["benchmark_mode"],
    int(r["batch"] or 0),
    -float(r["speedup_vs_base"] or 0),
    r["variant"],
))

fieldnames = [
    "target",
    "benchmark_mode",
    "batch",
    "variant",
    "status",
    "sign_pass",
    "verify_pass",
    "sign_ms",
    "sign_ops_s",
    "speedup_vs_base",
    "keygen_ops_s",
    "verify_ops_s",
    "sign_path",
    "log",
]

writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
writer.writeheader()
for row in out_rows:
    writer.writerow(row)
