#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

if len(sys.argv) != 3:
    print("usage: compare_amd_4090.py <amd_sig_summary.csv> <4090_csv>", file=sys.stderr)
    raise SystemExit(2)

amd_path = Path(sys.argv[1])
nv_path = Path(sys.argv[2])

def amd_target(row):
    scheme = row.get("scheme", "")
    mode = row.get("mode", "")
    if scheme == "ML-DSA":
        return {"2": "mldsa44", "3": "mldsa65", "5": "mldsa87"}.get(mode, f"mldsa_mode{mode}")
    if scheme == "Aigis-sig":
        return {"1": "aigis1", "2": "aigis2", "3": "aigis3"}.get(mode, f"aigis_mode{mode}")
    return f"{scheme}_mode{mode}"

def amd_bench_mode(log_name):
    if "_independent_" in log_name:
        return "independent"
    if "_paper_" in log_name:
        return "paper"
    return "paper"

def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

amd_best = {}
with amd_path.open(newline="", errors="replace") as f:
    for row in csv.DictReader(f):
        if row.get("status") != "PASS":
            continue
        target = amd_target(row)
        bench_mode = amd_bench_mode(row.get("log", ""))
        for op, field in (
            ("Keygen", "keygen_ops_s"),
            ("Sign", "sign_ops_s"),
            ("Verify", "verify_ops_s"),
        ):
            ops = as_float(row.get(field))
            key = (target, bench_mode, op)
            if ops > amd_best.get(key, {}).get("ops_s", -1):
                amd_best[key] = {
                    "target": target,
                    "benchmark_mode": bench_mode,
                    "operation": op,
                    "batch": row.get("batch", ""),
                    "ms": row.get(f"{op.lower()}_ms", ""),
                    "ops_s": ops,
                    "path": row.get(f"{op.lower()}_path", ""),
                    "log": row.get("log", ""),
                }

nv_rows = []
with nv_path.open(newline="", errors="replace-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        target = row.get("目标") or row.get("target") or ""
        bench_mode = row.get("模式") or row.get("mode") or row.get("benchmark_mode") or ""
        if not target or not bench_mode:
            continue
        nv_rows.append(row)

nv_best = {}
for row in nv_rows:
    target = row.get("目标") or row.get("target")
    bench_mode = row.get("模式") or row.get("mode") or row.get("benchmark_mode")
    candidates = (
        ("Keygen", row.get("Keygen_ops_s") or row.get("keygen_ops_s"), row.get("Keygen_ms") or row.get("keygen_ms"), row.get("Keygen路径") or row.get("keygen_path")),
        ("Sign", row.get("Sign_ops_s") or row.get("sign_ops_s"), row.get("Sign_ms") or row.get("sign_ms"), row.get("Sign路径") or row.get("sign_path")),
        ("Verify", row.get("Verify_ops_s") or row.get("verify_ops_s"), row.get("Verify_ms") or row.get("verify_ms"), ""),
    )
    for op, ops_text, ms_text, path_text in candidates:
        ops = as_float(ops_text)
        key = (target, bench_mode, op)
        if ops > nv_best.get(key, {}).get("ops_s", -1):
            nv_best[key] = {
                "target": target,
                "benchmark_mode": bench_mode,
                "operation": op,
                "batch": row.get("批量N") or row.get("batch") or "",
                "ms": ms_text or "",
                "ops_s": ops,
                "path": path_text or "",
            }

fieldnames = [
    "target",
    "benchmark_mode",
    "operation",
    "amd_best_batch",
    "amd_ms",
    "amd_ops_s",
    "amd_path",
    "rtx4090_batch",
    "rtx4090_ms",
    "rtx4090_ops_s",
    "rtx4090_path",
    "amd_vs_4090_ratio",
    "amd_log",
]
writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
writer.writeheader()

for key in sorted(set(amd_best) | set(nv_best)):
    amd = amd_best.get(key, {})
    nv = nv_best.get(key, {})
    amd_ops = amd.get("ops_s", 0.0)
    nv_ops = nv.get("ops_s", 0.0)
    ratio = amd_ops / nv_ops if nv_ops > 0 else 0.0
    writer.writerow({
        "target": key[0],
        "benchmark_mode": key[1],
        "operation": key[2],
        "amd_best_batch": amd.get("batch", ""),
        "amd_ms": amd.get("ms", ""),
        "amd_ops_s": f"{amd_ops:.0f}" if amd else "",
        "amd_path": amd.get("path", ""),
        "rtx4090_batch": nv.get("batch", ""),
        "rtx4090_ms": nv.get("ms", ""),
        "rtx4090_ops_s": f"{nv_ops:.0f}" if nv else "",
        "rtx4090_path": nv.get("path", ""),
        "amd_vs_4090_ratio": f"{ratio:.3f}" if amd and nv else "",
        "amd_log": amd.get("log", ""),
    })
