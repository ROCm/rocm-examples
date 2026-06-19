#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path


if len(sys.argv) != 2:
    print("usage: parse_kem_results.py <log_dir>", file=sys.stderr)
    raise SystemExit(2)


log_dir = Path(sys.argv[1])
rows = []

gpu_re = re.compile(r"^GPU:\s+(.+?)\s+\(")
runtime_re = re.compile(r"^Runtime:\s+(\S+)")
algorithm_re = re.compile(r"^Algorithm:\s+(.+?)\s+K=(\d+)\s+Q=(\d+)")
batch_re = re.compile(r"^---\s+batch=(\d+)\s+n_ops=(\d+)\s+mode=([^\s]+)(?:\s+streams=(\d+))?")
op_re = re.compile(r"^\s+(Keygen|Encaps|Decaps):\s+([0-9.]+)\s+ms/(?:batch|round)\s+(?:->|.)\s+([0-9.]+)\s+ops/sec")
profile_re = re.compile(
    r"Pipeline profile:\s+sample=([0-9.]+)\s+ntt=([0-9.]+)\s+matvec=([0-9.]+)\s+"
    r"invntt=([0-9.]+)\s+add=([0-9.]+)\s+pack=([0-9.]+)\s+total=([0-9.]+)\s+ms"
)


def new_row(log_name: str, common: dict, batch: str, n_ops: str, mode: str, streams: str) -> dict:
    row = {
        "algorithm": common.get("algorithm", ""),
        "k": common.get("k", ""),
        "q": common.get("q", ""),
        "runtime": common.get("runtime", ""),
        "gpu": common.get("gpu", ""),
        "batch": batch,
        "n_ops": n_ops,
        "mode": mode,
        "streams": streams or "1",
        "keygen_ms": "",
        "keygen_ops_s": "",
        "encaps_ms": "",
        "encaps_ops_s": "",
        "decaps_ms": "",
        "decaps_ops_s": "",
        "correctness": common.get("correctness", "UNKNOWN"),
        "profile_sample_ms": "",
        "profile_ntt_ms": "",
        "profile_matvec_ms": "",
        "profile_invntt_ms": "",
        "profile_add_ms": "",
        "profile_pack_ms": "",
        "profile_total_ms": "",
        "status": "PASS",
        "log": log_name,
    }
    return row


for log_path in sorted(log_dir.glob("*.log")):
    text = log_path.read_text(errors="replace")
    common = {"correctness": "UNKNOWN"}
    current = None

    if "FAIL" in text:
        common["correctness"] = "FAIL"
    elif "KEM" in text and "PASS" in text:
        common["correctness"] = "PASS"

    for line in text.splitlines():
        m = gpu_re.search(line)
        if m:
            common["gpu"] = m.group(1)
            continue
        m = runtime_re.search(line)
        if m:
            common["runtime"] = m.group(1)
            continue
        m = algorithm_re.search(line)
        if m:
            common["algorithm"] = m.group(1)
            common["k"] = m.group(2)
            common["q"] = m.group(3)
            continue
        m = batch_re.search(line)
        if m:
            if current:
                rows.append(current)
            current = new_row(log_path.name, common, m.group(1), m.group(2), m.group(3), m.group(4))
            continue
        m = op_re.search(line)
        if m and current:
            stage = m.group(1).lower()
            current[f"{stage}_ms"] = m.group(2)
            current[f"{stage}_ops_s"] = m.group(3)
            continue
        m = profile_re.search(line)
        if m and current:
            keys = [
                "profile_sample_ms",
                "profile_ntt_ms",
                "profile_matvec_ms",
                "profile_invntt_ms",
                "profile_add_ms",
                "profile_pack_ms",
                "profile_total_ms",
            ]
            for key, value in zip(keys, m.groups()):
                current[key] = value
            continue

    if current:
        if "exit_code=" in text and "exit_code=0" not in text:
            current["status"] = "FAIL"
        if common.get("correctness") == "FAIL":
            current["status"] = "FAIL"
        rows.append(current)


fieldnames = [
    "algorithm",
    "k",
    "q",
    "runtime",
    "gpu",
    "batch",
    "n_ops",
    "mode",
    "streams",
    "keygen_ms",
    "keygen_ops_s",
    "encaps_ms",
    "encaps_ops_s",
    "decaps_ms",
    "decaps_ops_s",
    "correctness",
    "profile_sample_ms",
    "profile_ntt_ms",
    "profile_matvec_ms",
    "profile_invntt_ms",
    "profile_add_ms",
    "profile_pack_ms",
    "profile_total_ms",
    "status",
    "log",
]

writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
writer.writeheader()
for row in rows:
    writer.writerow({key: row.get(key, "") for key in fieldnames})
