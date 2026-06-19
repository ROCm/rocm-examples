#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("usage: parse_sig_results.py <log_dir>", file=sys.stderr)
    raise SystemExit(2)

log_dir = Path(sys.argv[1])
rows = []

stage_re = re.compile(
    r"^\s*(Keygen|Sign|Verify)\s+(\d+)\s+([0-9.]+)\s+ms\s+([0-9.]+)\s+ops/s(?:\s+\[([^\]]+)\])?"
)
header_re = re.compile(r"^===\s+(.+?)\s+\(Mode=(\d+)\)\s+\|\s+Batch=(\d+)")

for log_path in sorted(log_dir.glob("*.log")):
    row = {
        "log": log_path.name,
        "scheme": "",
        "mode": "",
        "batch": "",
        "keygen_ms": "",
        "keygen_ops_s": "",
        "keygen_path": "",
        "sign_ms": "",
        "sign_ops_s": "",
        "sign_path": "",
        "verify_ms": "",
        "verify_ops_s": "",
        "sign_pass": "NO",
        "verify_pass": "NO",
        "status": "FAIL",
    }
    text = log_path.read_text(errors="replace")
    for line in text.splitlines():
        m = header_re.search(line)
        if m:
            row["scheme"] = m.group(1)
            row["mode"] = m.group(2)
            row["batch"] = m.group(3)
            continue
        m = stage_re.search(line)
        if m:
            stage = m.group(1).lower()
            row[f"{stage}_ms"] = m.group(3)
            row[f"{stage}_ops_s"] = m.group(4)
            if stage in ("keygen", "sign"):
                row[f"{stage}_path"] = m.group(5) or ""
            continue
        if "[Sign] correctness: all" in line and "PASS" in line:
            row["sign_pass"] = "YES"
        if "[Verify] correctness: all" in line and "PASS" in line:
            row["verify_pass"] = "YES"
    if row["sign_pass"] == "YES" and row["verify_pass"] == "YES":
        row["status"] = "PASS"
    rows.append(row)

fieldnames = [
    "scheme",
    "mode",
    "batch",
    "keygen_ms",
    "keygen_ops_s",
    "keygen_path",
    "sign_ms",
    "sign_ops_s",
    "sign_path",
    "verify_ms",
    "verify_ops_s",
    "sign_pass",
    "verify_pass",
    "status",
    "log",
]

writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
writer.writeheader()
for row in rows:
    writer.writerow({key: row.get(key, "") for key in fieldnames})
