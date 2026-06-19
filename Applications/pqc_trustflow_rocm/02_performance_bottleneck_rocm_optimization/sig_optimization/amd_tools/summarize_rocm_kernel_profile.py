#!/usr/bin/env python3
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


PHASE_RULES = [
    ("setup_mu_nonce", ("setup", "compute_mu", "rhoprime", "init_kernel")),
    ("sample_y", ("sample_y", "uniform_gamma", "gamma1")),
    ("ntt_invntt", ("ntt", "invntt")),
    ("matvec", ("matvec",)),
    ("reduce_normalize", ("reduce", "caddq", "freeze")),
    ("decompose", ("decompose",)),
    ("hash_challenge", ("hash_cp", "challenge", "cbuf")),
    ("pointwise_z_cs2_ct0", ("pointwise", "cp_shared", "add_y")),
    ("check_pack", ("check_pack", "pack")),
]


NAME_KEYS = (
    "KernelName",
    "Kernel Name",
    "kernel_name",
    "Name",
    "name",
    "Function",
    "function",
)

DURATION_KEYS = (
    "DurationNs",
    "Duration_ns",
    "duration_ns",
    "Duration (ns)",
    "Duration",
    "duration",
    "KernelDuration",
)


def classify(name: str) -> str:
    low = name.lower()
    for phase, keys in PHASE_RULES:
        if any(k in low for k in keys):
            return phase
    return "other"


def parse_float(value: str):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def duration_to_us(key: str, value: str):
    num = parse_float(value)
    if num is None:
        return None
    low = key.lower()
    if "ns" in low:
        return num / 1000.0
    if "us" in low or "µs" in low:
        return num
    if "ms" in low:
        return num * 1000.0
    # rocprof CSVs commonly store raw durations in ns even when the column is
    # simply named "Duration". Treat large values as ns, small as us.
    if num > 100000.0:
        return num / 1000.0
    return num


def find_key(row, candidates):
    for key in candidates:
        if key in row:
            return key
    lowered = {k.lower(): k for k in row}
    for key in candidates:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def parse_csv(path: Path):
    rows = []
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return rows
            for row in reader:
                name_key = find_key(row, NAME_KEYS)
                dur_key = find_key(row, DURATION_KEYS)
                if not name_key or not dur_key:
                    continue
                name = (row.get(name_key) or "").strip()
                if not name:
                    continue
                dur_us = duration_to_us(dur_key, row.get(dur_key))
                if dur_us is None:
                    continue
                # Keep likely GPU kernels and skip obvious host API rows if mixed.
                if name.startswith("hip") or name.startswith("hsa_"):
                    continue
                rows.append((name, dur_us))
    except OSError:
        pass
    return rows


def summarize(root: Path):
    phase_us = defaultdict(float)
    phase_count = defaultdict(int)
    kernel_us = defaultdict(float)
    kernel_count = defaultdict(int)
    parsed_files = []

    for path in root.rglob("*.csv"):
        rows = parse_csv(path)
        if not rows:
            continue
        parsed_files.append(path)
        for name, dur_us in rows:
            phase = classify(name)
            phase_us[phase] += dur_us
            phase_count[phase] += 1
            kernel_us[name] += dur_us
            kernel_count[name] += 1

    return parsed_files, phase_us, phase_count, kernel_us, kernel_count


def main(argv):
    root = Path(argv[1]) if len(argv) > 1 else Path("amd_results/profile")
    parsed_files, phase_us, phase_count, kernel_us, kernel_count = summarize(root)

    if not parsed_files:
        print(f"No rocprof CSV kernel data found under {root}.")
        print("Run: bash amd_tools/profile_sig_one.sh mldsa44_amd 1024")
        print("Then inspect amd_results/profile/*_rocprof*/ for CSV output.")
        return 2

    total_us = sum(phase_us.values())
    print(f"# ROCm kernel profile summary")
    print(f"root,{root}")
    print(f"parsed_csv_files,{len(parsed_files)}")
    print()
    print("phase,kernel_count,total_us,total_ms,percent")
    for phase, total in sorted(phase_us.items(), key=lambda kv: kv[1], reverse=True):
        pct = (total / total_us * 100.0) if total_us else 0.0
        print(f"{phase},{phase_count[phase]},{total:.3f},{total/1000.0:.3f},{pct:.2f}")

    print()
    print("top_kernel,kernel_count,total_us,total_ms,phase")
    for name, total in sorted(kernel_us.items(), key=lambda kv: kv[1], reverse=True)[:30]:
        print(f"{name},{kernel_count[name]},{total:.3f},{total/1000.0:.3f},{classify(name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
