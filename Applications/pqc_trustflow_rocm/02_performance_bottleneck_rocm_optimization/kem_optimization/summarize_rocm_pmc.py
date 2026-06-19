#!/usr/bin/env python3
import csv
import sys
from pathlib import Path


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    if len(sys.argv) != 2:
        print("usage: summarize_rocm_pmc.py <pmc_root_dir>", file=sys.stderr)
        raise SystemExit(2)

    root = Path(sys.argv[1])
    files = sorted(root.rglob("*counter_collection*.csv"))
    out = root / "pmc_summary.csv"

    rows = []
    for path in files:
        with path.open(newline="", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                kernel = row.get("Kernel_Name") or row.get("Name") or row.get("Kernel Name") or ""
                if not kernel:
                    continue
                numeric = {}
                for key, value in row.items():
                    val = as_float(value)
                    if val is not None:
                        numeric[key] = val
                rows.append((path, kernel, numeric))

    if not rows:
        out.write_text("status,message\nEMPTY,no counter_collection csv rows found\n", encoding="utf-8")
        print(f"[warn] no PMC counter rows found under {root}")
        print(f"[done] {out}")
        return

    # Aggregate numeric columns by kernel name.
    agg = {}
    for path, kernel, numeric in rows:
        entry = agg.setdefault(kernel, {"calls": 0, "source_files": set(), "sums": {}})
        entry["calls"] += 1
        entry["source_files"].add(str(path.relative_to(root)))
        for key, value in numeric.items():
            entry["sums"][key] = entry["sums"].get(key, 0.0) + value

    # Prefer commonly useful counters first, then include the rest.
    preferred = [
        "Duration", "Dispatch_Id", "SQ_WAVES", "GRBM_GUI_ACTIVE", "GPUBusy",
        "VALUUtilization", "VALUBusy", "SALUBusy", "MemUnitBusy",
        "MemUnitStalled", "FetchSize", "WriteSize", "FETCH_SIZE", "WRITE_SIZE",
        "L2CacheHit", "LDSBankConflict", "CU_OCCUPANCY",
        "MeanOccupancyPerCU", "MeanOccupancyPerActiveCU",
    ]
    all_cols = set()
    for entry in agg.values():
        all_cols.update(entry["sums"].keys())
    ordered_cols = [c for c in preferred if c in all_cols] + sorted(all_cols - set(preferred))

    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["kernel", "calls", "source_files"] + [f"sum_{c}" for c in ordered_cols] + [f"avg_{c}" for c in ordered_cols]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for kernel, entry in sorted(agg.items(), key=lambda item: item[1]["calls"], reverse=True):
            row = {
                "kernel": kernel,
                "calls": entry["calls"],
                "source_files": ";".join(sorted(entry["source_files"])),
            }
            for col in ordered_cols:
                total = entry["sums"].get(col, 0.0)
                row[f"sum_{col}"] = round(total, 3)
                row[f"avg_{col}"] = round(total / entry["calls"], 3) if entry["calls"] else ""
            writer.writerow(row)

    print(f"[done] {out}")


if __name__ == "__main__":
    main()
