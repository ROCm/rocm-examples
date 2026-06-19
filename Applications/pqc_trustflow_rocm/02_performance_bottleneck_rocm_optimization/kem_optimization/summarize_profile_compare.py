#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict
from pathlib import Path


KEY_KERNELS = {
    "keypair": "batch_kem_keypair_serial_kernel",
    "encaps": "batch_kem_encaps_serial_kernel",
    "decaps": "batch_kem_decaps_serial_kernel",
}


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def duration_ns(row):
    if "Duration" in row:
        return as_float(row["Duration"])
    return as_float(row.get("End_Timestamp")) - as_float(row.get("Start_Timestamp"))


def find_one(root, pattern):
    files = sorted(root.rglob(pattern))
    return files[0] if files else None


def summarize_kernel_file(path):
    agg = defaultdict(lambda: {
        "calls": 0,
        "total_ns": 0.0,
        "max_ns": 0.0,
        "vgpr": "",
        "sgpr": "",
        "scratch": "",
        "lds": "",
        "workgroup": "",
        "grid": "",
    })
    if not path:
        return agg

    with path.open(newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            name = row.get("Kernel_Name") or row.get("Name") or row.get("Kernel Name") or ""
            if not name:
                continue
            ns = max(0.0, duration_ns(row))
            entry = agg[name]
            entry["calls"] += 1
            entry["total_ns"] += ns
            entry["max_ns"] = max(entry["max_ns"], ns)
            entry["vgpr"] = row.get("VGPR_Count", entry["vgpr"])
            entry["sgpr"] = row.get("SGPR_Count", entry["sgpr"])
            entry["scratch"] = row.get("Scratch_Size", entry["scratch"])
            entry["lds"] = row.get("LDS_Block_Size", entry["lds"])
            entry["workgroup"] = "x".join([
                row.get("Workgroup_Size_X", ""),
                row.get("Workgroup_Size_Y", ""),
                row.get("Workgroup_Size_Z", ""),
            ]).strip("x")
            entry["grid"] = "x".join([
                row.get("Grid_Size_X", ""),
                row.get("Grid_Size_Y", ""),
                row.get("Grid_Size_Z", ""),
            ]).strip("x")
    return agg


def summarize_api_file(path):
    agg = defaultdict(lambda: {"calls": 0, "total_ns": 0.0, "max_ns": 0.0})
    if not path:
        return agg

    with path.open(newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            name = row.get("Function") or row.get("Name") or row.get("API_Name") or ""
            if not name:
                continue
            ns = max(0.0, duration_ns(row))
            entry = agg[name]
            entry["calls"] += 1
            entry["total_ns"] += ns
            entry["max_ns"] = max(entry["max_ns"], ns)
    return agg


def pct(new, old):
    if old == 0:
        return ""
    return round((new - old) * 100.0 / old, 2)


def main():
    if len(sys.argv) != 2:
        print("usage: summarize_profile_compare.py <profile_compare_dir>", file=sys.stderr)
        raise SystemExit(2)

    root = Path(sys.argv[1])
    runs_path = root / "profile_compare_runs.csv"
    if not runs_path.exists():
        print(f"missing {runs_path}", file=sys.stderr)
        raise SystemExit(1)

    runs = []
    with runs_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            runs.append(row)

    kernel_rows = []
    api_rows = []
    key_rows = []

    for run in runs:
        run_dir = root / run["run_dir"]
        kernel_path = find_one(run_dir / "rocprofv3", "*kernel_trace*.csv")
        api_path = find_one(run_dir / "rocprofv3", "*hip_api_trace*.csv")
        kernel_agg = summarize_kernel_file(kernel_path)
        api_agg = summarize_api_file(api_path)

        base = {
            "target": run["target"],
            "config": run["config"],
            "bounds": run["bounds"],
            "batch": run["batch"],
            "n_ops": run["n_ops"],
            "keygen_ops_s": run["keygen_ops_s"],
            "encaps_ops_s": run["encaps_ops_s"],
            "decaps_ops_s": run["decaps_ops_s"],
        }

        for name, v in sorted(kernel_agg.items(), key=lambda item: item[1]["total_ns"], reverse=True):
            row = dict(base)
            row.update({
                "kernel": name,
                "total_ms": round(v["total_ns"] / 1e6, 3),
                "avg_ms": round((v["total_ns"] / v["calls"]) / 1e6, 3) if v["calls"] else 0,
                "max_ms": round(v["max_ns"] / 1e6, 3),
                "calls": v["calls"],
                "vgpr": v["vgpr"],
                "sgpr": v["sgpr"],
                "scratch": v["scratch"],
                "lds": v["lds"],
                "workgroup": v["workgroup"],
                "grid": v["grid"],
            })
            kernel_rows.append(row)

        for name, v in sorted(api_agg.items(), key=lambda item: item[1]["total_ns"], reverse=True):
            row = dict(base)
            row.update({
                "function": name,
                "total_ms": round(v["total_ns"] / 1e6, 3),
                "avg_ms": round((v["total_ns"] / v["calls"]) / 1e6, 3) if v["calls"] else 0,
                "max_ms": round(v["max_ns"] / 1e6, 3),
                "calls": v["calls"],
            })
            api_rows.append(row)

        for op, needle in KEY_KERNELS.items():
            for name, v in kernel_agg.items():
                if needle in name:
                    row = dict(base)
                    row.update({
                        "operation": op,
                        "kernel": name,
                        "total_ms": round(v["total_ns"] / 1e6, 3),
                        "avg_ms": round((v["total_ns"] / v["calls"]) / 1e6, 3) if v["calls"] else 0,
                        "calls": v["calls"],
                        "vgpr": v["vgpr"],
                        "sgpr": v["sgpr"],
                        "scratch": v["scratch"],
                        "lds": v["lds"],
                        "workgroup": v["workgroup"],
                        "grid": v["grid"],
                    })
                    key_rows.append(row)

    def write_csv(path, rows, fields):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    kernel_fields = [
        "target", "config", "bounds", "batch", "n_ops", "keygen_ops_s",
        "encaps_ops_s", "decaps_ops_s", "total_ms", "avg_ms", "max_ms",
        "calls", "vgpr", "sgpr", "scratch", "lds", "workgroup", "grid", "kernel",
    ]
    api_fields = [
        "target", "config", "bounds", "batch", "n_ops", "keygen_ops_s",
        "encaps_ops_s", "decaps_ops_s", "total_ms", "avg_ms", "max_ms",
        "calls", "function",
    ]
    key_fields = [
        "target", "config", "bounds", "batch", "n_ops", "operation",
        "keygen_ops_s", "encaps_ops_s", "decaps_ops_s", "total_ms", "avg_ms",
        "calls", "vgpr", "sgpr", "scratch", "lds", "workgroup", "grid", "kernel",
    ]
    write_csv(root / "kernel_summary.csv", kernel_rows, kernel_fields)
    write_csv(root / "hip_api_summary.csv", api_rows, api_fields)
    write_csv(root / "key_kernel_summary.csv", key_rows, key_fields)

    by_target_op = defaultdict(dict)
    for row in key_rows:
        by_target_op[(row["target"], row["operation"])][row["config"]] = row

    compare_rows = []
    for (target, op), configs in sorted(by_target_op.items()):
        if "baseline" not in configs or "tuned" not in configs:
            continue
        b = configs["baseline"]
        t = configs["tuned"]
        b_ms = as_float(b["total_ms"])
        t_ms = as_float(t["total_ms"])
        compare_rows.append({
            "target": target,
            "operation": op,
            "baseline_bounds": b["bounds"],
            "tuned_bounds": t["bounds"],
            "baseline_total_ms": b["total_ms"],
            "tuned_total_ms": t["total_ms"],
            "kernel_time_change_pct": pct(t_ms, b_ms),
            "baseline_vgpr": b["vgpr"],
            "tuned_vgpr": t["vgpr"],
            "baseline_sgpr": b["sgpr"],
            "tuned_sgpr": t["sgpr"],
            "baseline_scratch": b["scratch"],
            "tuned_scratch": t["scratch"],
            "baseline_workgroup": b["workgroup"],
            "tuned_workgroup": t["workgroup"],
        })

    write_csv(root / "key_kernel_compare.csv", compare_rows, [
        "target", "operation", "baseline_bounds", "tuned_bounds",
        "baseline_total_ms", "tuned_total_ms", "kernel_time_change_pct",
        "baseline_vgpr", "tuned_vgpr", "baseline_sgpr", "tuned_sgpr",
        "baseline_scratch", "tuned_scratch", "baseline_workgroup", "tuned_workgroup",
    ])

    print(f"[done] {root / 'kernel_summary.csv'}")
    print(f"[done] {root / 'hip_api_summary.csv'}")
    print(f"[done] {root / 'key_kernel_summary.csv'}")
    print(f"[done] {root / 'key_kernel_compare.csv'}")


if __name__ == "__main__":
    main()
