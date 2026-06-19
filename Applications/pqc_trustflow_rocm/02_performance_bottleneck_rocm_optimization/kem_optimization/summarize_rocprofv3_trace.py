#!/usr/bin/env python3
import csv
import sys
from collections import defaultdict
from pathlib import Path


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def kernel_duration_ns(row):
    if "Duration" in row:
        return as_float(row["Duration"])
    return as_float(row.get("End_Timestamp")) - as_float(row.get("Start_Timestamp"))


def summarize_kernel(path):
    rows = list(csv.DictReader(path.open(newline="", errors="replace")))
    agg = defaultdict(lambda: {
        "count": 0,
        "total_ns": 0.0,
        "max_ns": 0.0,
        "vgpr": "",
        "sgpr": "",
        "scratch": "",
        "lds": "",
        "wg": "",
        "grid": "",
    })

    for row in rows:
        name = row.get("Kernel_Name") or row.get("Name") or row.get("Kernel Name") or ""
        if not name:
            continue
        ns = max(0.0, kernel_duration_ns(row))
        entry = agg[name]
        entry["count"] += 1
        entry["total_ns"] += ns
        entry["max_ns"] = max(entry["max_ns"], ns)
        entry["vgpr"] = row.get("VGPR_Count", entry["vgpr"])
        entry["sgpr"] = row.get("SGPR_Count", entry["sgpr"])
        entry["scratch"] = row.get("Scratch_Size", entry["scratch"])
        entry["lds"] = row.get("LDS_Block_Size", entry["lds"])
        entry["wg"] = "x".join([
            row.get("Workgroup_Size_X", ""),
            row.get("Workgroup_Size_Y", ""),
            row.get("Workgroup_Size_Z", ""),
        ]).strip("x")
        entry["grid"] = "x".join([
            row.get("Grid_Size_X", ""),
            row.get("Grid_Size_Y", ""),
            row.get("Grid_Size_Z", ""),
        ]).strip("x")

    print(f"\n# Kernel trace: {path}")
    print("total_ms,avg_ms,max_ms,calls,vgpr,sgpr,scratch,lds,workgroup,grid,kernel")
    for name, v in sorted(agg.items(), key=lambda item: item[1]["total_ns"], reverse=True):
        avg = v["total_ns"] / v["count"] if v["count"] else 0.0
        print(
            f"{v['total_ns']/1e6:.3f},"
            f"{avg/1e6:.3f},"
            f"{v['max_ns']/1e6:.3f},"
            f"{v['count']},"
            f"{v['vgpr']},"
            f"{v['sgpr']},"
            f"{v['scratch']},"
            f"{v['lds']},"
            f"{v['wg']},"
            f"{v['grid']},"
            f"{name}"
        )


def summarize_api(path):
    rows = list(csv.DictReader(path.open(newline="", errors="replace")))
    agg = defaultdict(lambda: {"count": 0, "total_ns": 0.0, "max_ns": 0.0})

    for row in rows:
        name = row.get("Function") or row.get("Name") or row.get("API_Name") or ""
        if not name:
            continue
        if "Duration" in row:
            ns = as_float(row["Duration"])
        else:
            ns = as_float(row.get("End_Timestamp")) - as_float(row.get("Start_Timestamp"))
        ns = max(0.0, ns)
        entry = agg[name]
        entry["count"] += 1
        entry["total_ns"] += ns
        entry["max_ns"] = max(entry["max_ns"], ns)

    print(f"\n# HIP API trace: {path}")
    print("total_ms,avg_ms,max_ms,calls,function")
    for name, v in sorted(agg.items(), key=lambda item: item[1]["total_ns"], reverse=True):
        avg = v["total_ns"] / v["count"] if v["count"] else 0.0
        print(f"{v['total_ns']/1e6:.3f},{avg/1e6:.3f},{v['max_ns']/1e6:.3f},{v['count']},{name}")


def main():
    if len(sys.argv) != 2:
        print("usage: summarize_rocprofv3_trace.py <rocprof_output_dir>", file=sys.stderr)
        raise SystemExit(2)

    root = Path(sys.argv[1])
    kernel_files = sorted(root.rglob("*kernel_trace*.csv"))
    api_files = sorted(root.rglob("*hip_api_trace*.csv"))

    if not kernel_files and not api_files:
        print(f"no rocprofv3 trace csv files found under {root}", file=sys.stderr)
        raise SystemExit(1)

    for path in kernel_files:
        summarize_kernel(path)
    for path in api_files:
        summarize_api(path)


if __name__ == "__main__":
    main()
