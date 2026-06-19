#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "amd_results"


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", errors="replace") as f:
        return list(csv.DictReader(f))


def best_feature_rows(rows):
    grouped = {}
    for row in rows:
        if row.get("status") != "PASS":
            continue
        key = (row.get("target", ""), row.get("benchmark_mode", ""), row.get("batch", ""))
        try:
            speedup = float(row.get("speedup_vs_base") or 0)
            ops = float(row.get("sign_ops_s") or 0)
        except ValueError:
            speedup, ops = 0.0, 0.0
        cur = grouped.get(key)
        if cur is None or (speedup, ops) > cur[0]:
            grouped[key] = ((speedup, ops), row)
    return [v[1] for k, v in sorted(grouped.items())]


def main():
    large_best = read_csv(RESULTS / "sig_large_best.csv")
    feature_ranked = read_csv(RESULTS / "sig_amd_feature_matrix_ranked.csv")
    resource = read_csv(RESULTS / "anchor03_resource" / "resource_summary.csv")

    lines = []
    lines.append("# AMD ROCm Optimization Claims")
    lines.append("")
    lines.append("## Implemented Candidates")
    lines.append("")
    lines.append("- Stable signing remains the resource-aware `decomp-pipeline` path.")
    lines.append("- `adaptive` is a runtime policy candidate: one binary selects the measured local winner by target, benchmark mode, and batch size, while falling back to base on cells where the matrix shows regressions.")
    lines.append("- `check8` and `check16` measure whether fewer host-side done-count checks reduce ROCm synchronization overhead.")
    lines.append("- `wave64_ctrl` measures whether 64-thread hash/check control kernels behave better on AMD wave64 hardware than 32-thread control kernels.")
    lines.append("- `BATCH_SIGN_CP_FUSE_ENABLE` is implemented as a measured AMD candidate: one ROCm kernel computes `cp*s1`, `cp*s2`, and `cp*t0` products for each rejection round.")
    lines.append("- `tail16_base` and `tail16_cp_fuse` separate small-tail finish behavior from the fused pointwise candidate.")
    lines.append("- `yhat_dup` measures whether duplicating `y` at sample time beats the explicit device-to-device copy.")
    lines.append("- The default build keeps these candidates off until the matrix proves a conservative target-specific gain.")
    lines.append("")

    sign_best = [r for r in large_best if r.get("operation") == "Sign"]
    if sign_best:
        lines.append("## Current Large-Sweep Sign Best")
        lines.append("")
        lines.append("| target | mode | batch | sign ops/s | path | log |")
        lines.append("| --- | --- | ---: | ---: | --- | --- |")
        for r in sign_best:
            lines.append(
                f"| {r.get('target','')} | {r.get('benchmark_mode','')} | "
                f"{r.get('batch','')} | {r.get('ops_s','')} | "
                f"{r.get('path','')} | {r.get('log','')} |"
            )
        lines.append("")

    if feature_ranked:
        lines.append("## AMD Feature Matrix Winners")
        lines.append("")
        lines.append("| target | mode | batch | best variant | speedup vs base | sign ops/s | log |")
        lines.append("| --- | --- | ---: | --- | ---: | ---: | --- |")
        for r in best_feature_rows(feature_ranked):
            lines.append(
                f"| {r.get('target','')} | {r.get('benchmark_mode','')} | "
                f"{r.get('batch','')} | {r.get('variant','')} | "
                f"{r.get('speedup_vs_base','')} | {r.get('sign_ops_s','')} | "
                f"{r.get('log','')} |"
            )
        lines.append("")
        lines.append("Matrix interpretation: local wins are useful evidence. The `adaptive` row tests whether those wins can be captured in one target/mode/batch-aware build without promoting a globally regressing macro.")
        lines.append("")
    else:
        lines.append("## AMD Feature Matrix")
        lines.append("")
        lines.append("Run `bash amd_tools/run_sig_amd_feature_matrix.sh` to generate `sig_amd_feature_matrix_ranked.csv` with per-variant speedups.")
        lines.append("")

    failures = [
        r for r in resource
        if (r.get("exit_code") and r.get("exit_code") != "0")
        or "FAIL" in (r.get("error_hint") or "")
        or "out of resources" in (r.get("error_hint") or "").lower()
    ]
    lines.append("## AMD Limitation Evidence")
    lines.append("")
    if failures:
        lines.append("The monolithic/cached-style signing candidates are retained as negative evidence; representative failures:")
        lines.append("")
        lines.append("| target/variant | exit | hint |")
        lines.append("| --- | ---: | --- |")
        for r in failures[:12]:
            lines.append(
                f"| {r.get('target','')} | {r.get('exit_code','')} | "
                f"{(r.get('error_hint') or '').replace('|', '/') } |"
            )
        lines.append("")
    else:
        lines.append("Run `bash anchor03_resource_attribution.sh` to regenerate monolithic-vs-decomp failure evidence.")
        lines.append("")

    lines.append("## Next Tuning Step")
    lines.append("")
    lines.append("Run `python3 amd_tools/select_sig_amd_variants.py`, inspect `amd_results/sig_amd_variant_plan.md`, then build selected variants. If `adaptive` is promoted, rerun smoke/debug/large-sweep to collect final evidence.")
    lines.append("")

    out = RESULTS / "optimization_claims.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
