#!/usr/bin/env python3
import csv
import math
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "amd_results"
RANKED = RESULTS / "sig_amd_feature_matrix_ranked.csv"
OUT_MD = RESULTS / "sig_amd_variant_plan.md"
OUT_ENV = RESULTS / "sig_amd_variant_plan.env"

TARGETS = ("mldsa44", "mldsa65", "mldsa87", "aigis1", "aigis2", "aigis3")
DEFAULT_MIN_SPEEDUP = 1.0000
DEFAULT_GEOMEAN = 1.0300
VALID_DECOMP_SIGN_PATHS = {"decomp-pipeline", "decomp-adaptive"}


def env_float(name, default):
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


MIN_SPEEDUP = env_float("SIG_AMD_SELECT_MIN_SPEEDUP", DEFAULT_MIN_SPEEDUP)
GEOMEAN_SPEEDUP = env_float("SIG_AMD_SELECT_GEOMEAN_SPEEDUP", DEFAULT_GEOMEAN)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", errors="replace") as f:
        return list(csv.DictReader(f))


def as_float(value, default=0.0):
    try:
        return float(value or 0)
    except ValueError:
        return default


def pass_row(row):
    return (
        row.get("status") == "PASS"
        and row.get("sign_pass") == "YES"
        and row.get("verify_pass") == "YES"
        and row.get("sign_path") in VALID_DECOMP_SIGN_PATHS
    )


def geomean(values):
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def variant_env_name(target):
    return f"SIG_AMD_VARIANT_{target.upper()}"


def build_cells(rows):
    cells = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        target = row.get("target", "")
        mode = row.get("benchmark_mode", "")
        batch = row.get("batch", "")
        variant = row.get("variant", "")
        if not target or not mode or not batch or not variant:
            continue
        cells[target][(mode, batch)][variant] = row
    return cells


def evaluate_target(target, target_cells):
    tested = {
        key: variants
        for key, variants in target_cells.items()
        if "base" in variants and pass_row(variants["base"])
    }
    if not tested:
        return "base", [], "no passing base rows"

    variants = sorted({
        variant
        for rows in tested.values()
        for variant in rows
        if variant != "base"
    })

    diagnostics = []
    best = None
    for variant in variants:
        speedups = []
        missing = []
        failed = []
        for key, rows in sorted(tested.items()):
            row = rows.get(variant)
            if row is None:
                missing.append(key)
                continue
            if not pass_row(row):
                failed.append(key)
                continue
            speedups.append(as_float(row.get("speedup_vs_base")))

        combo_count = len(tested)
        min_sp = min(speedups) if speedups else 0.0
        gm = geomean(speedups)
        mean = sum(speedups) / len(speedups) if speedups else 0.0
        wins = sum(1 for v in speedups if v > 1.0)
        losses = sum(1 for v in speedups if v < 1.0)

        ok = (
            len(speedups) == combo_count
            and not missing
            and not failed
            and min_sp >= MIN_SPEEDUP
            and gm >= GEOMEAN_SPEEDUP
        )
        if missing:
            reason = f"missing {len(missing)} cells"
        elif failed:
            reason = f"failed {len(failed)} cells"
        elif min_sp < MIN_SPEEDUP:
            reason = f"min speedup {min_sp:.4f} below {MIN_SPEEDUP:.4f}"
        elif gm < GEOMEAN_SPEEDUP:
            reason = f"geomean {gm:.4f} below {GEOMEAN_SPEEDUP:.4f}"
        else:
            reason = "selected"

        diag = {
            "target": target,
            "variant": variant,
            "combos": combo_count,
            "passed": len(speedups),
            "min": min_sp,
            "geomean": gm,
            "mean": mean,
            "wins": wins,
            "losses": losses,
            "ok": ok,
            "reason": reason,
        }
        diagnostics.append(diag)
        if ok and (best is None or (gm, min_sp, mean) > (best["geomean"], best["min"], best["mean"])):
            best = diag

    if best is None:
        return "base", diagnostics, "no conservative non-base winner"
    return best["variant"], diagnostics, "promoted by conservative matrix rule"


def local_winners(rows):
    winners = []
    grouped = defaultdict(list)
    for row in rows:
        if pass_row(row):
            key = (row.get("target", ""), row.get("benchmark_mode", ""), row.get("batch", ""))
            grouped[key].append(row)
    for key, group in sorted(grouped.items()):
        non_base = [
            r for r in group
            if r.get("variant") != "base" and as_float(r.get("speedup_vs_base")) > 1.0
        ]
        if not non_base:
            continue
        best = max(non_base, key=lambda r: as_float(r.get("speedup_vs_base")))
        winners.append((key, best))
    return winners


def main():
    rows = read_csv(RANKED)
    if not rows:
        raise SystemExit(f"missing or empty {RANKED}; run amd_tools/run_sig_amd_feature_matrix.sh first")

    RESULTS.mkdir(parents=True, exist_ok=True)
    cells = build_cells(rows)

    selections = {}
    all_diagnostics = []
    reasons = {}
    for target in TARGETS:
        selected, diagnostics, reason = evaluate_target(target, cells.get(target, {}))
        selections[target] = selected
        all_diagnostics.extend(diagnostics)
        reasons[target] = reason

    env_lines = [
        "# Generated by amd_tools/select_sig_amd_variants.py",
        "# Source this file with amd_tools/build_sig_amd_selected.sh.",
        f"# Rule: min_speedup>={MIN_SPEEDUP:.4f}, geomean>={GEOMEAN_SPEEDUP:.4f}.",
    ]
    for target in TARGETS:
        env_lines.append(f"{variant_env_name(target)}={selections[target]}")
    OUT_ENV.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    md = []
    md.append("# AMD SIG Variant Plan")
    md.append("")
    md.append(
        f"Conservative rule: a non-base variant must pass every measured cell, "
        f"keep min speedup >= {MIN_SPEEDUP:.4f}, and reach geomean >= {GEOMEAN_SPEEDUP:.4f}."
    )
    md.append("")
    md.append("## Selected Build")
    md.append("")
    md.append("| target | selected variant | reason |")
    md.append("| --- | --- | --- |")
    for target in TARGETS:
        md.append(f"| {target} | {selections[target]} | {reasons[target]} |")
    md.append("")
    md.append("## Candidate Diagnostics")
    md.append("")
    md.append("| target | variant | passed / combos | min | geomean | mean | wins | losses | decision |")
    md.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for d in all_diagnostics:
        md.append(
            f"| {d['target']} | {d['variant']} | {d['passed']} / {d['combos']} | "
            f"{d['min']:.4f} | {d['geomean']:.4f} | {d['mean']:.4f} | "
            f"{d['wins']} | {d['losses']} | {d['reason']} |"
        )
    md.append("")
    md.append("## Local Winners")
    md.append("")
    md.append("These rows are useful for the writeup, but are not promoted unless the target-level rule above passes.")
    md.append("")
    md.append("| target | mode | batch | variant | speedup vs base | log |")
    md.append("| --- | --- | ---: | --- | ---: | --- |")
    for (target, mode, batch), row in local_winners(rows):
        md.append(
            f"| {target} | {mode} | {batch} | {row.get('variant','')} | "
            f"{row.get('speedup_vs_base','')} | {row.get('log','')} |"
        )
    md.append("")
    md.append("## Build Command")
    md.append("")
    md.append("```bash")
    md.append("bash amd_tools/build_sig_amd_selected.sh amd_results/sig_amd_variant_plan.env")
    md.append("```")
    md.append("")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] wrote {OUT_ENV}")
    print(f"[OK] wrote {OUT_MD}")


if __name__ == "__main__":
    main()
