#!/usr/bin/env python3
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "amd_results"
VALID_DECOMP_SIGN_PATHS = {"decomp-pipeline", "decomp-adaptive"}


def fail(msg):
    print(f"[FAIL] {msg}", file=sys.stderr)
    return 1


def read_text(path):
    return path.read_text(errors="replace") if path.exists() else ""


def has_decomp_pass(text):
    return any(f"PASS [{path}]" in text for path in VALID_DECOMP_SIGN_PATHS)


def check_large_best():
    path = RESULTS / "sig_large_best.csv"
    if not path.exists():
        return fail(f"missing {path}")

    rows = list(csv.DictReader(path.open(newline="", errors="replace")))
    if not rows:
        return fail(f"empty {path}")

    bad_sign = [
        r for r in rows
        if r.get("operation") == "Sign" and r.get("path") not in VALID_DECOMP_SIGN_PATHS
    ]
    if bad_sign:
        return fail(f"non-decomp sign paths in sig_large_best.csv: {bad_sign[:3]}")

    print(f"[OK] {path} has {len(rows)} best-result rows; sign path is decomp-based")
    return 0


def check_policy_smoke():
    smoke_dir = RESULTS / "policy_smoke"
    logs = sorted(smoke_dir.glob("*.log"))
    if not logs:
        print(f"[WARN] no policy smoke logs under {smoke_dir}; run amd_tools/run_sig_policy_smoke.sh")
        return 0

    bad = []
    for path in logs:
        text = read_text(path)
        if "monolithic-precomp=off" not in text:
            bad.append((path.name, "missing monolithic-precomp=off"))
        if "cached-precomp=off" not in text:
            bad.append((path.name, "missing cached-precomp=off"))
        if "[Sign] correctness: all" not in text or not has_decomp_pass(text):
            bad.append((path.name, "missing decomp PASS"))
        if "FAIL" in text:
            bad.append((path.name, "contains FAIL"))
    if bad:
        return fail(f"policy smoke evidence failed: {bad[:5]}")

    print(f"[OK] policy smoke logs pass: {len(logs)} files")
    return 0


def check_feature_matrix():
    ranked = RESULTS / "sig_amd_feature_matrix_ranked.csv"
    if not ranked.exists():
        print(f"[WARN] no AMD feature matrix summary at {ranked}; run amd_tools/run_sig_amd_feature_matrix.sh")
        return 0

    rows = list(csv.DictReader(ranked.open(newline="", errors="replace")))
    if not rows:
        return fail(f"empty {ranked}")

    bad = [
        r for r in rows
        if r.get("status") == "PASS" and r.get("sign_path") not in VALID_DECOMP_SIGN_PATHS
    ]
    if bad:
        return fail(f"feature matrix has non-decomp sign paths: {bad[:3]}")

    print(f"[OK] AMD feature matrix summary present: {len(rows)} rows")
    return 0


def check_large_sweep_logs():
    sweep_dir = RESULTS / "large_sweep"
    logs = sorted(sweep_dir.glob("*.log"))
    if not logs:
        print(f"[WARN] no large sweep logs under {sweep_dir}; run amd_tools/run_sig_large_sweep.sh")
        return 0

    bad = []
    for path in logs:
        text = read_text(path)
        if "monolithic-precomp=on" in text:
            bad.append((path.name, "monolithic-precomp=on"))
        if "FAIL" in text:
            bad.append((path.name, "contains FAIL"))
        if "HSA_STATUS_ERROR_OUT_OF_RESOURCES" in text:
            bad.append((path.name, "contains HSA out of resources"))
    if bad:
        return fail(f"large sweep log check failed: {bad[:5]}")

    print(f"[OK] large sweep logs clean: {len(logs)} files")
    return 0


def check_profile_status():
    profile_dir = RESULTS / "profile"
    app_logs = sorted(profile_dir.glob("*_profile.log"))
    if not app_logs:
        print(f"[WARN] no app-level profile logs under {profile_dir}; run amd_tools/profile_sig_one.sh")
        return 0

    print(f"[OK] app-level profile logs present: {len(app_logs)} files")
    stale = []
    for path in profile_dir.glob("*_rocprof.log"):
        if "unrecognized arguments" in read_text(path):
            stale.append(path.name)
    if stale:
        print(f"[WARN] stale rocprof logs need rerun: {stale}")
    return 0


def main():
    rc = 0
    for check in (
        check_large_best,
        check_policy_smoke,
        check_feature_matrix,
        check_large_sweep_logs,
        check_profile_status,
    ):
        rc |= check()
    if rc == 0:
        print("[OK] competition evidence audit complete")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
