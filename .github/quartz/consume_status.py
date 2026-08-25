#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve the latest ready ROCm nightly reported by Quartz.

Adapted from ROCm/Quartz docs/status-json/example_consume_status.py for the
hourly quartz_test workflow. It answers one question: is the latest nightly a
build we can act on right now? A build qualifies when the Linux ROCm pipeline
reports a successful build in Quartz's latest.json. When it does, this emits the
exact version string to pin (rocm_version, e.g. 10.1.0a20260821); otherwise it
emits resolved=false and the workflow does nothing that hour.

"Latest when ready or nothing": only latest.json is consulted. There is no walk
back to an older dated build -- testing a stale nightly is never what we want.

Deduplication (test each version exactly once) lives in the workflow, not here:
this script only reports what the latest build is, and the workflow gates on a
cache marker keyed by (rocm_version, build_date).

read_status_json.py is the sibling vendored reader. Running this file as
`python3 .github/quartz/consume_status.py` puts this directory on sys.path[0],
so the plain import below resolves it with no path shim.
"""

import argparse
import os
import sys

from read_status_json import Status, StatusDocument, load_status

# The build this workflow depends on: the Linux ROCm pipeline. Gate on this
# specific platform + pipeline, never on overall_status (which folds in every
# pipeline across platforms and is routinely red even when ROCm itself is fine).
PLATFORM = "linux"
PIPELINE = "rocm"


def set_github_outputs(**outputs: str) -> None:
    """Append step outputs to $GITHUB_OUTPUT; a no-op when run outside Actions."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if not output_file:
        return
    with open(output_file, "a") as handle:
        for name, value in outputs.items():
            handle.write(f"{name}={value}\n")


def arch_is_good(status: StatusDocument, arch: str) -> bool:
    """Return True if this build passed the gate we depend on.

    - schema guard: the vendored reader is written against schema v2, so a
      document whose schema_version does not start with "2." is treated as not
      consumable rather than silently misread;
    - the Linux ROCm build must be Status.success;
    - if arch is non-empty, it must be one of the build's architectures (an
      empty arch means a platform-level gate only).
    """
    if not status.schema_version.startswith("2."):
        return False
    platform = status.platform(PLATFORM)
    if platform is None:
        return False
    if platform.pipeline_build_status(PIPELINE) != Status.success:
        return False
    if arch and arch not in platform.architectures:
        return False
    return True


def resolve(source: str | None, arch: str):
    """Load latest.json and decide whether it is ready to act on.

    Returns (resolved, status, source_label). Every fetch is wrapped: an absent
    or unreachable Quartz yields (False, None, "unavailable") so the workflow
    simply skips the hour and doubles as a Quartz availability canary.
    """
    try:
        status = load_status(source) if source else load_status()
    except Exception as exc:  # noqa: BLE001 - fail-safe: never break the workflow
        print(f"Quartz status unavailable: {exc}", file=sys.stderr)
        return False, None, "unavailable"
    if arch_is_good(status, arch):
        return True, status, "latest"
    return False, status, "not-ready"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "arch",
        nargs="?",
        default="",
        help="optional TheRock arch the build must include (e.g. gfx110X-all); "
        "empty = platform-level gate only",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="resolve latest.json only, with no fallback to an older dated build. "
        "This is the only supported mode; the flag is accepted for explicitness.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="override the status.json URL or path (default: Quartz latest nightly)",
    )
    args = parser.parse_args()

    resolved, status, source = resolve(args.source, args.arch)
    rocm_version = status.rocm_version if status else ""
    build_date = status.build_date if status else ""

    set_github_outputs(
        resolved=str(resolved).lower(),
        rocm_version=rocm_version,
        build_date=build_date,
        source=source,
    )

    if resolved:
        print(f"resolved {rocm_version} ({build_date}) from {source}")
    else:
        print(f"no ready build ({source})")


if __name__ == "__main__":
    main()
