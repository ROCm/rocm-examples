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

read_status_json is Quartz's reusable reader, not vendored here. The workflow
sparse-checks it out from ROCm/Quartz at a pinned commit and puts its directory
on PYTHONPATH, so the plain import below resolves it. To run this file or its
tests locally, point PYTHONPATH at a checkout, e.g.

    git clone --depth 1 https://github.com/ROCm/Quartz /tmp/quartz
    PYTHONPATH=/tmp/quartz/scripts/consumer python3 .github/quartz/consume_status.py
"""

import argparse
import os
import sys
from typing import NamedTuple

from read_status_json import (
    DEFAULT_SOURCE,
    Status,
    StatusDocument,
    load_status,
)

# The build this workflow depends on: the Linux ROCm pipeline. Gate on this
# specific platform + pipeline, never on overall_status (which folds in every
# pipeline across platforms and is routinely red even when ROCm itself is fine).
PLATFORM = "linux"
PIPELINE = "rocm"

# The status.json schema major this consumer understands. A bump to a new major
# can move or rename the fields the accessors read, so an unsupported major is a
# hard failure, not a soft skip (mirrors ROCm/Quartz example_consume_status.py).
SUPPORTED_SCHEMA_MAJOR = "2."


class Resolution(NamedTuple):
    """Outcome of resolve(): whether a ready build was found, the parsed status
    document (None when Quartz was unreachable), and a short source label
    (latest / not-ready / unavailable)."""

    resolved: bool
    status: StatusDocument | None
    source: str


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

    - the Linux ROCm build must be Status.success;
    - if arch is non-empty, it must be one of the build's architectures (an
      empty arch means a platform-level gate only).

    The schema-major guard lives in resolve(): an unsupported major is a hard
    failure there, not a soft "not ready" here.
    """
    platform = status.platform(PLATFORM)
    if platform is None:
        return False
    if platform.pipeline_build_status(PIPELINE) != Status.success:
        return False
    if arch and arch not in platform.architectures:
        return False
    return True


def resolve(source: str | None, arch: str) -> Resolution:
    """Load latest.json and decide whether it is ready to act on.

    A missing or unreachable Quartz -- a network error or a malformed/partial
    document (OSError/ValueError, the latter covering json.JSONDecodeError) --
    yields Resolution(False, None, "unavailable") so the workflow simply skips
    the poll and doubles as a Quartz availability canary. Anything else (e.g. an
    AttributeError from a reader change) is a real bug and propagates, rather
    than being mislabeled "unavailable" and retried forever.
    """
    try:
        status = load_status(source)
    except (OSError, ValueError) as exc:
        print(f"Quartz status unavailable: {exc}", file=sys.stderr)
        return Resolution(False, None, "unavailable")
    # An unsupported schema major is permanent, unlike a transient fetch hiccup:
    # retrying will not help, and a new major can move or rename the fields the
    # accessors read, so continuing risks silently misreading the document. Fail
    # loudly so the consumer gets updated, rather than reporting a false "nothing
    # ready" that hides the break on every future poll.
    if not status.schema_version.startswith(SUPPORTED_SCHEMA_MAJOR):
        sys.exit(
            f"schema_version {status.schema_version} unsupported "
            f"(this consumer handles {SUPPORTED_SCHEMA_MAJOR}x); update the consumer."
        )
    if arch_is_good(status, arch):
        return Resolution(True, status, "latest")
    return Resolution(False, status, "not-ready")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "arch",
        nargs="?",
        default="",
        help="optional TheRock arch the build must include (e.g. gfx110X-all); "
        "empty = platform-level gate only",
    )
    # --latest-only and --source are mutually exclusive: the first pins the
    # published latest.json, the second overrides where to read from.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--latest-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resolve latest.json only, with no fallback to an older dated build "
        "(the only supported mode; accepted for explicitness)",
    )
    mode.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="override the status.json URL or path (default: Quartz latest nightly)",
    )
    args = parser.parse_args()

    result = resolve(args.source, args.arch)
    rocm_version = result.status.rocm_version if result.status else ""
    build_date = result.status.build_date if result.status else ""

    # Install URLs straight from Quartz (summary.linux.urls), so the reusable
    # workflow pins the exact index/tarball base Quartz published rather than a
    # hardcoded guess. Empty when there is no status document.
    platform = result.status.platform(PLATFORM) if result.status else None
    wheels_url = (platform.url("wheels") or "") if platform else ""
    tarballs_url = (platform.url("tarballs") or "") if platform else ""

    set_github_outputs(
        resolved=str(result.resolved).lower(),
        rocm_version=rocm_version,
        build_date=build_date,
        source=result.source,
        wheels_url=wheels_url,
        tarballs_url=tarballs_url,
    )

    if result.resolved:
        print(f"resolved {rocm_version} ({build_date}) from {result.source}")
    else:
        print(f"no ready build ({result.source})")


if __name__ == "__main__":
    main()
