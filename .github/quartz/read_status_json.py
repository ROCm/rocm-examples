#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ---------------------------------------------------------------------------
# VENDORED from ROCm/Quartz: scripts/consumer/read_status_json.py
#   source commit: 4dc5adcefd54564d92d8edbc3915acb55f93f003
#   status.json schema: v2
# Copied unmodified so it can be re-synced from upstream on schema bumps.
# Do not edit here; all rocm-examples logic lives in consume_status.py.
# ---------------------------------------------------------------------------

"""Read helper for Quartz status.json files.

A small, dependency-free API for downstream projects that consume the
status.json files Quartz publishes for TheRock releases. It wraps the parsed
JSON with typed accessors so consumers do not hand-navigate nested dicts.

This module is read-only. It does not validate the document against the schema
and does not write it back; the canonical shape is defined in
docs/status-json/status_json_reference.jsonc. Missing keys are treated the same
as absent (an unreported pipeline or platform simply returns None or an empty
collection), matching how Quartz omits anything not yet reported.

Only the Python standard library is used.

API overview:

    load_status(source) -> StatusDocument

    StatusDocument
    |- rocm_version, build_date, release_type    release metadata
    |- schema_version, created_at, completed_at   more metadata
    |- overall_status                            worst-of rollup across platforms
    |- is_complete                               True once the build has finished
    |- build_id                                  (rocm_version, build_date) dedup key
    |- raw, pipelines                            escape hatch to the raw dicts
    |- platforms()                               names present, e.g. ["linux"]
    `- platform(name) -> PlatformStatus
       |- status                                 worst-of rollup for the platform
       |- architectures                          e.g. ["gfx942", "gfx1201"]
       |- urls / url(kind)                        artifact base URLs
       |- pipeline_build_status(pipeline)         Status value or None
                                                  (compare to Status.success)
       |- pipeline_test_counts(pipeline)          pass/fail counters
       |- native_package_status("rpm" | "deb")    Linux native package status
       `- tarball_url(version, target,            full tarball download URL
                      platform=None, with_tests=False)

Typical use:

    from read_status_json import load_status

    status = load_status()  # latest nightly
    print(status.rocm_version, status.overall_status)

    linux = status.platform("linux")
    if linux and linux.pipeline_build_status("rocm") == "success":
        wheels = linux.url("wheels")
        ...

Power users: the raw tree.

    The typed accessors cover the summary block, which is all most consumers
    need. For per-architecture or per-variant detail (individual run_ids,
    per-variant PyTorch/JAX results, timestamps), drop to the raw dicts. raw is
    the whole parsed document; pipelines is the deep pipelines subtree. Both are
    plain dict / list, so navigate them with ordinary Python. Guard for missing
    keys: anything not yet reported is simply absent. The exact keys are defined
    in docs/status-json/status_json_reference.jsonc (schema v2).

        status = load_status()

        # Everything the typed API exposes is also in status.raw.
        completed_at = status.raw.get("completed_at")

        # Walk the deep pipelines tree for detail the summary does not carry.
        for pipeline_name, pipeline in status.pipelines.items():
            print(pipeline_name, "->", list(pipeline))
"""

import argparse
import json
import urllib.request
from enum import StrEnum
from pathlib import Path

# latest nightly status.json published by Quartz for TheRock releases.
DEFAULT_SOURCE = (
    "https://raw.githubusercontent.com/ROCm/quartz/main/release-nightly/latest.json"
)

# Seconds to wait on a URL fetch before giving up. Without a timeout urlopen can
# block indefinitely if the server accepts the connection then stalls.
DEFAULT_TIMEOUT = 30

# Pipelines that may appear under a platform in the summary block. The set is
# stable across releases (see the schema reference).
PIPELINES = ("rocm", "pytorch", "jax", "native_packages")


# Copied verbatim from the Quartz producer (therock_status_document.Status) so
# this consumer stays dependency-free. Keep the members in sync with it.
class Status(StrEnum):
    """The status values used everywhere a status appears in status.json.

    in_progress is the sole non-terminal state; the rest are terminal.
    """

    in_progress = "in_progress"
    success = "success"
    failure = "failure"
    cancelled = "cancelled"
    skipped = "skipped"

    @property
    def is_terminal(self) -> bool:
        return self is not Status.in_progress


def build_tarball_url(
    base_url: str,
    platform: str,
    version: str,
    target: str,
    with_tests: bool = False,
) -> str:
    """Construct the full URL of one distribution tarball.

    TheRock names distribution tarballs
    therock-dist-{platform}-{target}[-tests]-{version}.tar.gz and publishes
    them under a shared base directory. This builds that name and joins it to
    the base URL.

    platform is linux or windows. target is either "multiarch" or a GPU target
    exactly as it appears in the filename, for example "gfx90a", "gfx94X-dcgpu",
    or "gfx110X-all". Set with_tests to select the variant that bundles the test
    assets (therock-dist-...-tests-...).
    """
    tests_segment = "-tests" if with_tests else ""
    filename = f"therock-dist-{platform}-{target}{tests_segment}-{version}.tar.gz"
    if not base_url.endswith("/"):
        base_url = base_url + "/"
    return base_url + filename


class PlatformStatus:
    """A per-platform view of the summary block (linux or windows)."""

    def __init__(self, name: str, data: dict):
        self.name = name
        self._data = data

    @property
    def status(self) -> str:
        """Worst-of rollup status for this platform."""
        return self._data.get("status", Status.in_progress)

    @property
    def architectures(self) -> list[str]:
        """Requested architectures for this platform, for example gfx942."""
        return self._data.get("architectures", [])

    @property
    def urls(self) -> dict[str, str]:
        """Base URLs for artifacts (tarballs, wheels, rpm, deb, artifacts).

        Each value is a base directory or index page, not a per-file link.
        """
        return self._data.get("urls", {})

    def url(self, kind: str) -> str | None:
        """Return one artifact base URL by kind, or None if it is not present.

        kind is one of: tarballs, wheels, rpm, deb, artifacts.
        """
        return self.urls.get(kind)

    def tarball_url(
        self,
        version: str,
        target: str,
        platform: str | None = None,
        with_tests: bool = False,
    ) -> str | None:
        """Build the full download URL of one distribution tarball.

        version is the ROCm version string (use StatusDocument.rocm_version). target is
        either "multiarch" or a specific GPU target as it appears in the tarball
        name, for example "gfx90a", "gfx94X-dcgpu", or "gfx110X-all". Set
        with_tests to select the tarball that also bundles the test assets.

        platform is the platform segment of the tarball name (linux or windows).
        It defaults to this platform's own name; override it only when the shared
        tarball directory holds builds for a platform other than this one.

        Returns None if this platform has no tarballs base URL (for example a
        platform that was skipped this release).
        """
        base_url = self.url("tarballs")
        if base_url is None:
            return None
        return build_tarball_url(
            base_url,
            platform or self.name,
            version,
            target,
            with_tests=with_tests,
        )

    def pipeline_build_status(self, pipeline: str) -> str | None:
        """Build status for a pipeline (rocm, pytorch, jax), or None if absent.

        For native_packages use native_package_status instead; it has no build
        phase, only per-package-type entries.
        """
        pipeline_summary = self._data.get(pipeline)
        if pipeline_summary is None:
            return None
        build = pipeline_summary.get("build")
        if build is None:
            return None
        return build.get("status")

    def pipeline_test_counts(self, pipeline: str) -> dict[str, int] | None:
        """Per-status test counters for a pipeline, or None if absent.

        The returned dict has keys success, failure, in_progress, cancelled,
        skipped. Counts are one per matrix entry, not one per architecture.
        """
        pipeline_summary = self._data.get(pipeline)
        if pipeline_summary is None:
            return None
        return pipeline_summary.get("test")

    def native_package_status(self, package_type: str) -> str | None:
        """Status of a native package type (rpm or deb), or None if absent.

        Native packages are Linux only, so this returns None on windows.
        """
        native = self._data.get("native_packages")
        if native is None:
            return None
        entry = native.get(package_type)
        if entry is None:
            return None
        return entry.get("status")


class StatusDocument:
    """A parsed status.json document with typed read accessors.

    Wraps the top-level metadata and the summary block. The detailed pipelines
    tree is available as a raw dict via the pipelines property for consumers
    that need per-architecture or per-variant detail.
    """

    def __init__(self, data: dict):
        self._data = data

    @property
    def raw(self) -> dict:
        """The underlying parsed JSON dict."""
        return self._data

    @property
    def schema_version(self) -> str:
        return self._data.get("schema_version", "")

    @property
    def rocm_version(self) -> str:
        return self._data.get("rocm_version", "")

    @property
    def build_date(self) -> str:
        """Build date as YYYYMMDD."""
        return self._data.get("build_date", "")

    @property
    def release_type(self) -> str:
        """Release tier: nightly, rc (prerelease), or dev."""
        return self._data.get("release_type", "")

    @property
    def created_at(self) -> str | None:
        return self._data.get("created_at")

    @property
    def completed_at(self) -> str | None:
        """Completion timestamp, or None while the build is still running."""
        return self._data.get("completed_at")

    @property
    def is_complete(self) -> bool:
        """True once the build has finished (completed_at is set)."""
        return self.completed_at is not None

    @property
    def overall_status(self) -> str:
        """Single worst-of rollup across all platforms."""
        return self._data.get("summary", {}).get("overall_status", Status.in_progress)

    @property
    def pipelines(self) -> dict:
        """The detailed pipelines tree as a raw dict (per-arch, per-variant)."""
        return self._data.get("pipelines", {})

    def platforms(self) -> list[str]:
        """Names of the platforms present in the summary (linux and/or windows)."""
        summary = self._data.get("summary", {})
        return [name for name in ("linux", "windows") if name in summary]

    def platform(self, name: str) -> PlatformStatus | None:
        """Return a per-platform view, or None if the platform is absent."""
        platform_data = self._data.get("summary", {}).get(name)
        if platform_data is None:
            return None
        return PlatformStatus(name, platform_data)

    @property
    def build_id(self) -> tuple[str, str]:
        """Stable identity of this build: (rocm_version, build_date).

        Use it to deduplicate, so a consumer does not act twice on the same
        build across scheduled polls.
        """
        return (self.rocm_version, self.build_date)


def load_status(
    source: str = DEFAULT_SOURCE, timeout: float = DEFAULT_TIMEOUT
) -> StatusDocument:
    """Load a status document from a URL or a local path.

    source defaults to the latest nightly. A value starting with
    http:// or https:// is fetched; anything else is read as a local file path.
    timeout is the per-fetch deadline in seconds for URL sources (ignored for
    local paths).

    Raises:
        urllib.error.URLError: If a URL source cannot be fetched (a socket
            timeout raises URLError wrapping a socket.timeout).
        OSError: If a local path cannot be read.
        json.JSONDecodeError: If the content is not valid JSON.
    """
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=timeout) as response:
            return StatusDocument(json.load(response))
    return StatusDocument(json.loads(Path(source).read_text()))


def _print_summary(status: StatusDocument) -> None:
    """Print the ROCm version and a per-platform summary (used by the CLI)."""
    print(f"ROCm version:  {status.rocm_version}")
    print(f"Release type:  {status.release_type}")
    print(f"Build date:    {status.build_date}")
    print(f"Overall:       {status.overall_status}")

    for name in status.platforms():
        platform = status.platform(name)
        if platform is None:
            continue
        architectures = ", ".join(platform.architectures) or "none"
        print(f"\n{name} ({platform.status})")
        print(f"  architectures: {architectures}")
        for pipeline in PIPELINES:
            if pipeline == "native_packages":
                for package_type in ("rpm", "deb"):
                    package_status = platform.native_package_status(package_type)
                    if package_status is not None:
                        print(f"  native_packages {package_type}: {package_status}")
                continue
            build_status = platform.pipeline_build_status(pipeline)
            if build_status is None:
                continue
            print(f"  {pipeline} build: {build_status}")
            counts = platform.pipeline_test_counts(pipeline)
            if counts:
                rendered = ", ".join(f"{k}={v}" for k, v in counts.items())
                print(f"  {pipeline} test:  {rendered}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        nargs="?",
        default=DEFAULT_SOURCE,
        help="URL or local path to a status.json (default: latest nightly)",
    )
    args = parser.parse_args()
    _print_summary(load_status(args.source))


if __name__ == "__main__":
    main()
