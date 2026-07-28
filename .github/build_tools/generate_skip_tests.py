#!/usr/bin/env python3
"""Generate CI skip artifacts for rocm-examples from the unified manifest.

Reads ``skip_manifest.SKIP_MANIFEST`` (the single source of truth) and, filtered
by the requested channel/target/distro, emits:

  * ``skip_tests.txt``  -- ctest names (scope contains "test", ctest key set).
                           Consumed by ``ctest --exclude-from-file``.
  * ``skip_build.txt``  -- repo-relative paths (scope contains "build").
                           Consumed by ``Common/SkipExamples.cmake``.
  * ``SKIP_FROM_TEST``  -- space-separated repo-relative leaf paths (scope test).
  * ``SKIP_FROM_BUILD`` -- space-separated repo-relative leaf paths (scope build).

``SKIP_FROM_*`` carry full paths (e.g. ``Libraries/hipFFT/callback``), not bare
dir names, so a shared leaf name like ``callback`` — which exists under hipFFT,
rocFFT, AND rocProfiler-SDK/counter_collection — only skips the intended one. The
participating parent Makefiles match these paths against their own directory; all
other Makefiles filter bare names and therefore ignore the path entries.

The two ``SKIP_FROM_*`` values are echoed to stdout and, when running under
GitHub Actions, appended to ``$GITHUB_ENV`` so the build/test steps can pass them
on the ``make`` command line. A human-readable summary is printed and, when
available, appended to ``$GITHUB_STEP_SUMMARY``.

NOTE: a bare local ``make``/``make test`` (without running this generator) skips
nothing — CI passes the generated lists explicitly. To reproduce a CI skip
locally, run this script and pass the echoed SKIP_FROM_* on the make line.
"""

import argparse
import os

from skip_manifest import SKIP_MANIFEST


def _entry_applies(entry, channel, target, distro, install_method):
    """Return True if this manifest entry applies to the requested context.

    A filter that is absent from the entry matches everything. A filter that is
    present must contain the requested value.
    """
    if "channels" in entry and channel not in entry["channels"]:
        return False
    if target and "targets" in entry and target not in entry["targets"]:
        return False
    if distro and "distros" in entry and distro not in entry["distros"]:
        return False
    if (
        install_method
        and "install_methods" in entry
        and install_method not in entry["install_methods"]
    ):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate CI skip artifacts for rocm-examples."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__)),
        help="Directory to write skip_tests.txt / skip_build.txt (default: script dir)",
    )
    parser.add_argument(
        "--channel",
        required=True,
        choices=["stable", "nightly"],
        help="CI channel: 'stable' = pinned rocm:7.14 native workflows, "
        "'nightly' = TheRock multi-arch reusable workflow",
    )
    parser.add_argument(
        "--target",
        default="",
        help="GPU target for target-specific skips (e.g. gfx1100)",
    )
    parser.add_argument(
        "--distro",
        default="",
        help="Distro key for distro-specific skips (e.g. ubuntu-24.04)",
    )
    parser.add_argument(
        "--install-method",
        default="",
        help="Install method for method-specific skips (e.g. whl-multi-arch, "
        "tarball-multi-arch, preinstalled). whl and tarball are both the "
        "'nightly' channel but ship different payloads.",
    )
    args = parser.parse_args()

    applicable = [
        e
        for e in SKIP_MANIFEST
        if _entry_applies(
            e, args.channel, args.target, args.distro, args.install_method
        )
    ]

    # Preserve manifest order, de-dup while keeping first occurrence.
    def _unique(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    skip_tests = _unique(
        e["ctest"]
        for e in applicable
        if "test" in e["scope"] and e.get("ctest")
    )
    skip_build_paths = _unique(
        e["path"] for e in applicable if "build" in e["scope"]
    )
    skip_from_test = _unique(
        e["path"] for e in applicable if "test" in e["scope"]
    )
    skip_from_build = _unique(
        e["path"] for e in applicable if "build" in e["scope"]
    )

    os.makedirs(args.output_dir, exist_ok=True)

    tests_path = os.path.join(args.output_dir, "skip_tests.txt")
    with open(tests_path, "w") as f:
        if skip_tests:
            f.write("\n".join(skip_tests) + "\n")

    build_path = os.path.join(args.output_dir, "skip_build.txt")
    with open(build_path, "w") as f:
        if skip_build_paths:
            f.write("\n".join(skip_build_paths) + "\n")

    skip_from_test_str = " ".join(skip_from_test)
    skip_from_build_str = " ".join(skip_from_build)

    # Echo the make variables so they can be captured / eyeballed.
    print(f"SKIP_FROM_TEST={skip_from_test_str}")
    print(f"SKIP_FROM_BUILD={skip_from_build_str}")

    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a") as f:
            f.write(f"SKIP_FROM_TEST={skip_from_test_str}\n")
            f.write(f"SKIP_FROM_BUILD={skip_from_build_str}\n")

    # Human-readable summary.
    label_bits = [f"channel={args.channel}"]
    if args.target:
        label_bits.append(f"target={args.target}")
    if args.distro:
        label_bits.append(f"distro={args.distro}")
    if args.install_method:
        label_bits.append(f"install_method={args.install_method}")
    label = ", ".join(label_bits)

    summary_lines = [f"### rocm-examples skip manifest ({label})", ""]
    if applicable:
        summary_lines.append("| example | scope | reason |")
        summary_lines.append("| --- | --- | --- |")
        for e in applicable:
            summary_lines.append(
                f"| `{e['path']}` | {'+'.join(e['scope'])} | {e['reason']} |"
            )
    else:
        summary_lines.append("_No examples skipped._")
    summary = "\n".join(summary_lines)
    print(summary)

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a") as f:
            f.write(summary + "\n")


if __name__ == "__main__":
    main()
