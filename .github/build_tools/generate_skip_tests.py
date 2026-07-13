#!/usr/bin/env python3
"""Generate skip_tests.txt for rocm-examples CI.

Output file is used by ctest --exclude-from-file in the workflow.
Run from repo root or with --output-dir pointing at .github/build_tools.
"""

import argparse
import os

# Tests to skip unconditionally on all targets/distros (upstream bugs in TheRock nightlies).
GLOBAL_SKIP_TESTS = []

# Tests to skip per GPU target (one list per target that has skips)
SKIP_TESTS = {
    # Add more targets as needed, e.g.:
    # "gfx1100": [],
}

# Tests to skip for a specific GPU target + distro combination.
# Keys are "<gpu_target>:<distro_key>", e.g. "gfx1151:sles-15.7".
DISTRO_SKIP_TESTS = {
    # Example:
    # "gfx1151:sles-15.7": ["some_test"],
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate skip_tests.txt for rocm-examples CI."
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__)),
        help="Directory to write skip_tests.txt (default: script dir)",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="GPU target whose skip list to write (e.g. gfx1151)",
    )
    parser.add_argument(
        "--distro",
        default="",
        help="Distro key for distro-specific skips (e.g. sles-15.7)",
    )
    args = parser.parse_args()

    lines = list(GLOBAL_SKIP_TESTS)
    for test in SKIP_TESTS.get(args.target, []):
        if test not in lines:
            lines.append(test)

    if args.distro:
        combo_key = f"{args.target}:{args.distro}"
        distro_lines = DISTRO_SKIP_TESTS.get(combo_key, [])
        for test in distro_lines:
            if test not in lines:
                lines.append(test)

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "skip_tests.txt")
    with open(path, "w") as f:
        if lines:
            f.write("\n".join(lines))
            f.write("\n")

    label = args.target
    if args.distro:
        label = f"{args.target} + {args.distro}"

    if not lines:
        print(f"No tests to skip for {label}.")
    else:
        print(f"Wrote {path} ({len(lines)} tests for {label})")


if __name__ == "__main__":
    main()
