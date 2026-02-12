#!/usr/bin/env python3
"""Generate skip_tests.txt for rocm-examples CI.

Output file is used by ctest --exclude-from-file in the workflow.
Run from repo root or with --output-dir pointing at .github/build_tools.
"""

import argparse
import os

# Tests to skip per GPU target (one list per target that has skips)
SKIP_TESTS = {
    # rccl is not supported on gfx1151 yet
    "gfx1151": [
        "rccl_allgather",
        "rccl_allreduce",
        "rccl_broadcast",
        "rccl_buffer_registration",
        "rccl_device_api",
        "rccl_gradient_allreduce",
        "rccl_reduce",
        "rccl_reducescatter",
        "rccl_send_recv",
    ],
    # Add more targets as needed, e.g.:
    # "gfx1100": [],
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
    args = parser.parse_args()

    lines = SKIP_TESTS.get(args.target, [])

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "skip_tests.txt")
    with open(path, "w") as f:
        if lines:
            f.write("\n".join(lines))
            f.write("\n")
    if not lines:
        print(f"No tests to skip for {args.target}.")
    else:
        print(f"Wrote {path} ({len(lines)} tests for {args.target})")


if __name__ == "__main__":
    main()
