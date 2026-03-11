#!/usr/bin/env python3
"""Generate skip_tests.txt for rocm-examples CI.

Output file is used by ctest --exclude-from-file in the workflow.
Run from repo root or with --output-dir pointing at .github/build_tools.
"""

import argparse
import os

# Tests to skip per GPU target (one list per target that has skips)
SKIP_TESTS = {
    "gfx1151": [
        # rccl is not supported on gfx1151 yet
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

# Tests to skip for a specific GPU target + distro combination.
# Keys are "<gpu_target>:<distro_key>", e.g. "gfx1151:sles-15.7".
DISTRO_SKIP_TESTS = {
    # rocjpeg and rocprofiler-sdk segfault on RHEL 8 with TheRock nightlies
    "gfx1100:rhel-8": [
        "rocjpeg_decode",
        "rocjpeg_decode_batched",
        "rocjpeg_decode_perf",
        "rocprofiler-sdk_api_buffered_tracing",
        "rocprofiler-sdk_api_callback_tracing",
        "rocprofiler-sdk_code_object_isa_decode",
        "rocprofiler-sdk_code_object_tracing",
        "rocprofiler-sdk_counter_collection_buffer",
        "rocprofiler-sdk_counter_collection_buffer_device_serialization",
        "rocprofiler-sdk_counter_collection_callback",
        "rocprofiler-sdk_counter_collection_print_functional_counters",
        "rocprofiler-sdk_counter_collection_device_profiling",
        "rocprofiler-sdk_counter_collection_device_profiling_sync",
        "rocprofiler-sdk_external_correlation_id_request",
        "rocprofiler-sdk_intercept_table",
        "rocprofiler-sdk_openmp_target",
        "rocprofiler-sdk_pc_sampling",
    ],
    "gfx1151:rhel-8": [
        "rocjpeg_decode",
        "rocjpeg_decode_batched",
        "rocjpeg_decode_perf",
        "rocprofiler-sdk_api_buffered_tracing",
        "rocprofiler-sdk_api_callback_tracing",
        "rocprofiler-sdk_code_object_isa_decode",
        "rocprofiler-sdk_code_object_tracing",
        "rocprofiler-sdk_counter_collection_buffer",
        "rocprofiler-sdk_counter_collection_buffer_device_serialization",
        "rocprofiler-sdk_counter_collection_callback",
        "rocprofiler-sdk_counter_collection_print_functional_counters",
        "rocprofiler-sdk_counter_collection_device_profiling",
        "rocprofiler-sdk_counter_collection_device_profiling_sync",
        "rocprofiler-sdk_external_correlation_id_request",
        "rocprofiler-sdk_intercept_table",
        "rocprofiler-sdk_openmp_target",
        "rocprofiler-sdk_pc_sampling",
    ],
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

    lines = list(SKIP_TESTS.get(args.target, []))

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
