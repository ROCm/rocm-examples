#!/usr/bin/env python3
"""Find source directories where CMake produced executables.

Scans the CMake build tree for executables and maps them back to source
directories that contain a Makefile. Used to selectively build only the
Makefile examples whose dependencies CMake confirmed are available.
"""

import argparse
import os
import stat
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Find source dirs where CMake built executables."
    )
    parser.add_argument(
        "--build-dir", required=True, help="CMake build directory"
    )
    parser.add_argument(
        "--source-dir", required=True, help="Source root directory"
    )
    parser.add_argument(
        "--output", required=True, help="Output file (one directory per line)"
    )
    args = parser.parse_args()

    build_dir = os.path.abspath(args.build_dir)
    source_dir = os.path.abspath(args.source_dir)

    dirs = set()
    for root, _dirnames, filenames in os.walk(build_dir):
        for fname in filenames:
            fpath = os.path.join(root, fname)
            # Skip non-executable files
            try:
                st = os.stat(fpath)
            except OSError:
                continue
            if not (st.st_mode & stat.S_IXUSR):
                continue
            # Skip shared libraries, scripts, cmake files, etc.
            if fname.endswith((".so", ".a", ".cmake", ".sh", ".py", ".txt")):
                continue
            if fname.startswith("lib"):
                continue

            # Map build path back to source path
            rel = os.path.relpath(root, build_dir)
            source_candidate = os.path.join(source_dir, rel)
            makefile = os.path.join(source_candidate, "Makefile")
            if os.path.isfile(makefile):
                dirs.add(rel)

    sorted_dirs = sorted(dirs)
    with open(args.output, "w") as f:
        for d in sorted_dirs:
            f.write(d + "\n")

    print(f"Found {len(sorted_dirs)} source directories with Makefiles matching CMake build targets")


if __name__ == "__main__":
    main()
