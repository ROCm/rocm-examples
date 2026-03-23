#!/usr/bin/env python3
"""Find source directories whose Makefile examples were also built by CMake.

Scans all Makefiles for 'EXAMPLE := <name>', then checks if that executable
exists anywhere in the CMake build tree. Outputs the list of source directories
whose Makefile targets have a matching CMake-built executable.

This lets us selectively build only Makefiles for examples whose dependencies
CMake confirmed are available, without requiring the directory structures to
match between source and build trees.
"""

import argparse
import os
import re
import stat


def find_makefile_examples(source_dir):
    """Walk source tree, find Makefiles with 'EXAMPLE := <name>'.

    Returns dict mapping example_name -> relative source directory.
    """
    examples = {}
    for root, _dirs, files in os.walk(source_dir):
        if "Makefile" not in files:
            continue
        makefile_path = os.path.join(root, "Makefile")
        try:
            with open(makefile_path) as f:
                for line in f:
                    m = re.match(r"^EXAMPLE\s*:=\s*(\S+)", line)
                    if m:
                        name = m.group(1)
                        rel = os.path.relpath(root, source_dir)
                        examples[name] = rel
                        break
        except OSError:
            continue
    return examples


def find_cmake_executables(build_dir):
    """Walk CMake build tree and collect names of executable files."""
    executables = set()
    for root, _dirs, files in os.walk(build_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                st = os.stat(fpath)
            except OSError:
                continue
            if not (st.st_mode & stat.S_IXUSR):
                continue
            # Skip non-executable artifacts
            if fname.endswith((".so", ".a", ".cmake", ".sh", ".py", ".txt")):
                continue
            if fname.startswith("lib"):
                continue
            executables.add(fname)
    return executables


def main():
    parser = argparse.ArgumentParser(
        description="Find source dirs whose Makefile examples were built by CMake."
    )
    parser.add_argument("--build-dir", required=True, help="CMake build directory")
    parser.add_argument("--source-dir", required=True, help="Source root directory")
    parser.add_argument(
        "--output", required=True, help="Output file (one directory per line)"
    )
    args = parser.parse_args()

    build_dir = os.path.abspath(args.build_dir)
    source_dir = os.path.abspath(args.source_dir)

    makefile_examples = find_makefile_examples(source_dir)
    cmake_executables = find_cmake_executables(build_dir)

    matched_dirs = []
    for name, rel_dir in sorted(makefile_examples.items()):
        if name in cmake_executables:
            matched_dirs.append(rel_dir)

    with open(args.output, "w") as f:
        for d in matched_dirs:
            f.write(d + "\n")

    print(
        f"Found {len(matched_dirs)} Makefile directories matching CMake-built "
        f"executables (out of {len(makefile_examples)} Makefiles total)"
    )


if __name__ == "__main__":
    main()
