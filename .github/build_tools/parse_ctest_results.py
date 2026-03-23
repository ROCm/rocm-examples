#!/usr/bin/env python3
"""Parse ctest JUnit XML results and extract test names that actually ran.

Used by the Makefile test step to only run tests that ctest executed,
ensuring parity between CMake and Makefile test sets.
"""

import argparse
import sys
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(
        description="Extract test names from ctest JUnit XML results."
    )
    parser.add_argument(
        "--junit", required=True, help="Path to ctest JUnit XML file"
    )
    parser.add_argument(
        "--output", required=True, help="Output file (one test name per line)"
    )
    args = parser.parse_args()

    try:
        tree = ET.parse(args.junit)
    except FileNotFoundError:
        print(f"Warning: {args.junit} not found (ctest may not have run)")
        print("Creating empty allow list — no Makefile tests will run")
        with open(args.output, "w") as f:
            pass
        return
    except ET.ParseError as e:
        print(f"Error parsing {args.junit}: {e}", file=sys.stderr)
        sys.exit(1)

    root = tree.getroot()
    test_names = []
    for testcase in root.iter("testcase"):
        name = testcase.get("name")
        if name:
            test_names.append(name)

    with open(args.output, "w") as f:
        for name in test_names:
            f.write(name + "\n")

    print(f"Extracted {len(test_names)} test names from ctest JUnit results")


if __name__ == "__main__":
    main()
