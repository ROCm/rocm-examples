#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for consume_status gate + resolve logic.

read_status_json is Quartz's reusable reader (not vendored here), so point
PYTHONPATH at a Quartz checkout before running:

    git clone --depth 1 https://github.com/ROCm/Quartz /tmp/quartz
    PYTHONPATH=/tmp/quartz/scripts/consumer \\
        python3 -m unittest discover -s .github/quartz/tests

No network: StatusDocument is built from in-memory dicts and load_status is
stubbed for the resolve() cases.
"""

import sys
import unittest
from pathlib import Path

# Put .github/quartz/ on sys.path so `import consume_status` resolves; the reader
# it imports (read_status_json) comes from Quartz via PYTHONPATH, see the module
# docstring.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import consume_status  # noqa: E402
from read_status_json import StatusDocument  # noqa: E402


def make_status(schema="2.0", build_status="success", arches=("gfx110X-all",)):
    """Build a minimal v2-shaped status document for the Linux ROCm gate."""
    return StatusDocument(
        {
            "schema_version": schema,
            "rocm_version": "10.1.0a20260821",
            "build_date": "20260821",
            "summary": {
                "linux": {
                    "status": build_status,
                    "architectures": list(arches),
                    "rocm": {"build": {"status": build_status}},
                }
            },
        }
    )


class ArchIsGoodTest(unittest.TestCase):
    def test_good_latest(self):
        self.assertTrue(consume_status.arch_is_good(make_status(), ""))

    def test_good_latest_with_matching_arch(self):
        self.assertTrue(consume_status.arch_is_good(make_status(), "gfx110X-all"))

    def test_arch_not_in_build(self):
        self.assertFalse(consume_status.arch_is_good(make_status(), "gfx9000"))

    def test_build_not_ready(self):
        self.assertFalse(
            consume_status.arch_is_good(make_status(build_status="in_progress"), "")
        )
        self.assertFalse(
            consume_status.arch_is_good(make_status(build_status="failure"), "")
        )

    def test_missing_linux_platform(self):
        status = StatusDocument({"schema_version": "2.0", "summary": {}})
        self.assertFalse(consume_status.arch_is_good(status, ""))


class ResolveTest(unittest.TestCase):
    def _patch_load(self, fn):
        self._orig = consume_status.load_status
        consume_status.load_status = fn
        self.addCleanup(setattr, consume_status, "load_status", self._orig)

    def test_resolve_ready(self):
        self._patch_load(lambda *a, **k: make_status())
        resolved, status, source = consume_status.resolve(None, "")
        self.assertTrue(resolved)
        self.assertEqual(source, "latest")
        self.assertEqual(status.rocm_version, "10.1.0a20260821")

    def test_resolve_not_ready(self):
        self._patch_load(lambda *a, **k: make_status(build_status="in_progress"))
        resolved, _status, source = consume_status.resolve(None, "")
        self.assertFalse(resolved)
        self.assertEqual(source, "not-ready")

    def test_resolve_unavailable(self):
        def boom(*a, **k):
            raise OSError("quartz down")

        self._patch_load(boom)
        resolved, status, source = consume_status.resolve(None, "")
        self.assertFalse(resolved)
        self.assertIsNone(status)
        self.assertEqual(source, "unavailable")

    def test_resolve_bad_schema_major_exits(self):
        self._patch_load(lambda *a, **k: make_status(schema="3.0"))
        with self.assertRaises(SystemExit):
            consume_status.resolve(None, "")

    def test_resolve_real_bug_propagates(self):
        def boom(*a, **k):
            raise AttributeError("reader renamed a field")

        self._patch_load(boom)
        with self.assertRaises(AttributeError):
            consume_status.resolve(None, "")


if __name__ == "__main__":
    unittest.main()
