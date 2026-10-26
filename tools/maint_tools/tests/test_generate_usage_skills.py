# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for tools/maint_tools/generate_usage_skills.py that need no built package.

The drift check against the installed package is
``scikitplot/tests/test_usage_skills.py``.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "generate_usage_skills.py"
SPEC = importlib.util.spec_from_file_location("generate_usage_skills", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
REPO = Path(__file__).resolve().parents[3]


class TextTests(unittest.TestCase):
    def test_a_rest_title_is_skipped(self):
        doc = "scikitplot.corpus\n=================\n\nIngest documents.\n\nMore."
        self.assertEqual(MODULE._first_paragraph(doc), "Ingest documents.")

    def test_roles_and_footnotes_become_plain(self):
        self.assertEqual(
            MODULE._plain("Like :class:`~numpy.random.Generator` [1]_ and :py:mod:`logging`."),
            "Like `Generator` and `logging`.",
        )
        self.assertEqual(MODULE._plain(":func:`the title <a.b>`"), "`the title`")

    def test_examples_are_taken_from_the_section(self):
        doc = "Summary.\n\nExamples\n--------\n>>> 1 + 1\n2\n\nNotes\n-----\nx"
        self.assertEqual(MODULE._examples(doc), ">>> 1 + 1\n2")
        self.assertEqual(MODULE._examples("Summary only."), "")


class StableSignatureTests(unittest.TestCase):
    def test_a_set_default_is_sorted_and_an_address_dropped(self):
        def helper():
            return None

        def f(a=frozenset({"b", "a", "c"}), b=helper):
            return a, b

        text = MODULE._signature("f", f)
        self.assertIn("frozenset({'a', 'b', 'c'})", text)
        self.assertNotIn(" at 0x", text)

    def test_a_long_signature_is_shortened(self):
        def g(aaaaaaaaaaaaaaaaaaaa=1, bbbbbbbbbbbbbbbbbbbbbbb=2, ccccccccccccccccccccccc=3, ddddddddddddddddddddddddd=4, eeeeeee=5):
            return None

        self.assertLessEqual(len(MODULE._signature("g", g)), MODULE.SIGNATURE_MAX + 4)


class NamingTests(unittest.TestCase):
    def test_root_and_subpackages(self):
        derived = {"annoy": "annoy", "cexternals/_annoy": "cexternals-annoy"}
        self.assertEqual(MODULE.skill_name("scikitplot", derived), "scikitplot")
        self.assertEqual(MODULE.skill_name("scikitplot.annoy", derived), "scikitplot-annoy")
        self.assertEqual(
            MODULE.skill_name("scikitplot.cexternals._annoy", derived), "scikitplot-cexternals-annoy"
        )
        with self.assertRaises(MODULE.SkillError):
            MODULE.skill_name("scikitplot.nope", derived)


class PolicyTests(unittest.TestCase):
    def test_the_repository_policy_loads(self):
        policy = MODULE.load_policy(REPO)
        self.assertIn("scikitplot", policy["usage_modules"])
        self.assertIn("scikitplot.externals", policy["usage_excluded"])


class RenderTests(unittest.TestCase):
    def test_a_skill_holds_its_frontmatter_api_and_example(self):
        api = {
            "doc": "Does things.",
            "rows": [
                {"name": "f", "kind": "function", "signature": "`f(x)`", "summary": "Do f.", "doc": ""}
            ],
            "children": [],
            "example": ">>> f(1)",
            "example_source": "scikitplot.misc.f",
        }
        original = MODULE.distributions_for
        MODULE.distributions_for = lambda module: []
        original_trees = MODULE._dist_trees
        MODULE._dist_trees = lambda: []
        try:
            files = MODULE.render("scikitplot.misc", "scikitplot-misc", api, [])
        finally:
            MODULE.distributions_for = original
            MODULE._dist_trees = original_trees
        skill = files["SKILL.md"]
        self.assertTrue(skill.startswith("---\nname: scikitplot-misc\n"))
        self.assertIn('scikitplot-module: "scikitplot.misc"', skill)
        self.assertIn("| `f` | function | Do f. |", skill)
        self.assertIn(">>> f(1)", skill)
        self.assertIn("| `f(x)` | function | Do f. |", files["references/API.md"])


if __name__ == "__main__":
    unittest.main()
