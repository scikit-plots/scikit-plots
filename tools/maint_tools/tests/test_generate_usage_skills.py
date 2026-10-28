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



#: Every construct whose text CPython, ``typing`` or the platform formats
#: differently by version (Rule 168): enums (3.12), ``typing.`` prefixes and
#: ``ForwardRef`` (3.11 vs 3.12), unions (3.14), paths (POSIX vs Windows),
#: inherited built-in docstrings, positional- and keyword-only markers.
FIXTURE = """\"\"\"Fixture: every construct whose text differs between Pythons.\"\"\"
import dataclasses, enum, pathlib, typing
from typing import Annotated, Optional, Protocol, NamedTuple
__all__ = ["Color", "Flags", "Proto", "Rec", "NT", "Offset", "f", "g", "HOME", "Model"]
class Color(enum.Enum):
    \"\"\"A color.\"\"\"
    RED = "red"
class Flags(enum.IntFlag):
    \"\"\"Flags.\"\"\"
    A = 1
class Proto(Protocol):
    \"\"\"A protocol.\"\"\"
    def run(self) -> None: ...
@dataclasses.dataclass
class Rec:
    \"\"\"A record.\"\"\"
    n: Annotated[int, "ge1"]
    color: Color = Color.RED
    tags: frozenset = frozenset({"b", "a", "c"})
    where: pathlib.Path = pathlib.Path(".")
    items: list = dataclasses.field(default_factory=list)
class NT(NamedTuple):
    \"\"\"A named tuple with forward references.\"\"\"
    a: "str"
    b: "Optional[Color]" = None
class Offset(int):
    pass
def f(x: Annotated[int, "m"], /, y: "Color" = Color.RED, *, z: typing.Optional[typing.List[int]] = None, **kw) -> typing.Optional[int]:
    \"\"\"Do f.\"\"\"
def g(*args: int, key: str = "k") -> "NT":
    \"\"\"Do g.\"\"\"
HOME = pathlib.Path(".")
class Model:
    \"\"\"Pydantic-like: an explicit __signature__ with typing objects.\"\"\"
    import inspect as _i
    __signature__ = _i.Signature([_i.Parameter("n", _i.Parameter.KEYWORD_ONLY, annotation=Annotated[int, "Ge(ge=1)"]),
                                  _i.Parameter("c", _i.Parameter.KEYWORD_ONLY, annotation=typing.List[typing.ForwardRef("Rec")])])
T = typing.TypeVar("T")
def h(cb: typing.Callable[[int, str], None], lit: typing.Literal["a", "b"], u: typing.Union[int, str, None], nested: typing.Dict[str, typing.Optional[typing.Tuple[int, ...]]], t: T, any_: typing.Any) -> "list[T]":
    \"\"\"Do h.\"\"\"
__all__ += ["h"]
"""

#: The table rows FIXTURE renders to: the same bytes on every supported Python.
GOLDEN = r"""| `Color(value)` | enum | A color. |
| `Flags(value)` | enum | Flags. |
| `Proto` | protocol | A protocol. |
| `Rec(n: Annotated[int, 'ge1'], color: scikitplot._usage_skills_fixture.Color = Color.RED, tags: frozenset = fr ...)` | class | A record. |
| `NT(a: 'str', b: 'Optional[Color]' = None)` | class | A named tuple with forward references. |
| `Offset(...)` | class |  |
| `f(x: Annotated[int, 'm'], /, y: 'Color' = Color.RED, *, z: List[int] \| None = None, **kw) -> int \| None` | function | Do f. |
| `g(*args: int, key: str = 'k') -> 'NT'` | function | Do g. |
| `HOME` | Path |  |
| `Model(*, n: Annotated[int, 'Ge(ge=1)'], c: List['Rec'])` | class | Pydantic-like: an explicit __signature__ with typing objects. |
| `h(cb: Callable[[int, str], None], lit: Literal['a', 'b'], u: int \| str \| None, nested: Dict[str, Tuple[int, ...)` | function | Do h. |
"""


def _fixture_rows() -> str:
    import types

    name = "scikitplot._usage_skills_fixture"
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        exec(compile(FIXTURE, name, "exec", dont_inherit=True), module.__dict__)  # noqa: S102 - fixed test text
        api = MODULE.module_api(name)
    finally:
        sys.modules.pop(name, None)
    return "".join(f"| {r['signature']} | {r['kind']} | {r['summary']} |\n" for r in api["rows"])


class CanonicalTextTests(unittest.TestCase):
    """The generated text depends on the source only, not on the interpreter (Rule 168)."""

    @unittest.skipIf(sys.version_info < (3, 9), "typing.Annotated is Python 3.9+")
    def test_the_fixture_renders_to_the_golden_rows_on_this_python(self):
        self.assertEqual(_fixture_rows(), GOLDEN)

    def test_an_enum_is_called_with_one_value(self):
        import enum

        class Color(enum.Enum):
            RED = "red"

        self.assertEqual(MODULE._signature("Color", Color), "`Color(value)`")
        self.assertEqual(MODULE._kind(Color), "enum")

    def test_unions_render_alike_whatever_typing_prints(self):
        import typing

        self.assertEqual(MODULE._annotation(typing.Optional[int]), "int | None")
        self.assertEqual(MODULE._annotation(typing.Union[None, int, str]), "int | str | None")
        if sys.version_info >= (3, 10):
            self.assertEqual(MODULE._annotation(eval("int | None")), "int | None")  # noqa: S307

    def test_a_path_default_names_no_platform_class(self):
        import pathlib

        self.assertEqual(MODULE._default(pathlib.PurePosixPath("a/b")), "Path('a/b')")
        self.assertEqual(MODULE._default(pathlib.PureWindowsPath("a\\b")), "Path('a/b')")
        self.assertEqual(MODULE._kind(pathlib.Path(".")), "Path")

    def test_a_set_default_is_sorted_and_an_address_dropped(self):
        self.assertEqual(MODULE._default(frozenset({"b", "a"})), "frozenset({'a', 'b'})")
        self.assertNotIn("0x", MODULE._default(object()))

    def test_a_builtin_docstring_is_not_inherited(self):
        offset = type("Offset", (int,), {"__module__": "scikitplot.x"})
        self.assertEqual(MODULE._summary(offset), "")

if __name__ == "__main__":
    unittest.main()
