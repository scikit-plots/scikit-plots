"""
Tests for ``_sphinx_collection.sections``: real document sections where the
directive's parent allows them, rubric headings everywhere else.

The structural cases run a small probe directive through a real docutils
parse, so the parent node, nested parsing and id registration are genuine.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from docutils import nodes
from docutils.core import publish_doctree, publish_from_doctree
from docutils.parsers.rst import Directive, directives
from docutils.writers import html5_polyglot

ROOT = Path(__file__).resolve().parents[2]
HOST_ROOT = ROOT.parents[2]

PROBE_NAME = "sk-collection-sections-probe"
QUIET = {"report_level": 5, "halt_level": 5, "warning_stream": False}


def _sections_module():
    externals = str(HOST_ROOT / "scikitplot" / "_externals")
    if externals not in sys.path:
        sys.path.insert(0, externals)
    return importlib.import_module("_sphinx_ext._sphinx_collection.sections")


@pytest.fixture(scope="module")
def sections():
    return _sections_module()


class _Logger:
    """Record ``warning`` calls the way a Sphinx logger adapter receives them."""

    def __init__(self):
        self.warnings = []

    def warning(self, message, *args, **kwargs):
        self.warnings.append((message, kwargs))


@pytest.fixture
def probe(sections):
    """Register a probe directive for one test and remove it afterwards."""
    call = SimpleNamespace(sections=[], style="auto", logger=None, results=[])

    class Probe(Directive):
        has_content = False

        def get_location(self):
            return "probe-location"

        def run(self):
            kwargs = {"style": call.style}
            if call.logger is not None:
                kwargs["logger"] = call.logger
            rendered = sections.render_sections(self, call.sections, **kwargs)
            call.results.append(rendered)
            return rendered

    registry = directives._directives
    assert PROBE_NAME not in registry
    directives.register_directive(PROBE_NAME, Probe)
    try:
        yield call
    finally:
        registry.pop(PROBE_NAME, None)


TOP_LEVEL = f"Page\n====\n\nIntro.\n\n.. {PROBE_NAME}::\n"
IN_ADMONITION = f"Page\n====\n\n.. note::\n\n   .. {PROBE_NAME}::\n"
IN_LIST_ITEM = f"Page\n====\n\n* item\n\n  .. {PROBE_NAME}::\n"
IN_TABLE_CELL = (
    "Page\n====\n\n.. list-table::\n\n   * - cell\n\n"
    f"       .. {PROBE_NAME}::\n"
)
NO_TITLE = f"Intro without a title.\n\n.. {PROBE_NAME}::\n"


def _parse(source):
    return publish_doctree(source, settings_overrides=QUIET)


GROUPS = [("Tutorials", "First body.\n\nSecond paragraph."), ("Demos", "* a\n* b")]


# -- constants and structural detection ---------------------------------------


def test_section_styles(sections):
    assert sections.SECTION_STYLES == ("auto", "section", "rubric")
    assert sorted(sections.__all__) == [
        "SECTION_STYLES",
        "attachment_parent",
        "render_sections",
        "sections_allowed",
    ]


def test_attachment_parent_without_state(sections):
    assert sections.attachment_parent(SimpleNamespace()) is None
    assert sections.attachment_parent(SimpleNamespace(state=None)) is None
    assert sections.attachment_parent(None) is None


def test_attachment_parent_uses_docutils_state_parent(sections):
    parent = nodes.section()
    directive = SimpleNamespace(state=SimpleNamespace(parent=parent))
    assert sections.attachment_parent(directive) is parent


def test_attachment_parent_uses_myst_renderer_current_node(sections):
    current = nodes.section()

    class MockState:
        """Like MyST's ``MockState``: unknown attributes raise a non-AttributeError."""

        def __init__(self):
            self._renderer = SimpleNamespace(current_node=current)

        def __getattr__(self, name):
            raise RuntimeError(f"MockingError: {name} not implemented")

    assert sections.attachment_parent(SimpleNamespace(state=MockState())) is current


def test_attachment_parent_myst_renderer_without_current_node(sections):
    state = SimpleNamespace(_renderer=SimpleNamespace(), parent=nodes.section())
    assert sections.attachment_parent(SimpleNamespace(state=state)) is None


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(RuntimeError("mocking"), id="runtime-error"),
        pytest.param(AttributeError("parent"), id="attribute-error"),
        pytest.param(KeyError("parent"), id="key-error"),
    ],
)
def test_attachment_parent_never_raises(sections, error):
    class Hostile:
        _renderer = None

        @property
        def parent(self):
            raise error

    directive = SimpleNamespace(state=Hostile())
    assert sections.attachment_parent(directive) is None
    assert sections.sections_allowed(directive) is False


@pytest.mark.parametrize(
    ("parent", "allowed"),
    [
        pytest.param(nodes.section(), True, id="section"),
        pytest.param(nodes.note(), False, id="admonition"),
        pytest.param(nodes.list_item(), False, id="list-item"),
        pytest.param(nodes.entry(), False, id="table-cell"),
        pytest.param(nodes.container(), False, id="container"),
        pytest.param(nodes.block_quote(), False, id="block-quote"),
        pytest.param(None, False, id="indeterminate"),
        pytest.param("document", False, id="not-a-node"),
    ],
)
def test_sections_allowed_by_parent_type(sections, parent, allowed):
    directive = SimpleNamespace(state=SimpleNamespace(parent=parent))
    assert sections.sections_allowed(directive) is allowed


def test_sections_allowed_under_document_root(sections, probe):
    probe.sections = GROUPS
    document = _parse(NO_TITLE)
    (rendered,) = probe.results
    assert [type(node) for node in rendered] == [nodes.section, nodes.section]
    assert all(node.parent is document for node in rendered)


# -- real sections ------------------------------------------------------------


@pytest.mark.parametrize(
    "style",
    [pytest.param("auto", id="auto"), pytest.param("section", id="section")],
)
def test_top_level_directive_gets_real_sections(probe, style):
    probe.sections, probe.style, probe.logger = GROUPS, style, _Logger()
    document = _parse(TOP_LEVEL)
    (rendered,) = probe.results
    assert [type(node) for node in rendered] == [nodes.section, nodes.section]
    assert [node[0].astext() for node in rendered] == ["Tutorials", "Demos"]
    assert all(isinstance(node[0], nodes.title) for node in rendered)
    # Each section is a navigable target registered with the document.
    assert [node["ids"] for node in rendered] == [["tutorials"], ["demos"]]
    assert document.ids["tutorials"] is rendered[0]
    assert document.ids["demos"] is rendered[1]
    # Bodies are parsed markup, not literal text.
    assert [type(child) for child in rendered[0][1:]] == [
        nodes.paragraph,
        nodes.paragraph,
    ]
    assert isinstance(rendered[1][1], nodes.bullet_list)
    assert not list(document.findall(nodes.rubric))
    assert probe.logger.warnings == []


def test_sections_are_nested_under_the_enclosing_section(probe):
    probe.sections = GROUPS
    document = _parse(TOP_LEVEL)
    (rendered,) = probe.results
    page = document if document.get("title") else rendered[0].parent
    assert isinstance(page, (nodes.document, nodes.section))
    assert all(node.parent is page for node in rendered)


def test_section_order_follows_input_order(probe):
    probe.sections = [("zeta", "z"), ("alpha", "a"), ("mid", "m")]
    _parse(TOP_LEVEL)
    (rendered,) = probe.results
    assert [node[0].astext() for node in rendered] == ["zeta", "alpha", "mid"]


def test_duplicate_labels_get_distinct_ids(probe):
    probe.sections = [("Same", "one"), ("Same", "two"), ("same", "three")]
    document = _parse(TOP_LEVEL)
    (rendered,) = probe.results
    ids = [node["ids"][0] for node in rendered]
    assert all(ids)
    assert len(set(ids)) == 3
    for identifier, node in zip(ids, rendered):
        assert document.ids[identifier] is node
    # docutils may attach an INFO note about the duplicate name; the body is kept.
    bodies = [list(node.findall(nodes.paragraph))[-1].astext() for node in rendered]
    assert bodies == ["one", "two", "three"]
    assert [node[0].astext() for node in rendered] == ["Same", "Same", "same"]


@pytest.mark.parametrize(
    "label",
    [
        pytest.param("日本語 グループ", id="cjk"),
        pytest.param("!!! ???", id="punctuation-only"),
        pytest.param("123", id="digits-only"),
        pytest.param("Ünïcødé & Friends", id="accents-and-ampersand"),
        pytest.param("x" * 5_000, id="very-long"),
    ],
)
def test_unusual_labels_still_get_an_id_and_exact_title(probe, label):
    probe.sections = [(label, "body")]
    document = _parse(TOP_LEVEL)
    ((section,),) = probe.results
    assert isinstance(section, nodes.section)
    assert section[0].astext() == label
    assert len(section["ids"]) == 1
    assert document.ids[section["ids"][0]] is section


@pytest.mark.parametrize(
    "style",
    [pytest.param("auto", id="sections"), pytest.param("rubric", id="rubrics")],
)
@pytest.mark.parametrize(
    "label",
    [
        pytest.param("<script>alert(1)</script>", id="script-tag"),
        pytest.param('"><img src=x onerror=alert(1)>', id="attribute-breakout"),
        pytest.param("`role`_ **bold** :ref:`x`", id="rst-inline-markup"),
        pytest.param(".. raw:: html", id="directive-syntax"),
    ],
)
def test_hostile_labels_are_plain_text(probe, style, label):
    probe.sections, probe.style = [(label, "body")], style
    document = _parse(TOP_LEVEL)
    (rendered,) = probe.results
    heading = rendered[0][0] if style == "auto" else rendered[0]
    assert isinstance(heading, nodes.title if style == "auto" else nodes.rubric)
    # The label is one literal text node: no inline markup, no raw HTML.
    assert [type(child) for child in heading.children] == [nodes.Text]
    assert heading.astext() == label
    assert not list(document.findall(nodes.raw))
    output = publish_from_doctree(
        document, writer=html5_polyglot.Writer(), settings_overrides=QUIET
    ).decode("utf-8")
    assert "<script>alert(1)" not in output
    assert "<img src=x" not in output


# -- rubric fallback ----------------------------------------------------------


def _assert_rubrics(rendered, labels):
    rubrics = [node for node in rendered if isinstance(node, nodes.rubric)]
    assert [node.astext() for node in rubrics] == labels
    assert not any(isinstance(node, nodes.section) for node in rendered)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(IN_ADMONITION, id="admonition"),
        pytest.param(IN_LIST_ITEM, id="list-item"),
        pytest.param(IN_TABLE_CELL, id="table-cell"),
    ],
)
@pytest.mark.parametrize(
    "style",
    [pytest.param("auto", id="auto"), pytest.param("section", id="section")],
)
def test_nested_directive_falls_back_to_rubrics(probe, source, style):
    probe.sections, probe.style = GROUPS, style
    document = _parse(source)
    (rendered,) = probe.results
    _assert_rubrics(rendered, ["Tutorials", "Demos"])
    assert [type(node) for node in rendered] == [
        nodes.rubric,
        nodes.paragraph,
        nodes.paragraph,
        nodes.rubric,
        nodes.bullet_list,
    ]
    # No section was smuggled into a parent that cannot hold one.
    page_sections = list(document.findall(nodes.section))
    assert len(page_sections) <= 1
    assert not list(document.findall(nodes.system_message))


def test_rubric_style_is_honoured_where_sections_would_be_valid(probe):
    probe.sections, probe.style, probe.logger = GROUPS, "rubric", _Logger()
    _parse(TOP_LEVEL)
    (rendered,) = probe.results
    _assert_rubrics(rendered, ["Tutorials", "Demos"])
    assert probe.logger.warnings == []


def test_unlabelled_sections_render_bodies_without_headings(probe):
    probe.sections, probe.style = [("", "Only body."), ("", "More body.")], "section"
    probe.logger = _Logger()
    _parse(TOP_LEVEL)
    (rendered,) = probe.results
    assert [type(node) for node in rendered] == [nodes.paragraph, nodes.paragraph]
    assert [node.astext() for node in rendered] == ["Only body.", "More body."]
    assert probe.logger.warnings == []


@pytest.mark.parametrize(
    "style",
    [
        pytest.param("auto", id="auto"),
        pytest.param("section", id="section"),
        pytest.param("rubric", id="rubric"),
    ],
)
def test_no_sections_render_nothing(probe, style):
    probe.sections, probe.style = [], style
    _parse(TOP_LEVEL)
    assert probe.results == [[]]


def test_empty_body_keeps_its_heading(probe):
    probe.sections, probe.style = [("Empty", "")], "rubric"
    _parse(TOP_LEVEL)
    (rendered,) = probe.results
    assert [type(node) for node in rendered] == [nodes.rubric]


def test_rendered_nodes_are_detached_from_the_scratch_container(probe):
    probe.sections = GROUPS
    document = _parse(IN_ADMONITION)
    (rendered,) = probe.results
    (note,) = document.findall(nodes.note)
    assert all(node.parent is note for node in rendered)
    assert not list(document.findall(nodes.container))


# -- warnings -----------------------------------------------------------------


def test_refused_section_request_warns_once_with_location(probe):
    probe.sections, probe.style, probe.logger = GROUPS, "section", _Logger()
    _parse(IN_ADMONITION)
    (rendered,) = probe.results
    _assert_rubrics(rendered, ["Tutorials", "Demos"])
    ((message, kwargs),) = probe.logger.warnings
    assert "':section-style: section' was requested" in message
    assert "Falling back to rubric" in message
    assert kwargs == {"location": "probe-location"}


@pytest.mark.parametrize(
    ("style", "with_logger"),
    [
        pytest.param("auto", True, id="auto-is-silent"),
        pytest.param("section", False, id="section-without-logger"),
        pytest.param("rubric", True, id="rubric-is-silent"),
    ],
)
def test_fallback_is_silent_unless_sections_were_demanded(probe, style, with_logger):
    logger = _Logger()
    probe.sections, probe.style = GROUPS, style
    probe.logger = logger if with_logger else None
    _parse(IN_ADMONITION)
    (rendered,) = probe.results
    _assert_rubrics(rendered, ["Tutorials", "Demos"])
    assert logger.warnings == []


# -- construction failure -----------------------------------------------------


class _FailingDocument:
    def note_implicit_target(self, *args, **kwargs):
        raise RuntimeError("structure refused")


def _fake_directive(parent):
    """A directive whose document refuses section targets but can still parse."""

    def nested_parse(lines, offset, node):
        for line in lines:
            node += nodes.paragraph(text=line)

    state = SimpleNamespace(
        parent=parent, document=_FailingDocument(), nested_parse=nested_parse
    )
    return SimpleNamespace(state=state, get_location=lambda: "fake-location")


@pytest.mark.parametrize(
    ("style", "with_logger", "warns"),
    [
        pytest.param("section", True, True, id="section-reports-the-gap"),
        pytest.param("section", False, False, id="section-without-logger"),
        pytest.param("auto", True, False, id="auto-is-silent"),
    ],
)
def test_section_construction_failure_falls_back_to_rubrics(
    sections, style, with_logger, warns
):
    logger = _Logger()
    directive = _fake_directive(nodes.section())
    rendered = sections.render_sections(
        directive,
        [("One", "body one"), ("Two", "body two")],
        style=style,
        logger=logger if with_logger else None,
    )
    assert [type(node) for node in rendered] == [
        nodes.rubric,
        nodes.paragraph,
        nodes.rubric,
        nodes.paragraph,
    ]
    assert [node.astext() for node in rendered] == ["One", "body one", "Two", "body two"]
    if warns:
        ((message, kwargs),) = logger.warnings
        assert "could not build document sections (structure refused)" in message
        assert kwargs == {"location": "fake-location"}
    else:
        assert logger.warnings == []


def test_default_style_is_auto_and_default_logger_is_silent(sections):
    directive = _fake_directive(nodes.note())
    rendered = sections.render_sections(directive, [("One", "body")])
    assert [type(node) for node in rendered] == [nodes.rubric, nodes.paragraph]


def test_body_lines_are_passed_to_the_parser_in_order(sections):
    seen = []

    def nested_parse(lines, offset, node):
        seen.append((list(lines), offset, lines.source(0) if len(lines) else None))

    state = SimpleNamespace(parent=nodes.note(), nested_parse=nested_parse)
    directive = SimpleNamespace(state=state)
    sections.render_sections(directive, [("A", "l1\nl2\r\nl3"), ("B", "")])
    assert seen == [
        (["l1", "l2", "l3"], 0, "<collection-section>"),
        ([], 0, None),
    ]
