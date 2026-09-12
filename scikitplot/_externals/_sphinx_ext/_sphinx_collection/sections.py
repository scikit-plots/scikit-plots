"""
Section rendering for collection directives, with a safe fallback.

A grouped collection wants real document sections: they get permalinks,
appear in the local table of contents, and are what screen readers and
"jump to heading" navigation act on. A rubric is only a styled paragraph --
it looks like a heading and behaves like nothing.

But a directive cannot always create a section. Docutils only accepts a
section where the document structure allows one; inside an admonition, a
card, a list item or a table cell, returning a section produces
``Unexpected section title`` or a malformed tree. And a directive has no
control over where an author puts it.

So this module does both, and decides per invocation:

* **Real sections** when the directive's parent is a document or a section.
* **Rubrics** otherwise, or if section construction raises for any reason.

The fallback is silent by default. An author who wrote
``:group-by: playlist`` inside a dropdown asked for grouping, not for a
build error about docutils' structural model, and the rubric gives them
exactly the reading experience they were after.

Notes
-----
**User-focused.** ``:section-style:`` takes ``auto`` (the default),
``section`` or ``rubric``. ``auto`` gives real headings wherever they are
possible. ``section`` demands them and warns if the context refuses.
``rubric`` always uses rubrics, which is useful when a gallery sits inside
a page whose heading levels you do not want disturbed.

**Developer-focused.** Detection is structural, not a guess: it inspects
the actual parent node the directive's output will be attached to. The
``try``/``except`` around construction is a second line of defence, not the
mechanism -- if it ever fires, the structural check missed a case and that
is worth knowing, so ``section`` mode reports it.
"""

from __future__ import annotations

from typing import Any, Sequence

from docutils import nodes
from docutils.statemachine import StringList

__all__ = [
    "SECTION_STYLES",
    "attachment_parent",
    "render_sections",
    "sections_allowed",
]

#: Accepted values for a ``:section-style:`` option.
SECTION_STYLES = ("auto", "section", "rubric")


def attachment_parent(directive: Any) -> Any:
    """
    Find the node this directive's output will be attached to.

    Parameters
    ----------
    directive : docutils.parsers.rst.Directive
        The directive being executed.

    Returns
    -------
    docutils.nodes.Node or None
        The parent node, or ``None`` if it cannot be determined.

    Notes
    -----
    The two parsers expose this differently and neither is optional:

    * reStructuredText uses docutils' own ``State``, where the answer is
      ``state.parent``.
    * MyST substitutes a ``MockState`` whose ``__getattr__`` *raises*
      ``MockingError`` for anything it has not implemented -- including
      ``parent``. A plain ``getattr(state, "parent", None)`` does not help,
      because the default only applies when ``AttributeError`` is raised.
      MyST's equivalent is ``state._renderer.current_node``.

    Probing for ``_renderer`` first is safe in both directions: it is a real
    attribute on ``MockState`` (so no ``__getattr__`` call), and absent on
    docutils' ``State`` (which has no ``__getattr__``, so the default
    applies). The ``try`` around ``state.parent`` then covers any third
    parser that behaves like neither.
    """
    state = getattr(directive, "state", None)
    if state is None:
        return None
    renderer = getattr(state, "_renderer", None)
    if renderer is not None:
        return getattr(renderer, "current_node", None)
    try:
        return state.parent
    except Exception:  # noqa: BLE001 - detection must never raise
        return None


def sections_allowed(directive: Any) -> bool:
    """
    Test whether this directive may create document sections.

    Parameters
    ----------
    directive : docutils.parsers.rst.Directive
        The directive being executed.

    Returns
    -------
    bool
        ``True`` when the node the directive's output attaches to is a
        document or a section, which are the only parents docutils accepts
        a section under. An indeterminate parent returns ``False``, so an
        unknown parser degrades to rubrics rather than to a broken tree.

    Notes
    -----
    Checking the real attachment point is exact: a directive at the top
    level of a page has a ``document`` or ``section`` parent, while one
    inside an admonition, card, list item or table cell does not.
    """
    return isinstance(attachment_parent(directive), (nodes.document, nodes.section))


def _make_section(directive: Any, label: str, source: str) -> nodes.section:
    """
    Build one titled section containing parsed markup.

    Parameters
    ----------
    directive : docutils.parsers.rst.Directive
        The directive being executed, used for its parser state.
    label : str
        The section heading text.
    source : str
        Markup source for the section body, in the calling page's language.

    Returns
    -------
    docutils.nodes.section
        The populated section, with an id registered so it gets a permalink
        and a table-of-contents entry.
    """
    section = nodes.section()
    title = nodes.title(text=label)
    section += title

    # Registering the id is what turns the heading into a link target and a
    # toctree entry; a section without one renders but cannot be navigated
    # to, which would forfeit the only reason to prefer sections.
    document = directive.state.document
    section["names"].append(nodes.fully_normalize_name(label))
    document.note_implicit_target(section, section)

    directive.state.nested_parse(
        StringList(source.splitlines(), source="<collection-section>"),
        0,
        section,
    )
    return section


def _as_rubrics(
    directive: Any, sections: Sequence[tuple[str, str]]
) -> list[nodes.Node]:
    """
    Render sections as rubric-and-body pairs.

    Parameters
    ----------
    directive : docutils.parsers.rst.Directive
        The directive being executed.
    sections : sequence of (str, str)
        Label and markup source for each section.

    Returns
    -------
    list of docutils.nodes.Node
        A rubric followed by the parsed body, for each section. An empty
        label emits no rubric, which is how the ungrouped case renders
        through the same code path.
    """
    container = nodes.container()
    for label, source in sections:
        if label:
            rubric = nodes.rubric(text=label)
            container += rubric
        directive.state.nested_parse(
            StringList(source.splitlines(), source="<collection-section>"),
            0,
            container,
        )
    return list(container.children)


def render_sections(
    directive: Any,
    sections: Sequence[tuple[str, str]],
    style: str = "auto",
    logger: Any = None,
) -> list[nodes.Node]:
    """
    Render labelled sections, preferring real sections over rubrics.

    Parameters
    ----------
    directive : docutils.parsers.rst.Directive
        The directive being executed.
    sections : sequence of (str, str)
        Label and markup source for each section. A label of ``""`` means
        "no heading", so an ungrouped collection uses this function too.
    style : {'auto', 'section', 'rubric'}, optional
        ``auto`` uses real sections where the context allows and rubrics
        otherwise. ``section`` asks for real sections and warns when the
        context refuses. ``rubric`` always uses rubrics.
    logger : logging.Logger, optional
        Used only to report a refused ``section`` request. Omitting it makes
        the fallback silent.

    Returns
    -------
    list of docutils.nodes.Node
        The rendered nodes.

    Notes
    -----
    Nothing here raises. A collection that cannot be sectioned is still
    rendered -- the grouping is a presentation choice, and failing a build
    over it would be a worse outcome than a heading that is styled rather
    than structural.
    """
    unlabelled = all(not label for label, _ in sections)
    if style == "rubric" or unlabelled:
        return _as_rubrics(directive, sections)

    if not sections_allowed(directive):
        if style == "section" and logger is not None:
            logger.warning(
                "':section-style: section' was requested, but this directive "
                "is not at a place in the document where a section is valid "
                "(it is inside another element). Falling back to rubric "
                "headings; use ':section-style: auto' to silence this.",
                location=directive.get_location(),
            )
        return _as_rubrics(directive, sections)

    try:
        return [_make_section(directive, label, source) for label, source in sections]
    except Exception as exc:  # noqa: BLE001 - fallback must be total
        # The structural check above should make this unreachable. If it
        # fires, it missed a case -- report it under an explicit request so
        # the gap is discoverable, but never fail the build for it.
        if style == "section" and logger is not None:
            logger.warning(
                f"could not build document sections ({exc}); "
                f"falling back to rubric headings.",
                location=directive.get_location(),
            )
        return _as_rubrics(directive, sections)
