# scikitplot/_brand/_kit/_svg.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
A small, safe SVG writer for brand assets.

Brand SVGs are built as a tree of :class:`Element` and serialised by
:func:`to_string`. The writer only produces static drawings: it refuses
scripts, event handlers, ``foreignObject`` and references to anything outside
the document, and it prints numbers the same way on every machine.

See Also
--------
scikitplot._brand._kit._color : the colours written into the documents.

Notes
-----
**User notes:** the same tree always gives the same bytes, so a generated
file can be compared, hashed and committed.

**Developer notes:**

- Names (tags, attributes, ids) must match ``[A-Za-z_][A-Za-z0-9_.:-]*``;
  text and attribute values are escaped.
- References are made only by :func:`url_ref`, to ids of the same document.
  An id prefix (:class:`IdSpace`) keeps the ids of two documents apart when
  both are inlined in one HTML page.
- Numbers are rounded to 3 decimals and printed without a trailing zero;
  ``nan`` and ``inf`` are refused.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Mapping, Sequence, Union

__all__ = [
    "Element",
    "IdSpace",
    "document",
    "fmt",
    "to_string",
    "url_ref",
]

Number = Union[int, float]
Value = Union[str, int, float]

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")
_REFUSED_TAGS = frozenset(
    {"script", "foreignobject", "iframe", "object", "embed", "image", "use", "a"}
)
_REFUSED_ATTRIBUTES = frozenset({"href", "xlink:href", "src", "style"})
_SVG_NS = "http://www.w3.org/2000/svg"


def fmt(value: Number) -> str:
    """
    Print a number for SVG: at most 3 decimals, no trailing zeros.

    Parameters
    ----------
    value : int or float
        A finite number.

    Returns
    -------
    str
        For example ``'12'``, ``'0.5'``, ``'-3.142'``; never ``'-0'``.

    Raises
    ------
    ValueError
        If ``value`` is ``nan`` or infinite.
    TypeError
        If ``value`` is not a number (``bool`` included).

    Examples
    --------
    >>> fmt(2.50000), fmt(-0.0001), fmt(1 / 3)
    ('2.5', '0', '0.333')
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected a number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise ValueError(f"SVG numbers must be finite, got {value}")
    text = f"{round(float(value), 3):.3f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _escape(text: str, quote: bool) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text.replace('"', "&quot;") if quote else text


def _check_name(kind: str, name: str) -> str:
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"not a valid SVG {kind} name: {name!r}")
    return name


class Element:
    """
    One SVG element: a tag, attributes, children and optional text.

    Parameters
    ----------
    tag : str
        Element name, for example ``"path"``.
    attrs : mapping of str to str, int or float, optional
        Attributes in the order they are written. ``None`` values are left out;
        numbers are printed with :func:`fmt`.
    children : iterable of Element, optional
        Child elements.
    text : str, optional
        Text content (escaped).

    Raises
    ------
    ValueError
        For a refused tag or attribute (scripts, event handlers ``on*``,
        external references, ``style``), or an invalid name.

    Notes
    -----
    **Developer notes:** ``style`` is refused because CSS can load external
    resources; presentation attributes cover what brand drawings need.
    """

    __slots__ = ("attrs", "children", "tag", "text")

    def __init__(
        self,
        tag: str,
        attrs: Mapping[str, Value | None] | None = None,
        children: Iterable[Element] | None = None,
        text: str | None = None,
    ) -> None:
        tag = _check_name("tag", tag)
        if tag.lower() in _REFUSED_TAGS:
            raise ValueError(f"the SVG tag <{tag}> is not allowed in brand assets")
        clean: list[tuple[str, str]] = []
        for name, value in (attrs or {}).items():
            _check_name("attribute", name)
            low = name.lower()
            if low.startswith("on") or low in _REFUSED_ATTRIBUTES:
                raise ValueError(
                    f"the SVG attribute {name!r} is not allowed in brand assets"
                )
            if value is None:
                continue
            if isinstance(value, str):
                if "url(" in value.lower() and not re.fullmatch(
                    r"url\(#[A-Za-z_][A-Za-z0-9_.:-]*\)", value
                ):
                    raise ValueError(
                        f"only url(#id) references to the same document are allowed, got {value!r}"
                    )
                clean.append((name, value))
            else:
                clean.append((name, fmt(value)))
        kids = list(children or ())
        for child in kids:
            if not isinstance(child, Element):
                raise TypeError(f"children must be Element, got {type(child).__name__}")
        if text is not None and not isinstance(text, str):
            raise TypeError("text must be a string")
        self.tag = tag
        self.attrs: tuple[tuple[str, str], ...] = tuple(clean)
        self.children: tuple[Element, ...] = tuple(kids)
        self.text = text

    def __repr__(self) -> str:
        return f"Element({self.tag!r}, {len(self.attrs)} attrs, {len(self.children)} children)"


class IdSpace:
    """
    Issue the ids of one document, all with the same prefix.

    Parameters
    ----------
    prefix : str
        Start of every id, for example ``"eukrina-horizontal-light"``.

    Raises
    ------
    ValueError
        If ``prefix`` is not a valid id start.

    Examples
    --------
    >>> ids = IdSpace("logo")
    >>> ids.new("accent"), ids.new("accent")
    ('logo-accent', 'logo-accent-2')
    """

    def __init__(self, prefix: str) -> None:
        self.prefix = _check_name("id", prefix)
        self._used: dict = {}

    def new(self, name: str) -> str:
        """
        Return a new id ``<prefix>-<name>``, numbered if already used.

        Parameters
        ----------
        name : str
            Readable part of the id.

        Returns
        -------
        str
            An id unique within this space.
        """
        base = _check_name("id", f"{self.prefix}-{name}")
        count = self._used.get(base, 0) + 1
        self._used[base] = count
        return base if count == 1 else f"{base}-{count}"


def url_ref(element_id: str) -> str:
    """
    Return ``url(#id)``, the only reference form this writer accepts.

    Parameters
    ----------
    element_id : str
        An id of the same document.

    Returns
    -------
    str
        ``"url(#<id>)"``.
    """
    return f"url(#{_check_name('id', element_id)})"


def document(
    view_box: Sequence[Number],
    children: Iterable[Element],
    *,
    title: str,
    width: Number | None = None,
    height: Number | None = None,
    desc: str | None = None,
    id_space: IdSpace | None = None,
) -> Element:
    """
    Return an ``<svg>`` root with an accessible title.

    Parameters
    ----------
    view_box : sequence of 4 numbers
        ``min-x, min-y, width, height`` in user units; width and height > 0.
    children : iterable of Element
        The drawing.
    title : str
        Accessible name (``<title>``, referenced by ``aria-labelledby``).
    width, height : int or float, optional
        Size of the rendered image; by default the viewBox size.
    desc : str, optional
        Longer description (``<desc>``).
    id_space : IdSpace, optional
        Gives the ids of ``<title>`` and ``<desc>``; ``"svg"`` by default.

    Returns
    -------
    Element
        The root element.

    Raises
    ------
    ValueError
        If the viewBox is not 4 numbers with positive width and height, or
        ``title`` is empty.
    """
    box = [float(v) for v in view_box]
    if len(box) != 4 or box[2] <= 0 or box[3] <= 0:  # noqa: PLR2004
        raise ValueError(
            f"viewBox must be 4 numbers with positive width and height, got {list(view_box)}"
        )
    if not title or not title.strip():
        raise ValueError("an SVG brand asset needs a title")
    ids = id_space or IdSpace("svg")
    title_id = ids.new("title")
    head = [Element("title", {"id": title_id}, text=title.strip())]
    labelled = title_id
    if desc:
        desc_id = ids.new("desc")
        head.append(Element("desc", {"id": desc_id}, text=desc.strip()))
        labelled = f"{title_id} {desc_id}"
    return Element(
        "svg",
        {
            "xmlns": _SVG_NS,
            "width": box[2] if width is None else width,
            "height": box[3] if height is None else height,
            "viewBox": " ".join(fmt(v) for v in box),
            "role": "img",
            "aria-labelledby": labelled,
        },
        head + list(children),
    )


def _write(element: Element, depth: int, out: list[str]) -> None:
    pad = "  " * depth
    attrs = "".join(
        f' {name}="{_escape(value, True)}"' for name, value in element.attrs
    )
    if not element.children and element.text is None:
        out.append(f"{pad}<{element.tag}{attrs}/>")
        return
    if not element.children:
        out.append(
            f"{pad}<{element.tag}{attrs}>{_escape(element.text or '', False)}</{element.tag}>"
        )
        return
    out.append(f"{pad}<{element.tag}{attrs}>")
    if element.text is not None:
        out.append(f"{pad}  {_escape(element.text, False)}")
    for child in element.children:
        _write(child, depth + 1, out)
    out.append(f"{pad}</{element.tag}>")


def to_string(root: Element, *, declaration: bool = False) -> str:
    """
    Serialise an element tree to SVG text.

    Parameters
    ----------
    root : Element
        Usually the result of :func:`document`.
    declaration : bool, default False
        Start with ``<?xml version="1.0" encoding="UTF-8"?>``.

    Returns
    -------
    str
        Two-space indented SVG ending with a newline; identical for identical
        trees.
    """
    if not isinstance(root, Element):
        raise TypeError(f"expected an Element, got {type(root).__name__}")
    out: list[str] = ['<?xml version="1.0" encoding="UTF-8"?>'] if declaration else []
    _write(root, 0, out)
    return "\n".join(out) + "\n"
