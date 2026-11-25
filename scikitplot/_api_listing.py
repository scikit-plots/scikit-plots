# scikitplot/_api_listing.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The public API of an installed scikit-plots module, read from the package itself.

One renderer for every place that lists the API: ``python -m scikitplot api``
(what an assistant runs to see the version the user has) and the usage-skill
generator (``tools/maint_tools/generate_usage_skills.py``). A module's public
names are its ``__all__``; each is described by its kind, its call signature
and the first sentence of its own docstring.

Notes
-----
**User note.** Run ``python -m scikitplot api scikitplot.corpus`` to list a
module's names, and ``--name ZipReader`` for one name's full signature and
docstring. The output is read from the installed package, so it matches the
version the project uses.

**Developer note.** Every text is built from the object's facts (parameter
kinds, defaults, annotations, the class's own docstring), never from a format
CPython, :mod:`typing` or the platform owns (``str(Signature)``, a typing
repr, ``PosixPath``): those change between Python versions, and the generated
skills are compared byte for byte (Rule 168). Only modules named
``scikitplot`` or ``scikitplot.<...>`` are imported, so the listing cannot be
used to import arbitrary code. Standard library only; nothing is written.
"""

from __future__ import annotations

import enum
import importlib
import importlib.util
import inspect
import pathlib
import pkgutil
import re
import types
import typing
from typing import Any

__all__ = [
    "SIGNATURE_MAX",
    "ApiListingError",
    "describe",
    "first_paragraph",
    "kind",
    "module_listing",
    "name_detail",
    "own_doc",
    "render_markdown",
    "render_text",
    "signature",
    "summary",
]

#: Longest signature shown in a listing; ``name_detail`` shows it whole.
SIGNATURE_MAX = 110

_MODULE_RE = re.compile(r"scikitplot(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
_SECTION = re.compile(r"^([A-Z][A-Za-z ]+)\n-{3,}\s*$", re.MULTILINE)
_TITLE = re.compile(r"\A[^\n]+\n([=\-~^*#\"'`+])\1{2,}[ \t]*(\n|\Z)")
_ROLE = re.compile(r":(?:[a-z]+:)?[a-z]+:`(~?)([^`<]*?)(?:\s*<([^<>`]+)>)?`")
_FOOTNOTE = re.compile(r"\s*\[(?:\d+|#\w*|\*)\]_")
_TYPING_PREFIX = re.compile(r"(?<![\w.])typing(?:_extensions)?\.")
_FORWARD_REF = re.compile(r"""ForwardRef\((['"])(.*?)\1(?:, [^)]*)?\)""")
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
_P = inspect.Parameter
#: ``int | str`` (3.10+) has its own origin; ``typing.Union`` is the same object from 3.14.
_UNION_TYPE = getattr(types, "UnionType", None)
#: ``typing.Annotated`` exists from Python 3.9.
_ANNOTATED = getattr(typing, "Annotated", None)


class ApiListingError(ValueError):
    """
    A module cannot be listed: a bad name, an import failure, or an undeclared API.

    Parameters
    ----------
    message : str
        What failed, naming the module.
    reason : {"name", "import", "api"}
        ``"name"``: not ``scikitplot`` or a module below it; ``"import"``: the
        module does not import here (often: the installed release predates
        it); ``"api"``: no ``__all__``, a declared name that does not resolve,
        or an unknown ``name``.
    """

    def __init__(self, message: str, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


# --------------------------------------------------------------------------- text


def _plain(text: str) -> str:
    """
    Return reST text as plain text.

    ``:py:class:`~numpy.random.Generator``` becomes ```Generator```, as Sphinx
    shows it (``~`` keeps the last part); a role with an explicit title keeps
    the title; footnote references (``[1]_``) are dropped.
    """

    def role(match: re.Match) -> str:
        tilde, target, explicit = match.groups()
        if explicit:
            return f"`{target.strip()}`"
        target = target.strip().lstrip(".")
        if tilde:
            target = target.rsplit(".", 1)[-1]
        return f"`{target}`"

    return _FOOTNOTE.sub("", _ROLE.sub(role, text))


def first_paragraph(doc: str) -> str:
    """Return the first paragraph, after a reST title, before any numpydoc section."""
    doc = inspect.cleandoc(doc or "")
    doc = _TITLE.sub("", doc, count=1).lstrip("\n")
    head = _SECTION.split(doc, maxsplit=1)[0]
    return _plain(" ".join(head.strip().split("\n\n", 1)[0].split()))


def own_doc(obj: object) -> str:
    """
    Return the docstring *obj* or a scikit-plots base of it defines, else ``""``.

    A class without one of its own would otherwise show a built-in base's text
    (``int([x]) -> integer``), which belongs to the interpreter and changes
    between versions.
    """
    if inspect.isclass(obj):
        for klass in obj.__mro__:
            if str(getattr(klass, "__module__", "")).split(".")[0] != "scikitplot":
                break
            doc = klass.__dict__.get("__doc__")
            if isinstance(doc, str) and doc.strip():
                return inspect.cleandoc(doc)
        return ""
    return inspect.getdoc(obj) or ""


def summary(obj: object) -> str:
    """One line: the first sentence of a function's, class's or module's docstring."""
    if not (inspect.isroutine(obj) or inspect.isclass(obj) or inspect.ismodule(obj)):
        return ""
    doc = inspect.getdoc(obj) if inspect.ismodule(obj) else own_doc(obj)
    paragraph = first_paragraph(doc or "")
    return re.split(r"(?<=[.!?])\s", paragraph, maxsplit=1)[0]


def _is_protocol(obj: object) -> bool:
    return inspect.isclass(obj) and bool(getattr(obj, "_is_protocol", False))


# One return per kind, in order of precedence.
def kind(obj: object) -> str:  # noqa: PLR0911
    """Return what a public name is: ``class``, ``function``, ``exception``, ..."""
    if _is_protocol(obj):
        return "protocol"
    if inspect.isclass(obj):
        if issubclass(obj, BaseException):
            return "exception"
        return "enum" if issubclass(obj, enum.Enum) else "class"
    if type(obj).__module__ in ("typing", "typing_extensions"):
        return "type alias"
    if inspect.isroutine(obj):
        return "function"
    if inspect.ismodule(obj):
        return "module"
    if isinstance(obj, pathlib.PurePath):
        return "Path"  # PosixPath or WindowsPath, by the platform
    return type(obj).__name__


def _default(value: object) -> str:
    """
    Return the text of a default value, the same in every process and on every platform.

    A set's repr follows string hashing, which differs per process; an
    object's repr may hold its memory address; a path's repr names the
    platform's class.
    """
    if isinstance(value, (set, frozenset)):
        items = ", ".join(sorted(_default(item) for item in value))
        return (
            f"{type(value).__name__}({{{items}}})"
            if items
            else f"{type(value).__name__}()"
        )
    if isinstance(value, pathlib.PurePath):
        return f"Path({value.as_posix()!r})"
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    return _ADDRESS.sub("", repr(value))


# One case per typing construct, as the generated skills have always rendered them.
def _annotation(value: object) -> str:  # noqa: PLR0911, PLR0912
    """
    Return the text of an annotation, built from its parts on every Python.

    A string (``from __future__ import annotations``, a quoted name) is shown
    quoted, as written; a ``ForwardRef`` as its string; ``None`` as ``None``;
    a union as ``A | B``; ``Annotated``, ``Literal``, ``Callable`` and any
    other generic as ``Name[args]`` with each argument rendered the same way;
    a class by its dotted name (only the name for a built-in); a type variable
    by its name. Only an unknown construct falls back to its repr, without
    ``typing.`` prefixes.
    """
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, typing.ForwardRef):
        return repr(value.__forward_arg__)
    if value is None or value is type(None):
        return "None"
    if value is Ellipsis:
        return "..."
    if isinstance(value, list):  # the parameter list of Callable[[A, B], R]
        return f"[{', '.join(_annotation(item) for item in value)}]"
    origin, args = typing.get_origin(value), typing.get_args(value)
    if origin is typing.Union or (_UNION_TYPE is not None and origin is _UNION_TYPE):
        rendered = [_annotation(a) for a in args if a is not type(None)]
        if len(rendered) < len(args):
            rendered.append("None")
        return " | ".join(rendered)
    if _ANNOTATED is not None and origin is _ANNOTATED:
        metadata = ", ".join(_ADDRESS.sub("", repr(m)) for m in value.__metadata__)
        return f"Annotated[{_annotation(args[0])}, {metadata}]"
    if origin is getattr(typing, "Literal", None):
        return f"Literal[{', '.join(repr(a) for a in args)}]"
    if origin is not None:
        name = (
            getattr(value, "_name", None)
            or getattr(origin, "__qualname__", None)
            or repr(origin)
        )
        if not args:
            return name
        # Callable carries exactly two arguments: (parameters, result).
        if name == "Callable" and len(args) == 2:  # noqa: PLR2004
            params, result = args
            params_text = _annotation(params if params is Ellipsis else list(params))
            return f"Callable[{params_text}, {_annotation(result)}]"
        return f"{name}[{', '.join(_annotation(a) for a in args)}]"
    if isinstance(value, typing.TypeVar):
        return value.__name__
    if isinstance(value, type):
        module, name = value.__module__, value.__qualname__
        return name if module == "builtins" else f"{module}.{name}"
    text = _FORWARD_REF.sub(lambda m: repr(m.group(2)), repr(value))
    return _TYPING_PREFIX.sub("", text)


def _parameters(sig: inspect.Signature) -> str:
    """Render ``(a: T = d, /, b, *args, c, **kw) -> R`` from the parameters' facts."""
    parts: list[str] = []
    seen_keyword_only = seen_positional_only = False
    for param in sig.parameters.values():
        if param.kind is _P.POSITIONAL_ONLY:
            seen_positional_only = True
        elif seen_positional_only:
            parts.append("/")
            seen_positional_only = False
        if param.kind is _P.KEYWORD_ONLY and not seen_keyword_only:
            if not any(p.kind is _P.VAR_POSITIONAL for p in sig.parameters.values()):
                parts.append("*")
            seen_keyword_only = True
        prefix = {_P.VAR_POSITIONAL: "*", _P.VAR_KEYWORD: "**"}.get(param.kind, "")
        text = prefix + param.name
        if param.annotation is not _P.empty:
            text += f": {_annotation(param.annotation)}"
        if param.default is not _P.empty:
            text += (" = " if param.annotation is not _P.empty else "=") + _default(
                param.default
            )
        parts.append(text)
    if seen_positional_only:
        parts.append("/")
    text = f"({', '.join(parts)})"
    if sig.return_annotation is not _P.empty:
        text += f" -> {_annotation(sig.return_annotation)}"
    return text


def signature(name: str, obj: object, max_length: int | None = SIGNATURE_MAX) -> str:
    """
    Return the call signature of a public name, as plain text.

    An enum is called with one member value, ``Name(value)``, whatever its
    metaclass's signature looks like in this Python; a protocol is never
    called, so only its name is shown; a signature that cannot be read is
    ``Name(...)``. Parameter text longer than *max_length* keeps its first
    ``max_length - 4`` characters and ends ``" ...)"`` (one character more
    than *max_length*, as the generated skills have always shown it);
    ``None`` keeps it whole.
    """
    if not (inspect.isroutine(obj) or inspect.isclass(obj)):
        return name
    if _is_protocol(obj):
        return name
    if inspect.isclass(obj) and issubclass(obj, enum.Enum):
        return f"{name}(value)"
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return f"{name}(...)"
    text = " ".join(_parameters(sig).split())
    if max_length is not None and len(text) > max_length:
        text = text[: max_length - 4] + " ...)"
    return f"{name}{text}"


def describe(name: str, obj: object) -> dict[str, str]:
    """Return ``name``, ``kind``, ``signature`` (cut at ``SIGNATURE_MAX``) and ``summary``."""
    return {
        "name": name,
        "kind": kind(obj),
        "signature": signature(name, obj),
        "summary": summary(obj),
    }


# --------------------------------------------------------------------------- modules


def _import(module: str) -> types.ModuleType:
    if not isinstance(module, str) or not _MODULE_RE.match(module):
        raise ApiListingError(
            f"{module!r} is not scikitplot or a module below it"
            " (for example 'scikitplot.corpus').",
            "name",
        )
    try:
        return importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - reported with its cause
        raise ApiListingError(
            f"{module} does not import here: {type(exc).__name__}: {exc}",
            "import",
        ) from exc


def _is_unloaded_submodule(mod: types.ModuleType, module: str, name: str) -> bool:
    """
    Whether *name* is a submodule of the package *mod* that is not imported yet.

    Resolving such a name would import it (a lazy package imports on
    attribute access); a listing names it without importing it, so listing a
    package never imports its whole tree.
    """
    if not hasattr(mod, "__path__") or name in vars(mod):
        return False
    try:
        return importlib.util.find_spec(f"{module}.{name}") is not None
    except (ImportError, ValueError):
        return False


class _Unavailable:
    """A declared name that does not resolve here, with the reason."""

    def __init__(self, exc: BaseException) -> None:
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        self.reason = f"{type(exc).__name__}: {first}" if first else type(exc).__name__


def _declared(
    mod: types.ModuleType,
    module: str,
    *,
    lazy_submodules: bool = False,
    strict: bool = True,
) -> list[tuple[str, Any]]:
    """
    Return ``(name, object)`` for every public name in ``__all__``, in order.

    With *lazy_submodules*, a submodule not imported yet is returned as
    ``(name, None)`` instead of being imported. Without *strict*, a name that
    does not resolve is returned with an :class:`_Unavailable` instead of
    failing the whole listing (a partial install, or a build-generated name
    in an unbuilt checkout).
    """
    names = getattr(mod, "__all__", None)
    if names is None:
        raise ApiListingError(f"{module} declares no __all__", "api")
    found, failed = [], []
    for name in names:
        if name.startswith("_"):
            continue
        if lazy_submodules and _is_unloaded_submodule(mod, module, name):
            found.append((name, None))
            continue
        try:
            found.append((name, getattr(mod, name)))
        except Exception as exc:  # noqa: BLE001 - reported with its cause
            if not strict:
                found.append((name, _Unavailable(exc)))
                continue
            failed.append(f"{name} ({type(exc).__name__}: {exc})")
    if failed:
        raise ApiListingError(
            f"{module}: names in __all__ that do not resolve here: {failed}", "api"
        )
    return found


def _version() -> str:
    try:
        import scikitplot  # noqa: PLC0415

        return str(getattr(scikitplot, "__version__", "unknown"))
    except Exception:  # noqa: BLE001 - the listing does not depend on it
        return "unknown"


def module_listing(module: str = "scikitplot") -> dict[str, Any]:
    """
    Read one module's declared public API from the installed package.

    Parameters
    ----------
    module : str, default "scikitplot"
        ``scikitplot`` or a dotted module below it.

    Returns
    -------
    dict
        ``module``; ``version`` (of the installed scikit-plots); ``doc`` (the
        module docstring's first paragraph); ``rows`` (one dict per public name
        in ``__all__`` order: ``name``, ``kind``, ``signature``, ``summary``;
        a submodule is listed by its dotted name, kind ``module``, without
        being imported; a name that does not resolve here, kind
        ``unavailable``, with the reason as its summary); ``children`` (for a module whose
        ``__all__`` lists no other name: its public subpackages, each with
        ``module``, ``summary`` and ``sphinx``, whether it defines a Sphinx
        ``setup``).

    Raises
    ------
    ApiListingError
        When the name is not a scikit-plots module, the module does not
        import, or it has no ``__all__``.

    Examples
    --------
    >>> from scikitplot._api_listing import module_listing
    >>> listing = module_listing("scikitplot._api_listing")
    >>> [row["name"] for row in listing["rows"]][:2]
    ['SIGNATURE_MAX', 'ApiListingError']
    """
    mod = _import(module)
    rows = []
    for name, obj in _declared(mod, module, lazy_submodules=True, strict=False):
        if isinstance(obj, _Unavailable):
            rows.append(
                {
                    "name": name,
                    "kind": "unavailable",
                    "signature": name,
                    "summary": obj.reason,
                }
            )
        elif obj is None or inspect.ismodule(obj):
            dotted = f"{module}.{name}"
            rows.append(
                {"name": name, "kind": "module", "signature": dotted, "summary": ""}
            )
        else:
            rows.append(describe(name, obj))
    children = []
    if not any(r["kind"] not in ("module", "unavailable") for r in rows) and hasattr(
        mod, "__path__"
    ):
        for info in sorted(pkgutil.iter_modules(mod.__path__), key=lambda i: i.name):
            if (
                not info.ispkg
                or info.name.startswith("__")
                or info.name in ("tests", "test")
            ):
                continue
            child = importlib.import_module(f"{module}.{info.name}")
            children.append(
                {
                    "module": f"{module}.{info.name}",
                    "summary": summary(child),
                    "sphinx": callable(getattr(child, "setup", None)),
                }
            )
    return {
        "module": module,
        "version": _version(),
        "doc": first_paragraph(mod.__doc__ or ""),
        "rows": rows,
        "children": children,
    }


def name_detail(module: str, name: str) -> dict[str, str]:
    """
    Return one public name of *module* in full: its whole signature and docstring.

    Raises
    ------
    ApiListingError
        As :func:`module_listing`; ``reason="api"`` also when *name* is not in
        the module's ``__all__`` (the message names the closest ones) or does
        not resolve here. Other names are not resolved.
    """
    import difflib  # noqa: PLC0415

    mod = _import(module)
    names = getattr(mod, "__all__", None)
    if names is None:
        raise ApiListingError(f"{module} declares no __all__", "api")
    public = [n for n in names if not n.startswith("_")]
    if name not in public:
        close = difflib.get_close_matches(name, public, n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise ApiListingError(f"{module} has no public name {name!r}.{hint}", "api")
    # Only the requested name is resolved: another name that does not
    # resolve here (a partial install) does not hide this one.
    try:
        obj = getattr(mod, name)
    except Exception as exc:  # noqa: BLE001 - reported with its cause
        raise ApiListingError(
            f"{module}.{name} does not resolve here: {type(exc).__name__}: {exc}", "api"
        ) from exc
    doc = own_doc(obj) if (inspect.isroutine(obj) or inspect.isclass(obj)) else ""
    return {
        "module": module,
        "version": _version(),
        "name": name,
        "kind": kind(obj),
        "signature": signature(name, obj, max_length=None),
        "doc": doc or summary(obj),
    }


# --------------------------------------------------------------------------- rendering


def render_text(listing: dict[str, Any]) -> str:
    """Render a :func:`module_listing` as aligned plain text, one name a line."""
    lines = [f"{listing['module']} (scikit-plots {listing['version']})"]
    if listing["doc"]:
        lines.append(listing["doc"])
    lines.append("")
    width = max([len(r["kind"]) for r in listing["rows"]] + [len("module")])
    for row in listing["rows"]:
        lines.append(f"{row['kind']:<{width}} {row['signature']}")
        if row["summary"]:
            lines.append(f"{'':<{width}} {row['summary']}")
    for child in listing["children"]:
        lines.append(f"{'module':<{width}} {child['module']}")
        if child["summary"]:
            lines.append(f"{'':<{width}} {child['summary']}")
    if not listing["rows"] and not listing["children"]:
        lines.append("(no public names)")
    return "\n".join(lines) + "\n"


def render_markdown(listing: dict[str, Any]) -> str:
    """Render a :func:`module_listing` as a Markdown table (pipes escaped)."""

    def cell(text: str) -> str:
        return text.replace("|", "\\|")

    lines = [
        f"# `{listing['module']}` (scikit-plots {listing['version']})",
        "",
    ]
    if listing["doc"]:
        lines += [listing["doc"], ""]
    if listing["rows"]:
        lines += ["| Name | Kind | Summary |", "|---|---|---|"]
        lines += [
            f"| `{cell(r['signature'])}` | {r['kind']} | {cell(r['summary'])} |"
            for r in listing["rows"]
        ]
    for child in listing["children"]:
        lines.append(f"- `{child['module']}`: {cell(child['summary'])}")
    return "\n".join(lines) + "\n"
