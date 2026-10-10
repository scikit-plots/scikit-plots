r"""
User-supplied packs and formats, from one YAML or JSON file or many.

Notes
-----
**User notes.** Write a pack for your own domain and point cleanprompt at it::

    # hr_pack.yaml
    name: hr
    version: 1
    summary: Our HR system's identifiers.
    requires: [personal]
    fields:
      - names: [employee_number, badge_id]
        kind: EMPLOYEE
        role: id
    patterns:
      - kind: EMPLOYEE
        pattern: '\bEMP-\d{6}\b'
        intent: An employee number in our format.
        examples_yes: ['EMP-004121']
        examples_no: ['EMP-12']

.. code-block:: bash

    cleanprompt packs --pack-file hr_pack.yaml --check
    cleanprompt batch hr/ --out hr-safe/ --pack-file hr_pack.yaml --pack hr

A file may hold one pack, one format, or a bundle of several::

    packs:   [ {name: hr, ...}, {name: payroll, ...} ]
    formats: [ {name: tickets, extensions: [.tkt], splitter: keyvalue, ...} ]

JSON works exactly the same way and needs nothing installed; YAML needs
PyYAML, and says so if it is missing.

**Developer notes — a custom definition gets no special trust.**

It goes through the same validator as a built-in, so its patterns' examples are
executed before it is used (``I11``), unknown keys are refused, and a validator
can only be *named* from the fixed registry in ``_hooks.py``. YAML is read with
``safe_load``, which constructs only plain data. A file larger than
:data:`MAX_FILE_BYTES` is refused unread, because a definition that big is not
a definition.

Redefining a built-in is an error unless asked for. A custom ``personal`` pack
that silently replaced the built-in would change what every other pack
``requires`` — so ``conflict='replace'`` is spelled out, and an *identical*
redefinition is not a conflict at all.

See Also
--------
scikitplot.cleanprompt._catalog : The catalog these merge into.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ._catalog import Catalog, builtin_catalog, catalog_from_documents, require_yaml
from ._exceptions import CleanPromptError
from ._packs import PackError

__all__ = [
    "MAX_FILE_BYTES",
    "load_custom",
    "with_custom",
]

#: The largest definition file read. Built-in packs are a few kilobytes.
MAX_FILE_BYTES = 1024 * 1024


def _read(path: Path) -> Any:
    """Parse one definition file, by its suffix."""
    if not path.is_file():
        msg = f"definition file {str(path)!r} does not exist"
        raise CleanPromptError(msg)
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        msg = f"{path.name} is {size} bytes, above the {MAX_FILE_BYTES}-byte limit for a definition file"
        raise CleanPromptError(msg)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return json.loads(text)
        except ValueError as exc:
            raise PackError(path.name, [f"not valid JSON: {exc}"]) from exc
    if suffix in (".yaml", ".yml"):
        yaml = require_yaml()
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise PackError(path.name, [f"not valid YAML: {exc}"]) from exc
    msg = f"{path.name}: a definition file must end in .yaml, .yml or .json"
    raise CleanPromptError(msg)


def _split(document: Any, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Return ``(packs, formats)`` from one parsed file.

    Notes
    -----
    **Developer notes.** The shape decides, deterministically. Every pack and
    every format has a ``name`` and a bundle never does, so ``name`` is
    checked first: with it, a mapping holding ``splitter`` is a format and any
    other is a pack. Without it, a mapping holding ``packs`` or ``formats`` is
    a bundle. Testing for ``packs`` first would be wrong — a *format* has a
    ``packs`` key of its own — and was, until a test wrote a format file.
    """
    if not isinstance(document, Mapping):
        raise PackError(label, ["the file must contain a mapping"])
    if "name" not in document and ("packs" in document or "formats" in document):
        extra = sorted(set(document) - {"packs", "formats"})
        if extra:
            raise PackError(
                label,
                [
                    f"a bundle may only hold 'packs' and 'formats', not {', '.join(extra)}"
                ],
            )
        packs = document.get("packs") or []
        formats = document.get("formats") or []
        if not isinstance(packs, list) or not isinstance(formats, list):
            raise PackError(label, ["'packs' and 'formats' must each be a list"])
        return (
            {f"{label}#packs[{i}]": item for i, item in enumerate(packs)},
            {f"{label}#formats[{i}]": item for i, item in enumerate(formats)},
        )
    if "splitter" in document:
        return {}, {label: document}
    return {label: document}, {}


def load_custom(paths: str | Path | Iterable[str | Path]) -> Catalog:
    """
    Load user definitions from one or more files into a catalog of their own.

    Parameters
    ----------
    paths : path or iterable of paths
        ``.yaml``, ``.yml`` or ``.json`` files.

    Returns
    -------
    Catalog
        A partial catalog of only the custom definitions, every one validated.
        Its ``requires`` may name built-in packs; those are checked when it is
        merged by :func:`with_custom`.

    Raises
    ------
    PackError
        If any definition is invalid, listing every problem in it.
    CleanPromptError
        If a file is missing, too large, of an unknown type, or two files
        define the same name.
    """
    items = [paths] if isinstance(paths, (str, Path)) else list(paths)
    packs: dict[str, Any] = {}
    formats: dict[str, Any] = {}
    for item in items:
        path = Path(item)
        found_packs, found_formats = _split(_read(path), path.name)
        packs.update(found_packs)
        formats.update(found_formats)
    return catalog_from_documents(_label(packs), _label(formats), partial=True)


def _label(documents: Mapping[str, Any]) -> dict[str, Any]:
    """
    Give every document a label ending in a recognised suffix.

    Notes
    -----
    **Developer notes.** The label becomes the spec's ``source``, which is what
    an error message and a conflict report name. A bundle member is labelled
    ``file.yaml#packs[2]`` so the message points at the entry, not just the
    file.
    """
    out = {}
    for label, document in documents.items():
        out[
            label if label.endswith((".yaml", ".yml", ".json")) else f"{label}.json"
        ] = document
    return out


def with_custom(
    paths: str | Path | Iterable[str | Path] | None,
    conflict: str = "error",
    base: Catalog | None = None,
) -> Catalog:
    """
    Return the built-in catalog with custom definitions merged in.

    Parameters
    ----------
    paths : path, iterable of paths, or None
        Definition files. ``None`` or empty returns ``base`` unchanged.
    conflict : {'error', 'replace'}, default='error'
        What to do when a custom definition differs from a built-in of the same
        name.
    base : Catalog, optional
        Defaults to :func:`~scikitplot.cleanprompt._catalog.builtin_catalog`.

    Returns
    -------
    Catalog
        Validated as a whole, including every ``requires``.

    Examples
    --------
    >>> import json, tempfile, pathlib
    >>> folder = pathlib.Path(tempfile.mkdtemp())
    >>> _ = (folder / "hr.json").write_text(
    ...     json.dumps(
    ...         {
    ...             "name": "hr",
    ...             "version": 1,
    ...             "summary": "HR ids.",
    ...             "requires": ["personal"],
    ...             "fields": [
    ...                 {"names": ["badge_id"], "kind": "EMPLOYEE", "role": "id"}
    ...             ],
    ...         }
    ...     )
    ... )
    >>> [p.name for p in with_custom(folder / "hr.json").resolve_packs("hr")]
    ['personal', 'hr']
    """
    catalog = base if base is not None else builtin_catalog()
    if not paths:
        return catalog
    return catalog.merge(load_custom(paths), conflict=conflict)
