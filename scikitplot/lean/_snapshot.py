# scikitplot/lean/_snapshot.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Save a :class:`~scikitplot.lean.LeanIndex` to one file and load it back.

A snapshot is the index's state, not its caches: declarations, files,
references, contexts and findings. Loading one takes seconds where reading
the project again takes minutes, and gives the same reports.

Format (``lean.index.snapshot/1``)::

    b"LEANIDX\\0"  8 bytes
    header size   8 bytes, unsigned, little-endian
    header        JSON, UTF-8, padded to a multiple of 8 bytes
    sections      each starting at a multiple of 8 bytes

The header names each section (offset from the first, length, type) and
holds the SHA-256 of all of them. Integer sections are little-endian
64-bit (``q``) or 32-bit (``i``); text sections are UTF-8 strings joined in
one buffer with a ``q`` section of offsets.

Notes
-----
**User.** Nothing in a snapshot runs: no pickle, only numbers and text. A
damaged or altered snapshot raises :class:`ValueError` when loaded: a wrong
checksum, a number out of range or offsets out of order. A snapshot names
files by label, never by absolute path; reports quote source text only
when ``load`` is given a ``root``, and then only from files inside it.

**Developer.** The same files give the same bytes, whatever order they
were added in: the index is renumbered first, and references and contexts
are numbered by first use in that order. Sections are aligned to 8 bytes
so that a later version can map them into memory without copying.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from array import array
from pathlib import Path
from typing import Any, Iterable

__all__: list[str] = []

#: What a snapshot does with each attribute of a LeanIndex. A new attribute
#: must be placed here (tests check it): written, rebuilt from what is
#: written, chosen again when loading, or a cache that starts empty.
SAVED = frozenset(
    {
        "_name", "_kind", "_file", "_start", "_end", "_ref_ptr", "_ref_data",
        "_ctx_of", "_flags", "_attributes", "_shape_of", "_private", "_claims",
        "_vacuous", "_ref_names", "_ctx_keys", "_labels", "_imports",
        "_file_lines", "_problems", "_kinds", "_learned", "_skipped",
        "_layers", "_layer_of",
        "_dialect", "_commands", "_max_depth", "_max_bytes", "_suffix",
        "_visibility", "_source_root",
    }
)  # fmt: skip
REBUILT = frozenset(
    {
        "_namespace", "_opens", "_by_name", "_ref_ids", "_ctx_ids",
        "_label_ids", "_first_node", "_paths", "_intern", "_generation",
        "_in_order",
    }
)  # fmt: skip
CHOSEN_WHEN_LOADING = frozenset({"_graph", "_workers", "_flat"})

#: The format this module writes and reads.
FORMAT = "lean.index.snapshot/1"
_MAGIC = b"LEANIDX\x00"
_ALIGN = 8
#: Bytes before the header: the magic string and the header's size.
_FRAME = 16
#: Bytes of one statement-shape digest.
_DIGEST = 16


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _ints(values: Iterable[int], code: str) -> bytes:
    """Return *values* as little-endian integers of type *code* (``q`` or ``i``)."""
    packed = array(code, values)
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def _texts(values: Iterable[str]) -> tuple[bytes, bytes]:
    """Return *values* as one UTF-8 buffer and its ``q`` offsets (with the end)."""
    buffer = bytearray()
    offsets = [0]
    for value in values:
        buffer += value.encode("utf-8")
        offsets.append(len(buffer))
    return bytes(buffer), _ints(offsets, "q")


def dumps(index: Any) -> bytes:
    """Return the snapshot of *index* as bytes; see the module docstring."""
    index._canonical()
    n = len(index._name)
    # References and contexts numbered by first use, in declaration order:
    # the same files give the same numbers, whatever order they were added.
    ref_new: dict[int, int] = {}
    data = [ref_new.setdefault(old, len(ref_new)) for old in index._ref_data]
    ref_names = [""] * len(ref_new)
    for old, new in ref_new.items():
        ref_names[new] = index._ref_names[old]
    ctx_new: dict[int, int] = {}
    ctx_of = [ctx_new.setdefault(old, len(ctx_new)) for old in index._ctx_of]
    contexts = [""] * len(ctx_new)
    for old, new in ctx_new.items():
        namespace, opens = index._ctx_keys[old]
        contexts[new] = "\x00".join((namespace, *opens))
    kinds = sorted(set(index._kind))
    kind_id = {kind: i for i, kind in enumerate(kinds)}
    shaped = sorted(index._shape_of)
    meta = {
        "options": {
            "dialect": index._dialect,
            "commands": sorted(index._commands),
            "max_scope_depth": index._max_depth,
            "max_file_bytes": index._max_bytes,
            "suffix": index._suffix,
            "visibility": index._visibility,
            "source_root": index._source_root,
        },
        "labels": list(index._labels),
        "layers": [list(layer) for layer in index._layers],
        "layer_of": list(index._layer_of),
        "imports": [list(found) for found in index._imports],
        "file_lines": list(index._file_lines),
        "problems": [
            [file_id, [list(problem) for problem in index._problems[file_id]]]
            for file_id in sorted(index._problems)
        ],
        "kinds_counted": sorted(index._kinds.items()),
        "learned": sorted([k, list(v)] for k, v in index._learned.items()),
        "skipped": sorted(list(item) for item in index._skipped),
        "flags": [[n, list(index._flags[n])] for n in sorted(index._flags)],
        "attributes": [
            [n, list(index._attributes[n])] for n in sorted(index._attributes)
        ],
        "vacuous": [[n, index._vacuous[n]] for n in sorted(index._vacuous)],
        "private": sorted(index._private),
        "claims": sorted(index._claims),
    }
    names_text, names_at = _texts(name or "" for name in index._name)
    refs_text, refs_at = _texts(ref_names)
    ctx_text, ctx_at = _texts(contexts)
    kinds_text, kinds_at = _texts(kinds)
    sections = {
        "meta": (
            json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            "json",
        ),
        "name_text": (names_text, "utf8"),
        "name_at": (names_at, "q"),
        "named": (bytes(name is not None for name in index._name), "bool"),
        "kind_text": (kinds_text, "utf8"),
        "kind_at": (kinds_at, "q"),
        "kind_of": (_ints((kind_id[k] for k in index._kind), "i"), "i"),
        "file_of": (_ints(index._file, "q"), "q"),
        "start": (_ints(index._start, "q"), "q"),
        "end": (_ints(index._end, "q"), "q"),
        "ref_text": (refs_text, "utf8"),
        "ref_at": (refs_at, "q"),
        "ref_ptr": (_ints(index._ref_ptr, "q"), "q"),
        "ref_data": (_ints(data, "i"), "i"),
        "ctx_text": (ctx_text, "utf8"),
        "ctx_at": (ctx_at, "q"),
        "ctx_of": (_ints(ctx_of, "i"), "i"),
        "shape_of": (_ints(shaped, "i"), "i"),
        "shapes": (b"".join(index._shape_of[n] for n in shaped), "bytes"),
    }
    body = bytearray()
    table = {}
    digest = hashlib.sha256()
    for name, (blob, kind) in sections.items():
        body += b"\x00" * (-len(body) % _ALIGN)
        table[name] = {"offset": len(body), "length": len(blob), "type": kind}
        body += blob
    digest.update(body)
    header = {
        "format": FORMAT,
        "declarations": n,
        "files": len(index._labels),
        "sections": table,
        "sha256": digest.hexdigest(),
    }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % _ALIGN)
    return _MAGIC + len(encoded).to_bytes(8, "little") + encoded + bytes(body)


def save(index: Any, path: str | Path) -> Path:
    """Write *index*'s snapshot to *path* atomically; return the path."""
    from ._discovery import _atomic_write  # noqa: PLC0415

    target = Path(path)
    if target.exists() and target.is_dir():
        raise ValueError(f"LeanIndex.save: {target} is a folder.")
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(target, dumps(index))
    return target


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _fail(message: str) -> ValueError:
    return ValueError(f"LeanIndex.load: {message}")


def _read_ints(blob: bytes, code: str) -> array:
    size = array(code).itemsize
    if len(blob) % size:
        raise _fail(f"a {code} section is not a whole number of items.")
    values = array(code)
    values.frombytes(blob)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _read_texts(text: bytes, at: array, count: int | None, what: str) -> list[str]:
    if (count is not None and len(at) != count + 1) or not at or at[0] != 0:
        raise _fail(f"the {what} offsets do not match its count.")
    if at[-1] != len(text) or any(a > b for a, b in zip(at, at[1:])):
        raise _fail(f"the {what} offsets are out of order or out of range.")
    try:
        return [text[at[i] : at[i + 1]].decode("utf-8") for i in range(len(at) - 1)]
    except UnicodeDecodeError as exc:
        raise _fail(f"the {what} text is not UTF-8.") from exc


def _in_range(values: Iterable[int], size: int, what: str) -> None:
    for value in values:
        if not 0 <= value < size:
            raise _fail(f"a {what} number is out of range.")


def _sections(blob: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Check *blob*'s frame and checksum; return the header and each section."""
    if blob[:8] != _MAGIC:
        raise _fail("not a LeanIndex snapshot.")
    if len(blob) < _FRAME:
        raise _fail("the snapshot is truncated.")
    size = int.from_bytes(blob[8:_FRAME], "little")
    if _FRAME + size > len(blob):
        raise _fail("the snapshot is truncated.")
    try:
        header = json.loads(blob[_FRAME : _FRAME + size].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise _fail("the header is not valid JSON.") from exc
    if not isinstance(header, dict) or header.get("format") != FORMAT:
        found = header.get("format") if isinstance(header, dict) else None
        raise _fail(f"format {found!r} is not {FORMAT!r}.")
    body = blob[_FRAME + size :]
    if hashlib.sha256(body).hexdigest() != header.get("sha256"):
        raise _fail("the checksum does not match: the file is damaged or altered.")
    parts = {}
    for name, place in header.get("sections", {}).items():
        start, length = place.get("offset"), place.get("length")
        if (
            not (isinstance(start, int) and isinstance(length, int))
            or start < 0
            or length < 0
        ):
            raise _fail(f"section {name!r} has no valid place.")
        if start + length > len(body):
            raise _fail(f"section {name!r} runs past the end.")
        parts[name] = body[start : start + length]
    return header, parts


def loads(
    blob: bytes, index_class: Any, *, root: str | Path | None, **options: Any
) -> Any:
    """Return the index *blob* holds; see :func:`load`."""
    header, parts = _sections(blob)
    try:
        return _build(header, parts, index_class, root, options)
    except (TypeError, KeyError, IndexError, AttributeError) as exc:
        raise _fail(
            f"the snapshot's content is not valid ({type(exc).__name__})."
        ) from exc


#: The sections a snapshot must have.
_SECTIONS = frozenset(
    {
        "meta", "name_text", "name_at", "named", "kind_text", "kind_at", "kind_of",
        "file_of", "start", "end", "ref_text", "ref_at", "ref_ptr", "ref_data",
        "ctx_text", "ctx_at", "ctx_of", "shape_of", "shapes",
    }
)  # fmt: skip


def _build(
    header: dict[str, Any],
    parts: dict[str, bytes],
    index_class: Any,
    root: str | Path | None,
    options: dict[str, Any],
) -> Any:
    """Decode a snapshot's sections, check every one, and build its index."""
    if _SECTIONS - set(parts):
        raise _fail(f"sections missing: {', '.join(sorted(_SECTIONS - set(parts)))}.")
    try:
        meta = json.loads(parts["meta"].decode("utf-8"))
        saved = dict(meta["options"])
        n, files = int(header["declarations"]), int(header["files"])
    except (UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
        raise _fail("the metadata is not valid.") from exc
    cols = _columns(parts, n)
    _check(cols, meta, n, files)
    base = _roots(root)
    index = index_class(
        dialect=saved["dialect"],
        commands=saved["commands"],
        max_scope_depth=saved["max_scope_depth"],
        max_file_bytes=saved["max_file_bytes"],
        suffix=saved["suffix"],
        visibility=saved["visibility"],
        source_root=saved["source_root"],
        **options,
    )
    _fill_files(index, cols, meta, n, base)
    _fill_declarations(index, cols, meta, n)
    index._generation += 1
    return index


def _columns(parts: dict[str, bytes], n: int) -> dict[str, Any]:
    """Decode every section into arrays and lists (offsets checked as text is cut)."""

    def texts(name: str, count: int | None, what: str) -> list[str]:
        return _read_texts(
            parts[f"{name}_text"], _read_ints(parts[f"{name}_at"], "q"), count, what
        )

    return {
        "names": texts("name", n, "name"),
        "named": parts["named"],
        "kinds": texts("kind", None, "kind"),
        "kind_of": _read_ints(parts["kind_of"], "i"),
        "file_of": _read_ints(parts["file_of"], "q"),
        "start": _read_ints(parts["start"], "q"),
        "end": _read_ints(parts["end"], "q"),
        "ref_names": texts("ref", None, "reference"),
        "ref_ptr": _read_ints(parts["ref_ptr"], "q"),
        "ref_data": _read_ints(parts["ref_data"], "i"),
        "contexts": texts("ctx", None, "context"),
        "ctx_of": _read_ints(parts["ctx_of"], "i"),
        "shape_of": _read_ints(parts["shape_of"], "i"),
        "shapes": parts["shapes"],
    }


def _check(cols: dict[str, Any], meta: dict[str, Any], n: int, files: int) -> None:
    """Check every count and every number before the index uses any of them."""
    per_node = ("named", "kind_of", "file_of", "start", "end", "ctx_of")
    if any(len(cols[name]) != n for name in per_node):
        raise _fail("the declaration sections do not have one entry each.")
    labels = meta.get("labels", [])
    per_file = (labels, meta.get("imports", []), meta.get("file_lines", []))
    if any(len(values) != files for values in per_file):
        raise _fail("the file sections do not have one entry each.")
    if len(set(labels)) != files:
        raise _fail("two files have the same label.")
    layers = meta.get("layers", [["", None]])
    names = [layer[0] for layer in layers]
    if not layers or names[0] != "" or len(set(names)) != len(names):
        raise _fail("the layers are not valid.")
    layer_of = meta.get("layer_of", [0] * files)
    if len(layer_of) != files:
        raise _fail("the file sections do not have one entry each.")
    _in_range(layer_of, len(layers), "layer")
    for label, layer in zip(labels, layer_of):
        if layer and not label.startswith(f"{names[layer]}:"):
            raise _fail("a file's label does not name its layer.")
    ptr, data = cols["ref_ptr"], cols["ref_data"]
    if len(ptr) != n + 1 or ptr[0] != 0 or ptr[-1] != len(data):
        raise _fail("the reference offsets do not match the declarations.")
    if any(a > b for a, b in zip(ptr, ptr[1:])):
        raise _fail("the reference offsets are out of order.")
    _in_range(data, len(cols["ref_names"]), "reference")
    _in_range(cols["ctx_of"], len(cols["contexts"]), "context")
    _in_range(cols["kind_of"], len(cols["kinds"]), "kind")
    _in_range(cols["file_of"], files, "file")
    if any(a > b for a, b in zip(cols["file_of"], cols["file_of"][1:])):
        raise _fail("declarations are not grouped by file.")
    if len(cols["shapes"]) != _DIGEST * len(cols["shape_of"]):
        raise _fail("the shape digests do not match their declarations.")
    _in_range(cols["shape_of"], n, "declaration")
    for key in ("flags", "attributes", "vacuous"):
        _in_range((pair[0] for pair in meta.get(key, [])), n, "declaration")
    _in_range(meta.get("private", []), n, "declaration")
    _in_range(meta.get("claims", []), n, "declaration")
    _in_range((pair[0] for pair in meta.get("problems", [])), files, "file")


def _fill_files(
    index: Any, cols: dict[str, Any], meta: dict[str, Any], n: int, base: Path | None
) -> None:
    """Add the files: labels, paths (inside *base* only), imports, problems."""
    layers = [(str(name), root) for name, root in meta.get("layers", [["", None]])]
    index._layers = layers
    index._layer_of = [int(x) for x in meta.get("layer_of", [0] * len(meta["labels"]))]
    for file_id, label in enumerate(meta["labels"]):
        index._label_ids[label] = file_id
        index._labels.append(label)
        name = layers[index._layer_of[file_id]][0]
        within = label[len(name) + 1 :] if name else label
        index._paths.append(_inside(base.get(name) if base else None, within))
        index._imports.append(tuple(meta["imports"][file_id]))
        index._file_lines.append(int(meta["file_lines"][file_id]))
    firsts = [n] * len(index._labels)
    for node in range(n - 1, -1, -1):
        firsts[cols["file_of"][node]] = node
    for file_id in range(len(firsts) - 2, -1, -1):  # a file without declarations
        firsts[file_id] = min(firsts[file_id], firsts[file_id + 1])
    index._first_node = firsts
    index._problems = {
        f: tuple((int(a), str(b)) for a, b in p) for f, p in meta["problems"]
    }
    index._learned = {k: (str(v[0]), int(v[1])) for k, v in meta["learned"]}
    index._skipped = [(str(a), str(b)) for a, b in meta["skipped"]]
    index._kinds.update(dict(meta["kinds_counted"]))
    if index._layers[0][1] is None and index._source_root is not None:
        index._layers[0] = ("", index._source_root)


def _fill_declarations(
    index: Any, cols: dict[str, Any], meta: dict[str, Any], n: int
) -> None:
    """Add the declarations, their references and contexts, and their findings."""
    index._ref_names = cols["ref_names"]
    index._ref_ids = {name: i for i, name in enumerate(cols["ref_names"])}
    for text in cols["contexts"]:
        namespace, *opens = text.split("\x00")
        index._context_id(sys.intern(namespace), index._interned(tuple(opens)))
    if len(index._ctx_keys) != len(cols["contexts"]):
        raise _fail("a context is listed twice.")
    index._ref_ptr, index._ref_data = cols["ref_ptr"], cols["ref_data"]
    index._ctx_of = cols["ctx_of"]
    index._file = array("l", cols["file_of"])
    index._start, index._end = array("l", cols["start"]), array("l", cols["end"])
    kinds = [sys.intern(kind) for kind in cols["kinds"]]
    names, named = cols["names"], cols["named"]
    for node in range(n):
        name = names[node] if named[node] else None
        index._name.append(name)
        index._kind.append(kinds[cols["kind_of"][node]])
        namespace, opens = index._ctx_keys[cols["ctx_of"][node]]
        index._namespace.append(namespace)
        index._opens.append(opens)
        if name is not None:
            held = index._by_name.get(name)
            if held is None:
                index._by_name[name] = node
            elif isinstance(held, int):
                index._by_name[name] = [held, node]
            else:
                held.append(node)
    index._flags = {node: tuple(v) for node, v in meta["flags"]}
    index._attributes = {node: tuple(v) for node, v in meta["attributes"]}
    index._vacuous = dict(meta["vacuous"])
    index._private = set(meta["private"])
    index._claims = set(meta["claims"])
    shapes = cols["shapes"]
    index._shape_of = {
        node: shapes[_DIGEST * i : _DIGEST * (i + 1)]
        for i, node in enumerate(cols["shape_of"])
    }


def _roots(root: Any) -> dict[str, Path] | None:
    """Return the folder of each layer name (``""``: the first), checked, or ``None``."""
    if root is None:
        return None
    pairs = root.items() if isinstance(root, dict) else [("", root)]
    folders = {}
    for name, folder in pairs:
        path = Path(folder).resolve()
        if not path.is_dir():
            raise _fail(f"root {folder!s} is not a folder.")
        folders[str(name)] = path
    return folders


def _inside(base: Path | None, label: str) -> Path | None:
    """Return the file *label* names under *base*, or ``None`` if it is not inside it."""
    if base is None:
        return None
    path = (base / label).resolve()
    try:
        inside = os.path.commonpath([str(base), str(path)]) == str(base)
    except ValueError:  # another drive (Windows)
        return None
    return path if inside and path.is_file() else None


def load(
    path: str | Path,
    index_class: Any,
    *,
    root: str | Path | None = None,
    **options: Any,
) -> Any:
    """Read a snapshot written by :func:`save`; see :meth:`LeanIndex.load`."""
    return loads(Path(path).read_bytes(), index_class, root=root, **options)
