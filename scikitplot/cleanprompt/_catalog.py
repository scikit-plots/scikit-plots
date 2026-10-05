"""
The catalog: every pack and format, where they come from, and how to select.

Notes
-----
**User notes.** Three ways to ask for packs, all through the same resolver::

    catalog.resolve_packs("all")  # every pack
    catalog.resolve_packs("addressbook")  # one, plus what it requires
    catalog.resolve_packs(["pandas", "patient"])  # any combination
    catalog.resolve_packs("auto", formats=[csv])  # what these formats suggest

``requires`` is followed transitively, so ``addressbook`` brings ``personal``
with it. An unknown name is an error that suggests the nearest real one.

**Developer notes — built-ins are YAML, shipped as JSON (``D-13.4``).**

``_config/packs/*.yaml`` and ``_config/formats/*.yaml`` are the source. The base
tier cannot import PyYAML, so :func:`compile_config` turns them into
``_config/_compiled.json``, which :func:`builtin_catalog` reads with the
standard library. :func:`check_compiled` re-compiles and reports any
difference; a test calls it, so the two cannot drift silently (invariant
``I10``). Every document is validated on *both* sides — when compiled, and
again when the JSON is loaded — so a hand-edited JSON file is held to the same
rules as the YAML it came from.

**Developer notes — one resolver, deterministic output.**

Selection returns packs in a fixed order: dependencies before the packs that
need them, ties broken by name. The order decides which detector is built
first and therefore which name a duplicate pattern keeps, so it must not depend
on the order a user happened to list packs in (invariant ``I12``).

See Also
--------
scikitplot.cleanprompt._packs : The pack schema.
scikitplot.cleanprompt._formats : The format schema.
scikitplot.cleanprompt._custom : User-supplied packs and formats.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from ._exceptions import CapabilityError, CleanPromptError
from ._formats import FormatSpec, format_from_document
from ._packs import CodeSpec, FieldSpec, PackError, PackSpec, pack_from_document

__all__ = [
    "AT_REST_PACKS",
    "COMPILED_PATH",
    "CONFIG_DIR",
    "Catalog",
    "at_rest_findings",
    "builtin_catalog",
    "check_compiled",
    "compile_config",
    "write_compiled",
]

#: Where the built-in YAML and its compiled form live.
CONFIG_DIR = Path(__file__).resolve().parent / "_config"

#: The compiled catalog the base tier reads.
COMPILED_PATH = CONFIG_DIR / "_compiled.json"

#: Version of the compiled file's layout. Bumped only when the layout changes.
COMPILED_SCHEMA = 1

#: Packs whose patterns describe credentials. No built-in definition file may
#: contain text one of these patterns accepts (invariant ``I14``).
AT_REST_PACKS = ("secrets",)

#: Selection keywords with a meaning of their own.
ALL = "all"
AUTO = "auto"
NONE = "none"


def canonical(document: Any) -> str:
    """
    Return the canonical JSON text of a document.

    Parameters
    ----------
    document : object
        Anything :func:`json.dumps` accepts.

    Returns
    -------
    str
        Sorted keys, no insignificant whitespace, ASCII escapes.

    Notes
    -----
    **Developer notes.** This is the form fingerprints are computed over and
    the form compiled-versus-source is compared in, so it has one spelling:
    ``hash()`` would differ between processes and pretty-printing would make
    whitespace significant.

    Examples
    --------
    >>> canonical({"b": 1, "a": [2]})
    '{"a":[2],"b":1}'
    """
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


@dataclass(frozen=True)
class Catalog:
    """
    An immutable set of packs and formats.

    Parameters
    ----------
    packs : mapping
        Name to :class:`~scikitplot.cleanprompt._packs.PackSpec`.
    formats : mapping
        Name to :class:`~scikitplot.cleanprompt._formats.FormatSpec`.
    documents : mapping
        ``"pack:NAME"`` or ``"format:NAME"`` to the canonical JSON the spec was
        built from. Carried so a plan can be fingerprinted by content.
    partial : bool, default=False
        A fragment meant only to be merged — a set of custom definitions whose
        ``requires`` name packs that live in the catalog it will join. Cross
        references are checked when it is merged, not before.

    Raises
    ------
    CleanPromptError
        If two formats claim one extension, or a ``requires`` names a pack that
        is not in the catalog, or the requirements form a cycle.
    """

    packs: Mapping[str, PackSpec] = field(default_factory=dict)
    formats: Mapping[str, FormatSpec] = field(default_factory=dict)
    documents: Mapping[str, str] = field(default_factory=dict)
    partial: bool = field(default=False, compare=False)

    def __post_init__(self) -> None:
        owners: dict[str, str] = {}
        problems = []
        for spec in sorted(self.formats.values(), key=lambda one: one.name):
            for extension in spec.extensions:
                if extension in owners:
                    problems.append(
                        f"extension {extension!r} is claimed by both "
                        f"{owners[extension]!r} and {spec.name!r}"
                    )
                owners[extension] = spec.name
        if self.partial:
            if problems:
                msg = "the definitions are inconsistent:\n" + "\n".join(
                    f"  - {p}" for p in problems
                )
                raise CleanPromptError(msg)
            return
        for spec in self.packs.values():
            problems.extend(
                f"pack {spec.name!r} requires unknown pack {other!r}"
                for other in spec.requires
                if other not in self.packs
            )
        for spec in self.formats.values():
            problems.extend(
                f"format {spec.name!r} suggests unknown pack {other!r}"
                for other in spec.packs
                if other not in self.packs
            )
        if problems:
            msg = "the catalog is inconsistent:\n" + "\n".join(
                f"  - {p}" for p in problems
            )
            raise CleanPromptError(msg)
        self._order()

    # -- selection --------------------------------------------------------

    def _order(self) -> list[str]:
        """Return every pack name, dependencies first, ties by name."""
        order: list[str] = []
        state: dict[str, int] = {}

        def visit(name: str, trail: tuple[str, ...]) -> None:
            if state.get(name) == 2:  # ruff: ignore[magic-value-comparison]
                return
            if state.get(name) == 1:
                cycle = " -> ".join((*trail, name))
                msg = f"pack requirements form a cycle: {cycle}"
                raise CleanPromptError(msg)
            state[name] = 1
            for other in sorted(self.packs[name].requires):
                visit(other, (*trail, name))
            state[name] = 2
            order.append(name)

        for name in sorted(self.packs):
            visit(name, ())
        return order

    def _unknown(
        self,
        what: str,
        name: str,
        known: Iterable[str],
        listed: Iterable[str] | None = None,
    ) -> CleanPromptError:
        """
        Build the error for an unknown name, suggesting the nearest ones.

        Notes
        -----
        **Developer notes.** ``known`` is what suggestions are drawn from and
        ``listed`` what the message enumerates. For formats they differ: a
        near-miss extension is worth suggesting, but listing all seventy
        extensions buries the dozen format names a reader is looking for.
        """
        candidates = sorted(known)
        close = difflib.get_close_matches(name, candidates, n=3, cutoff=0.6)
        hint = f"; did you mean {', '.join(close)}?" if close else ""
        shown = sorted(listed) if listed is not None else candidates
        return CleanPromptError(
            f"unknown {what} {name!r}{hint} Known {what}s: {', '.join(shown)}"
        )

    def resolve_packs(
        self,
        selection: str | Iterable[str] | None = ALL,
        formats: Iterable[FormatSpec] | None = None,
    ) -> tuple[PackSpec, ...]:
        """
        Return the packs a selection means, with their requirements.

        Parameters
        ----------
        selection : str or iterable of str or None, default='all'
            ``'all'``, ``'none'``, ``'auto'``, one name, or several. ``None``
            means ``'none'``.
        formats : iterable of FormatSpec, optional
            Consulted only by ``'auto'``: the union of their suggested packs.

        Returns
        -------
        tuple of PackSpec
            Dependencies first, ties broken by name.

        Raises
        ------
        CleanPromptError
            If a name is unknown.

        Examples
        --------
        >>> [p.name for p in builtin_catalog().resolve_packs("addressbook")]
        ['personal', 'addressbook']
        """
        if selection is None or selection == NONE:
            return ()
        if selection == ALL:
            wanted = set(self.packs)
        elif selection == AUTO:
            wanted = {name for spec in formats or () for name in spec.packs}
        else:
            names = [selection] if isinstance(selection, str) else list(selection)
            wanted = set()
            for name in names:
                if name in (ALL, AUTO, NONE):
                    msg = f"{name!r} is a selection on its own and cannot be combined with pack names"
                    raise CleanPromptError(msg)
                if name not in self.packs:
                    raise self._unknown("pack", name, self.packs)
                wanted.add(name)
        closure = set()
        stack = list(wanted)
        while stack:
            name = stack.pop()
            if name in closure:
                continue
            closure.add(name)
            stack.extend(self.packs[name].requires)
        return tuple(self.packs[name] for name in self._order() if name in closure)

    def resolve_formats(
        self, selection: str | Iterable[str] | None = ALL
    ) -> tuple[FormatSpec, ...]:
        """
        Return the formats a selection means, sorted by name.

        Parameters
        ----------
        selection : str or iterable of str or None, default='all'
            ``'all'``, one name, or several; ``None`` or ``'none'`` for none.
            An extension such as ``'.sh'`` is accepted in place of a name.

        Returns
        -------
        tuple of FormatSpec
            Sorted by name.

        Raises
        ------
        CleanPromptError
            If a name or extension is unknown.
        """
        if selection is None or selection == NONE:
            return ()
        if selection == ALL:
            return tuple(self.formats[name] for name in sorted(self.formats))
        names = [selection] if isinstance(selection, str) else list(selection)
        by_extension = {
            ext: spec.name for spec in self.formats.values() for ext in spec.extensions
        }
        chosen = set()
        for name in names:
            key = name.lower() if name.startswith(".") else name
            if key in self.formats:
                chosen.add(key)
            elif key in by_extension:
                chosen.add(by_extension[key])
            elif f".{key}" in by_extension:
                chosen.add(by_extension[f".{key}"])
            else:
                raise self._unknown(
                    "format",
                    name,
                    list(self.formats) + sorted(by_extension),
                    listed=self.formats,
                )
        return tuple(self.formats[name] for name in sorted(chosen))

    def format_for(
        self, path: str | Path, allowed: Iterable[FormatSpec] | None = None
    ) -> FormatSpec | None:
        """
        Return the format that reads a file, or ``None``.

        Parameters
        ----------
        path : str or path-like
            The file name. Only the name is used; nothing is opened.
        allowed : iterable of FormatSpec, optional
            Restrict the answer to these formats.

        Returns
        -------
        FormatSpec or None
            ``None`` when no allowed format claims the extension.

        Notes
        -----
        **Developer notes.** A dotfile such as ``.env`` has no suffix to
        :mod:`pathlib`, so its whole name is tried as the extension. That is
        the only special case, and it is the one that matters: ``.env`` is
        where credentials live.

        Examples
        --------
        >>> builtin_catalog().format_for("deploy/.env").name
        'env'
        >>> builtin_catalog().format_for("notes.MD").name
        'markdown'
        """
        name = Path(str(path)).name.lower()
        suffix = Path(name).suffix
        candidates = [suffix] if suffix else []
        if name.startswith("."):
            candidates.append(name)
        pool = list(allowed) if allowed is not None else list(self.formats.values())
        for candidate in candidates:
            for spec in pool:
                if candidate in spec.extensions:
                    return spec
        return None

    # -- derived views ----------------------------------------------------

    @staticmethod
    def field_index(packs: Iterable[PackSpec]) -> dict[str, FieldSpec]:
        """
        Return normalised field name to rule, across packs.

        Parameters
        ----------
        packs : iterable of PackSpec
            Resolved packs.

        Returns
        -------
        dict
            One entry per field name.

        Raises
        ------
        CleanPromptError
            If two packs give one field name different kinds. Silently keeping
            either would decide what a value is called by the order packs
            were listed in.
        """
        index: dict[str, FieldSpec] = {}
        owner: dict[str, str] = {}
        problems = []
        for pack in packs:
            for rule in pack.fields:
                for name in rule.names:
                    existing = index.get(name)
                    if existing is not None and existing.kind != rule.kind:
                        problems.append(
                            f"field {name!r} is {existing.kind} in pack {owner[name]!r} "
                            f"and {rule.kind} in pack {pack.name!r}"
                        )
                        continue
                    index[name] = rule
                    owner.setdefault(name, pack.name)
        if problems:
            msg = "selected packs disagree:\n" + "\n".join(f"  - {p}" for p in problems)
            raise CleanPromptError(msg)
        return index

    @staticmethod
    def code_vocabulary(packs: Iterable[PackSpec]) -> CodeSpec:
        """
        Return the union of the packs' code sections.

        Parameters
        ----------
        packs : iterable of PackSpec
            Resolved packs.

        Returns
        -------
        CodeSpec
            Sorted and deduplicated.
        """
        keywords: set[str] = set()
        methods: set[str] = set()
        roles: dict[str, str] = {}
        for pack in packs:
            keywords.update(pack.code.column_keywords)
            methods.update(pack.code.column_methods)
            for dtype, role in pack.code.dtype_roles:
                roles.setdefault(dtype, role)
        return CodeSpec(
            tuple(sorted(keywords)),
            tuple(sorted(methods)),
            tuple(sorted(roles.items())),
        )

    def fingerprint(
        self, packs: Iterable[PackSpec], formats: Iterable[FormatSpec]
    ) -> str:
        """
        Return a digest of the selected definitions' content.

        Parameters
        ----------
        packs, formats : iterable
            The selection.

        Returns
        -------
        str
            Hex SHA-256 over their canonical documents.
        """
        payload = {
            "packs": [self.documents.get(f"pack:{p.name}", p.name) for p in packs],
            "formats": [
                self.documents.get(f"format:{f.name}", f.name) for f in formats
            ],
        }
        return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()

    def merge(self, other: Catalog, conflict: str = "error") -> Catalog:
        """
        Return this catalog with another's definitions added.

        Parameters
        ----------
        other : Catalog
            Usually a custom catalog.
        conflict : {'error', 'replace'}, default='error'
            What to do when both define a name *differently*. An identical
            redefinition is never a conflict.

        Returns
        -------
        Catalog
            A new catalog.

        Raises
        ------
        CleanPromptError
            On a conflict under ``'error'``, naming every clashing definition.
        """
        if conflict not in ("error", "replace"):
            msg = f"conflict must be 'error' or 'replace', got {conflict!r}"
            raise CleanPromptError(msg)
        packs = dict(self.packs)
        formats = dict(self.formats)
        documents = dict(self.documents)
        clashes = []
        for kind, mine, theirs in (
            ("pack", packs, other.packs),
            ("format", formats, other.formats),
        ):
            for name, spec in theirs.items():
                key = f"{kind}:{name}"
                if name in mine and self.documents.get(  # ruff: ignore[collapsible-if]
                    key,
                ) != other.documents.get(key):
                    if conflict == "error":
                        clashes.append(
                            f"{kind} {name!r} ({mine[name].source} vs {spec.source})"
                        )
                        continue
                mine[name] = spec
                if key in other.documents:
                    documents[key] = other.documents[key]
        if clashes:
            msg = (
                "custom definitions redefine existing ones; pass conflict='replace' "
                "(or --replace-builtins) to override them:\n"
                + "\n".join(f"  - {c}" for c in clashes)
            )
            raise CleanPromptError(msg)
        return Catalog(packs, formats, documents)


def catalog_from_documents(
    packs: Mapping[str, Any],
    formats: Mapping[str, Any],
    source: str = "<builtin>",
    partial: bool = False,
) -> Catalog:
    """
    Validate raw documents and build a catalog from them.

    Parameters
    ----------
    packs, formats : mapping
        Name (or file) to parsed document.
    source : str, default='<builtin>'
        Recorded on specs that do not carry a more specific source.
    partial : bool, default=False
        Build a fragment for merging; see :class:`Catalog`.

    Returns
    -------
    Catalog
        Validated.

    Raises
    ------
    PackError
        On the first invalid document, listing all of its problems.
    CleanPromptError
        If the catalog as a whole is inconsistent.
    """
    pack_specs: dict[str, PackSpec] = {}
    format_specs: dict[str, FormatSpec] = {}
    documents: dict[str, str] = {}
    for label, document in sorted(packs.items()):
        spec = pack_from_document(
            document, label if label.endswith((".yaml", ".yml", ".json")) else source
        )
        if spec.name in pack_specs:
            msg = f"pack {spec.name!r} is defined twice ({pack_specs[spec.name].source}, {spec.source})"
            raise CleanPromptError(msg)
        pack_specs[spec.name] = spec
        documents[f"pack:{spec.name}"] = canonical(document)
    for label, document in sorted(formats.items()):
        spec = format_from_document(
            document, label if label.endswith((".yaml", ".yml", ".json")) else source
        )
        if spec.name in format_specs:
            msg = f"format {spec.name!r} is defined twice ({format_specs[spec.name].source}, {spec.source})"
            raise CleanPromptError(msg)
        format_specs[spec.name] = spec
        documents[f"format:{spec.name}"] = canonical(document)
    return Catalog(pack_specs, format_specs, documents, partial=partial)


def require_yaml() -> Any:
    """
    Import PyYAML, or explain how to get it.

    Returns
    -------
    module
        :mod:`yaml`.

    Raises
    ------
    CapabilityError
        When PyYAML is not installed. JSON needs nothing and is suggested.
    """
    try:
        import yaml  # ruff: ignore[import-outside-top-level]
    except ImportError as exc:
        msg = (
            "reading YAML needs PyYAML. Install it, or write the definition as "
            "JSON, which the standard library reads."
        )
        raise CapabilityError(
            msg, tier="yaml", status="ABSENT", install_hint='pip install "pyyaml>=5.1"'
        ) from exc
    return yaml


def at_rest_findings(
    texts: Mapping[str, str],
    packs: Iterable[PackSpec],
) -> list[str]:
    r"""
    Report every credential-shaped value written whole in a definition file.

    Parameters
    ----------
    texts : mapping of str to str
        File label to the file's text, exactly as stored.
    packs : iterable of PackSpec
        The packs whose patterns define "credential-shaped".

    Returns
    -------
    list of str
        One line per finding, naming the file, the line and the pattern kind,
        sorted. Empty when every file is clean. The matched text is never
        included: a finding must not repeat what it found.

    Notes
    -----
    **User notes.** The fix for a finding is always the same: write the
    example as fragments, ``['sk_live_', '0123...']``, which the pack loader
    joins. See :mod:`scikitplot.cleanprompt._packs`.

    **Developer notes — invariant ``I14``.** The built-in definitions are
    committed, pushed and published, so they pass through secret scanners that
    cannot know a value is an example. This holds the files to the package's
    own definition of a credential rather than to a list of what one scanner
    blocks today: a pattern added to a credential pack is enforced on the
    files from the moment it exists. A pattern's validator is applied, so a
    match the detector itself would reject is not a finding.

    Examples
    --------
    >>> from scikitplot.cleanprompt._catalog import at_rest_findings, builtin_catalog
    >>> secrets = [builtin_catalog().packs["secrets"]]
    >>> at_rest_findings({"notes.yaml": "nothing to see"}, secrets)
    []
    >>> at_rest_findings({"notes.yaml": "a\nxox" + "b-1234567890-abcdefghij"}, secrets)
    ['notes.yaml:2: a whole SLACK_TOKEN-shaped value; write the example as fragments']
    """
    findings = set()
    for pack in packs:
        for spec in pack.patterns:
            compiled = re.compile(spec.pattern, spec.flags)
            for label, text in texts.items():
                for match in compiled.finditer(text):
                    if not match.group():
                        continue
                    if spec.validate is not None and not spec.validate(match):
                        continue
                    line = text.count("\n", 0, match.start()) + 1
                    findings.add((label, line, spec.kind))
    return [
        f"{label}:{line}: a whole {kind}-shaped value; write the example as fragments"
        for label, line, kind in sorted(findings)
    ]


def _definition_texts(folder: Path) -> dict[str, str]:
    """Return every YAML definition under ``folder`` as stored, by relative path."""
    texts = {}
    for section in ("packs", "formats"):
        base = folder / section
        for path in sorted(base.glob("*.yaml")) + sorted(base.glob("*.yml")):
            texts[f"{section}/{path.name}"] = path.read_text(encoding="utf-8")
    return texts


def _read_yaml_dir(folder: Path) -> dict[str, Any]:
    """Read every ``*.yaml`` in a folder with ``safe_load``, keyed by file name."""
    yaml = require_yaml()
    documents = {}
    for path in sorted(folder.glob("*.yaml")) + sorted(folder.glob("*.yml")):
        try:
            documents[path.name] = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:  # ruff: ignore[try-except-in-loop]
            msg = f"{path.name}: not valid YAML: {exc}"
            raise PackError(path.name, [msg]) from exc
    return documents


def compile_config(config_dir: str | Path = CONFIG_DIR) -> dict[str, Any]:
    """
    Compile the YAML definitions into the document the base tier reads.

    Parameters
    ----------
    config_dir : str or path-like, default=CONFIG_DIR
        A folder with ``packs/`` and ``formats/`` subfolders.

    Returns
    -------
    dict
        ``{"schema": 1, "packs": {...}, "formats": {...}}``, every document
        validated, keyed by its ``name``.

    Raises
    ------
    CapabilityError
        If PyYAML is not installed.
    PackError
        If any definition is invalid, or if a definition file or the compiled
        document would contain a whole credential-shaped value (``I14``).

    Notes
    -----
    **Developer notes.** The at-rest check runs here, on the YAML as stored
    and on the JSON as it would be written, so the compiled file cannot be
    produced in a state a secret scanner refuses. It is not repeated when the
    base tier loads the catalog: that path must stay cheap, and
    :func:`check_compiled` covers a hand-edited file.
    """
    return _compile(Path(config_dir))[0]


def _compile(folder: Path) -> tuple[dict[str, Any], Catalog]:
    """Compile ``folder``, returning the document and the catalog it validates as."""
    raw_packs = _read_yaml_dir(folder / "packs")
    raw_formats = _read_yaml_dir(folder / "formats")
    catalog = catalog_from_documents(raw_packs, raw_formats)
    compiled = {
        "schema": COMPILED_SCHEMA,
        "packs": {doc["name"]: doc for doc in raw_packs.values()},
        "formats": {doc["name"]: doc for doc in raw_formats.values()},
    }
    texts = _definition_texts(folder)
    texts[COMPILED_PATH.name] = _render(compiled)
    findings = at_rest_findings(texts, _at_rest_packs(catalog))
    if findings:
        raise PackError(str(folder), findings)
    return compiled, catalog


def _at_rest_packs(catalog: Catalog) -> tuple[PackSpec, ...]:
    """Return the credential packs this catalog defines, in declared order."""
    return tuple(catalog.packs[name] for name in AT_REST_PACKS if name in catalog.packs)


def _render(compiled: Mapping[str, Any]) -> str:
    """Return the compiled document as the text written to disk."""
    return json.dumps(compiled, sort_keys=True, indent=1, ensure_ascii=True) + "\n"


def write_compiled(
    config_dir: str | Path = CONFIG_DIR, target: str | Path | None = None
) -> Path:
    """
    Compile the YAML definitions and write the JSON the base tier reads.

    Parameters
    ----------
    config_dir : str or path-like, default=CONFIG_DIR
        The source folder.
    target : str or path-like, optional
        Where to write. Defaults to ``_compiled.json`` inside ``config_dir``.

    Returns
    -------
    pathlib.Path
        The file written.
    """
    folder = Path(config_dir)
    destination = Path(target) if target is not None else folder / "_compiled.json"
    destination.write_text(_render(compile_config(folder)), encoding="utf-8")
    builtin_catalog.cache_clear()
    return destination


def check_compiled(
    config_dir: str | Path = CONFIG_DIR, target: str | Path | None = None
) -> list[str]:
    """
    Report every difference between the YAML source and the compiled JSON.

    Parameters
    ----------
    config_dir : str or path-like, default=CONFIG_DIR
        The source folder.
    target : str or path-like, optional
        The compiled file. Defaults to ``_compiled.json`` inside ``config_dir``.

    Returns
    -------
    list of str
        Empty when they agree. Otherwise one line per differing definition,
        and the command that fixes it.
    """
    folder = Path(config_dir)
    destination = Path(target) if target is not None else folder / "_compiled.json"
    fresh, catalog = _compile(folder)
    if not destination.is_file():
        return [
            f"{destination.name} is missing",
            "run: python -m scikitplot.cleanprompt packs --compile",
        ]
    stored_text = destination.read_text(encoding="utf-8")
    stored = json.loads(stored_text)
    problems = at_rest_findings(
        {destination.name: stored_text},
        _at_rest_packs(catalog),
    )
    for section in ("packs", "formats"):
        mine, theirs = fresh.get(section, {}), stored.get(section, {})
        for name in sorted(set(mine) | set(theirs)):
            if name not in theirs:
                problems.append(
                    f"{section[:-1]} {name!r} is in the YAML but not compiled"
                )
            elif name not in mine:
                problems.append(
                    f"{section[:-1]} {name!r} is compiled but has no YAML source"
                )
            elif canonical(mine[name]) != canonical(theirs[name]):
                problems.append(f"{section[:-1]} {name!r} differs from its YAML source")
    if stored.get("schema") != COMPILED_SCHEMA:
        problems.append(
            f"compiled schema is {stored.get('schema')!r}, expected {COMPILED_SCHEMA}"
        )
    if problems:
        problems.append("run: python -m scikitplot.cleanprompt packs --compile")
    return problems


@lru_cache(maxsize=1)
def builtin_catalog() -> Catalog:
    """
    Return the built-in catalog, read from the compiled JSON.

    Returns
    -------
    Catalog
        Cached for the life of the process.

    Raises
    ------
    CleanPromptError
        If the compiled file is missing or malformed — which is a packaging
        defect, reported as one rather than as an empty catalog.

    Notes
    -----
    **Developer notes.** Standard library only. Every definition is validated
    again on load, which executes every pattern's examples; this costs a few
    milliseconds once per process and means a corrupted or hand-edited file is
    refused rather than trusted.
    """
    if not COMPILED_PATH.is_file():
        msg = (
            f"the built-in pack catalog {COMPILED_PATH.name} is missing from this "
            "installation; this is a packaging defect"
        )
        raise CleanPromptError(msg)
    try:
        document = json.loads(COMPILED_PATH.read_text(encoding="utf-8"))
    except ValueError as exc:
        msg = f"the built-in pack catalog is not valid JSON: {exc}"
        raise CleanPromptError(msg) from exc
    if document.get("schema") != COMPILED_SCHEMA:
        msg = f"the built-in pack catalog has schema {document.get('schema')!r}, expected {COMPILED_SCHEMA}"
        raise CleanPromptError(msg)
    return catalog_from_documents(
        document.get("packs", {}), document.get("formats", {})
    )
