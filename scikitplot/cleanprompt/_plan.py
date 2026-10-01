"""
A fluent, immutable plan: which packs, which formats, which rules.

This is cleanprompt's counterpart to ``FluentCorpus``. A plan is data: it
names a selection and never touches a file. :meth:`FluentCleanPrompt.materialize`
turns it into a :class:`~scikitplot.cleanprompt._runtime.Cleaner`, which does.

Notes
-----
**User notes.** Build it in whatever order reads best; it means the same thing
either way::

    from scikitplot.cleanprompt import FluentCleanPrompt

    cleaner = (
        FluentCleanPrompt()
        .packs("patient", "pandas")  # or "all", or "auto"
        .formats("ipynb", "csv", ".env")  # names or extensions
        .custom("hr_pack.yaml")  # your own definitions
        .style("placeholder")
        .infer_roles()
        .materialize()
    )

    for item in cleaner.encode_tree("project/"):
        print(item.status, item.relative)

Three selection modes cover what people ask for: ``packs("all")`` for
everything, ``packs("addressbook")`` for one domain, and any combination.
``packs("auto")`` lets each file's format choose, which is the default.

**Developer notes — the rules that make a plan trustworthy.**

*Immutable.* Every setter returns a new object; a plan handed to two
cleaners cannot be changed under either.

*Setting twice is an error by default.* ``.packs("a").packs("b")`` is almost
always a mistake — two cells of a notebook that each thought they were in
charge — so it raises unless ``conflict='replace'`` or ``conflict='extend'``
says what was meant. That is corpus's rule, kept for the same reason.

*Validated before it runs.* :meth:`FluentCleanPrompt.validate` returns every
problem at once — unknown pack, unknown format, unknown role, a custom file
that does not load — and :meth:`FluentCleanPrompt.build` raises with all of
them. Nothing about a plan is discovered for the first time halfway through a
directory.

*Fingerprinted by content* (invariant ``I12``). The digest covers the
selection *and* the definitions it resolves to, so editing a pack changes the
fingerprint of every plan that uses it, while reordering ``.packs()``
arguments does not.

See Also
--------
scikitplot.cleanprompt._runtime : The cleaner a plan materializes into.
scikitplot.cleanprompt._catalog : Where selections are resolved.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from ._catalog import ALL, AUTO, NONE, Catalog, builtin_catalog, canonical
from ._engines import ENGINE_MODES
from ._exceptions import CleanPromptError
from ._policy import PROFILES
from ._schema import ROLES
from ._surrogates import STYLES

__all__ = [
    "KEEPABLE",
    "PLAN_SCHEMA",
    "CleanPlan",
    "FluentCleanPrompt",
    "load_plan",
    "plan_from_dict",
    "save_plan",
]

#: Version of the plan-file layout written by :func:`save_plan`.
PLAN_SCHEMA = 1

#: Parts of an artefact removed unless kept.
KEEPABLE = ("figures", "outputs", "paths")

_CONFLICTS = ("error", "replace", "extend")


@dataclass(frozen=True)
class CleanPlan:
    """
    Everything a cleaner needs to know, and nothing it discovers.

    Parameters
    ----------
    packs : tuple of str
        Pack names, or one of ``('all',)``, ``('auto',)``, ``('none',)``.
    formats : tuple of str
        Format names or extensions, or ``('all',)``.
    custom : tuple of str
        Paths of custom definition files.
    conflict : {'error', 'replace'}
        How custom definitions treat a built-in of the same name.
    profile : str or None
        A core-pattern profile: ``'minimal'``, ``'balanced'`` or ``'strict'``.
    core : bool
        Whether the built-in structural patterns run at all.
    style : str
        ``'placeholder'`` or ``'surrogate'``.
    keep : tuple of str
        Artefact parts to send rather than remove.
    roles : tuple of (str, str)
        Declared column roles.
    infer_roles : bool
        Whether to guess column roles from names.
    ner : str
        Entity engine mode; ``'none'`` for no entity detection.
    language : str
        Language for entity detection.
    hide : tuple of str
        Literal terms to hide.
    allow : tuple of str
        Literal terms never to hide.
    remember : bool
        Once a value is hidden, hide it wherever it appears later — in prose,
        in another file, in the next turn — not only where a pattern or field
        rule finds it again.

    Notes
    -----
    **Developer notes.** Tuples, never lists or sets, so a plan is hashable and
    its canonical form does not depend on iteration order: every collection is
    sorted when it enters, by :class:`FluentCleanPrompt`.
    """

    packs: tuple[str, ...] = (AUTO,)
    formats: tuple[str, ...] = (ALL,)
    custom: tuple[str, ...] = ()
    conflict: str = "error"
    profile: str | None = None
    core: bool = True
    style: str = "placeholder"
    keep: tuple[str, ...] = ()
    roles: tuple[tuple[str, str], ...] = ()
    infer_roles: bool = False
    ner: str = NONE
    language: str = "en"
    hide: tuple[str, ...] = ()
    allow: tuple[str, ...] = ()
    remember: bool = True
    configured: tuple[str, ...] = field(default=(), compare=False)

    def as_dict(self) -> dict[str, Any]:
        """
        Return the plan as plain data.

        Returns
        -------
        dict
            JSON-safe. ``configured`` is left out: it records how the plan was
            built, not what it means.
        """
        out = {}
        for item in fields(self):
            if item.name == "configured":
                continue
            value = getattr(self, item.name)
            out[item.name] = (
                [list(pair) for pair in value]
                if item.name == "roles"
                else (list(value) if isinstance(value, tuple) else value)
            )
        return out

    def packs_selection(self) -> str | tuple[str, ...]:
        """Return ``packs`` in the form :meth:`Catalog.resolve_packs` takes."""
        return (
            self.packs[0]
            if len(self.packs) == 1 and self.packs[0] in (ALL, AUTO, NONE)
            else self.packs
        )

    def formats_selection(self) -> str | tuple[str, ...]:
        """Return ``formats`` in the form :meth:`Catalog.resolve_formats` takes."""
        return (
            self.formats[0]
            if len(self.formats) == 1 and self.formats[0] in (ALL, NONE)
            else self.formats
        )

    def catalog(self) -> Catalog:
        """
        Return the catalog this plan resolves against.

        Returns
        -------
        Catalog
            The built-ins, with the plan's custom definitions merged in.
        """
        if not self.custom:
            return builtin_catalog()
        from ._custom import with_custom  # ruff: ignore[import-outside-top-level]

        return with_custom(self.custom, conflict=self.conflict)

    def validate(self) -> list[str]:
        """
        Return every problem with this plan, without raising.

        Returns
        -------
        list of str
            Empty when the plan can be materialized.
        """
        problems: list[str] = []
        if self.conflict not in ("error", "replace"):
            problems.append(f"conflict: {self.conflict!r} must be 'error' or 'replace'")
        if self.profile is not None and self.profile not in PROFILES:
            problems.append(
                f"profile: {self.profile!r} is not one of {', '.join(sorted(PROFILES))}"
            )
        if self.style not in STYLES:
            problems.append(f"style: {self.style!r} is not one of {', '.join(STYLES)}")
        problems.extend(
            f"keep: {part!r} is not one of {', '.join(KEEPABLE)}"
            for part in self.keep
            if part not in KEEPABLE
        )
        for column, role in self.roles:
            if role not in ROLES:
                problems.append(f"roles: {column!r} has unknown role {role!r}")
        if self.ner not in (*ENGINE_MODES, NONE):
            problems.append(
                f"ner: {self.ner!r} is not one of {', '.join(ENGINE_MODES)}"
            )
        try:
            catalog = self.catalog()
        except CleanPromptError as exc:
            problems.append(f"custom: {exc}")
            return problems
        try:
            formats = catalog.resolve_formats(self.formats_selection())
        except CleanPromptError as exc:
            problems.append(f"formats: {exc}")
            formats = ()
        try:
            packs = catalog.resolve_packs(self.packs_selection(), formats=formats)
            catalog.field_index(packs)
        except CleanPromptError as exc:
            problems.append(f"packs: {exc}")
        return problems

    def fingerprint(self) -> str:
        """
        Return a digest of the selection and the definitions it resolves to.

        Returns
        -------
        str
            Hex SHA-256.

        Raises
        ------
        CleanPromptError
            If the plan does not validate.
        """
        problems = self.validate()
        if problems:
            msg = "the plan is not valid:\n" + "\n".join(f"  - {p}" for p in problems)
            raise CleanPromptError(msg)
        catalog = self.catalog()
        formats = catalog.resolve_formats(self.formats_selection())
        packs = catalog.resolve_packs(self.packs_selection(), formats=formats)
        payload = {
            "plan": self.as_dict(),
            "definitions": catalog.fingerprint(packs, formats),
            "schema": "cleanplan1",
        }
        return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()


_BOOL_FIELDS = ("core", "infer_roles", "remember")
_TUPLE_FIELDS = ("packs", "formats", "custom", "keep", "hide", "allow")
_TEXT_FIELDS = ("conflict", "style", "ner", "language")


def plan_from_dict(  # ruff: ignore[too-many-branches]
    document: Mapping[str, Any],
    source: str = "<plan>",
) -> CleanPlan:
    """
    Build a plan from plain data, refusing anything it does not understand.

    Parameters
    ----------
    document : mapping
        What :meth:`CleanPlan.as_dict` produces, optionally with ``schema``
        and ``fingerprint`` keys.
    source : str, default='<plan>'
        Named in messages.

    Returns
    -------
    CleanPlan
        Not yet validated against the catalog; see :meth:`CleanPlan.validate`.

    Raises
    ------
    CleanPromptError
        Listing every malformed or unknown key at once.
    """
    if not isinstance(document, Mapping):
        msg = f"{source}: a plan must be a mapping"
        raise CleanPromptError(msg)
    known = {item.name for item in fields(CleanPlan)} - {"configured"}
    problems = [
        f"unknown key {key!r}"
        for key in sorted(set(document) - known - {"schema", "fingerprint"})
    ]
    if document.get("schema", PLAN_SCHEMA) != PLAN_SCHEMA:
        problems.append(f"schema {document.get('schema')!r} is not {PLAN_SCHEMA}")
    values: dict[str, Any] = {}
    for key in _TUPLE_FIELDS:
        if key in document:
            raw = document[key]
            if not isinstance(raw, list) or not all(
                isinstance(item, str) and item for item in raw
            ):
                problems.append(f"{key}: must be a list of non-empty strings")
            else:
                values[key] = tuple(sorted(set(raw)))
    for key in _BOOL_FIELDS:
        if key in document:
            if not isinstance(document[key], bool):
                problems.append(f"{key}: must be true or false")
            else:
                values[key] = document[key]
    for key in _TEXT_FIELDS:
        if key in document:
            if not isinstance(document[key], str):
                problems.append(f"{key}: must be a string")
            else:
                values[key] = document[key]
    if "profile" in document:
        if document["profile"] is not None and not isinstance(document["profile"], str):
            problems.append("profile: must be a string or null")
        else:
            values["profile"] = document["profile"]
    if "roles" in document:
        raw = document["roles"]
        if not isinstance(raw, list) or not all(
            isinstance(pair, list)
            and len(pair) == 2  # ruff: ignore[magic-value-comparison]
            and all(isinstance(x, str) for x in pair)
            for pair in raw
        ):
            problems.append("roles: must be a list of [column, role] pairs")
        else:
            values["roles"] = tuple(sorted((column, role) for column, role in raw))
    if problems:
        msg = f"{source}: the plan file is not valid:\n" + "\n".join(
            f"  - {p}" for p in problems
        )
        raise CleanPromptError(msg)
    return CleanPlan(**values, configured=tuple(sorted(values)))


def save_plan(plan: CleanPlan, path: str | os.PathLike) -> str:
    """
    Write a plan, with its fingerprint, to a JSON file a team can share.

    Parameters
    ----------
    plan : CleanPlan
        Must validate.
    path : path-like
        The file to write.

    Returns
    -------
    str
        The fingerprint written.

    Raises
    ------
    CleanPromptError
        If the plan does not validate.

    Notes
    -----
    **User notes.** Commit the file. Paths of custom pack files are kept as
    written and read relative to the folder you run from, so save and use a
    plan from the project root.
    Everyone who runs with ``--plan`` gets
    the same packs, formats and rules — and if a pack those rules depend on
    changes (a new version of scikit-plots, an edited custom file), the
    fingerprint no longer matches and the run is refused until someone looks
    at the change and saves the plan again.
    """
    fingerprint = plan.fingerprint()
    document = {"schema": PLAN_SCHEMA, **plan.as_dict(), "fingerprint": fingerprint}
    Path(path).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return fingerprint


def load_plan(path: str | os.PathLike, check: bool = True) -> CleanPlan:
    """
    Read a plan file, and refuse it if what it means has changed.

    Parameters
    ----------
    path : path-like
        A ``.json`` plan file (``.yaml`` also works with PyYAML).
    check : bool, default=True
        Compare the file's ``fingerprint``, when it has one, with the plan's
        fingerprint now.

    Returns
    -------
    CleanPlan
        Validated.

    Raises
    ------
    CleanPromptError
        If the file is malformed, the plan does not validate, or the
        definitions it resolves to have changed since it was saved.

    Notes
    -----
    **Developer notes.** The fingerprint covers the selection *and* the
    content of every pack and format it resolves to (invariant ``I12``), so
    this check is what turns "we approved this plan" into something a run can
    verify rather than assume.
    """
    from ._custom import _read  # ruff: ignore[import-outside-top-level]

    source = Path(path)
    document = _read(source)
    plan = plan_from_dict(document, source.name)
    problems = plan.validate()
    if problems:
        msg = f"{source.name}: the plan is not valid:\n" + "\n".join(
            f"  - {p}" for p in problems
        )
        raise CleanPromptError(msg)
    expected = document.get("fingerprint")
    if check and expected is not None:
        actual = plan.fingerprint()
        if expected != actual:
            msg = (
                f"{source.name}: the packs or formats this plan uses have changed since it was "
                f"saved (fingerprint {str(expected)[:12]}… is now {actual[:12]}…). Review the change, "
                "then save the plan again."
            )
            raise CleanPromptError(msg)
    return plan


def _names(values: Iterable[Any], what: str) -> tuple[str, ...]:
    """Return non-empty strings, or raise naming the bad one."""
    out = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            msg = f"{what}: every entry must be a non-empty string, got {value!r}"
            raise CleanPromptError(msg)
        out.append(value.strip())
    return tuple(out)


class FluentCleanPrompt:
    """
    Build a :class:`CleanPlan` one decision at a time.

    Parameters
    ----------
    plan : CleanPlan, optional
        Start from an existing plan. Defaults to ``packs('auto')`` over every
        format, placeholders, core patterns on, no entity engine.

    Notes
    -----
    **User notes.** Every method returns a *new* builder. Each domain may be set
    once; set it again with ``conflict='replace'`` to overwrite or
    ``conflict='extend'`` to add to it.

    Examples
    --------
    >>> plan = FluentCleanPrompt().packs("patient").formats("csv").plan()
    >>> plan.packs, plan.formats
    (('patient',), ('csv',))
    >>> FluentCleanPrompt().packs("patient").packs("finance")
    Traceback (most recent call last):
    ...
    scikitplot.cleanprompt._exceptions.CleanPromptError: packs is already set to ('patient',); pass conflict='replace' or conflict='extend'
    >>> sorted(
    ...     FluentCleanPrompt()
    ...     .packs("patient")
    ...     .packs("finance", conflict="extend")
    ...     .plan()
    ...     .packs
    ... )
    ['finance', 'patient']
    """

    __slots__ = ("_plan",)

    def __init__(self, plan: CleanPlan | None = None) -> None:
        self._plan = plan if plan is not None else CleanPlan()

    def __repr__(self) -> str:
        configured = ", ".join(self._plan.configured) or "defaults"
        return f"FluentCleanPrompt({configured})"

    def _with(
        self, domain: str, value: Any, conflict: str = "error"
    ) -> FluentCleanPrompt:
        """Return a builder with one domain set, honouring ``conflict``."""
        if conflict not in _CONFLICTS:
            msg = f"conflict must be one of {', '.join(_CONFLICTS)}, got {conflict!r}"
            raise CleanPromptError(msg)
        plan = self._plan
        if domain in plan.configured:
            if conflict == "error":
                msg = (
                    f"{domain} is already set to {getattr(plan, domain)!r}; "
                    "pass conflict='replace' or conflict='extend'"
                )
                raise CleanPromptError(msg)
            if conflict == "extend":
                current = getattr(plan, domain)
                if not isinstance(current, tuple):
                    msg = f"{domain} holds one value and cannot be extended"
                    raise CleanPromptError(msg)
                value = tuple(sorted(set(current) | set(value)))
        configured = tuple(sorted(set(plan.configured) | {domain}))
        return FluentCleanPrompt(
            replace(plan, **{domain: value}, configured=configured)
        )

    # -- selection --------------------------------------------------------

    def packs(self, *names: str, conflict: str = "error") -> FluentCleanPrompt:
        """
        Choose packs: names, or ``'all'``, ``'auto'`` or ``'none'``.

        Parameters
        ----------
        *names : str
            One or more pack names, or a single mode keyword.
        conflict : {'error', 'replace', 'extend'}, default='error'
            What to do if packs were already chosen.

        Returns
        -------
        FluentCleanPrompt
            A new builder.
        """
        return self._with(
            "packs", tuple(sorted(set(_names(names or (ALL,), "packs")))), conflict
        )

    def formats(self, *names: str, conflict: str = "error") -> FluentCleanPrompt:
        """
        Choose which formats are read: names, extensions, or ``'all'``.

        Parameters
        ----------
        *names : str
            Format names (``'csv'``), extensions (``'.env'``) or ``'all'``.
        conflict : {'error', 'replace', 'extend'}, default='error'
            What to do if formats were already chosen.

        Returns
        -------
        FluentCleanPrompt
            A new builder.
        """
        return self._with(
            "formats", tuple(sorted(set(_names(names or (ALL,), "formats")))), conflict
        )

    def custom(
        self, *paths: str, conflict: str = "error", replace_builtins: bool = False
    ) -> FluentCleanPrompt:
        """
        Add custom pack or format definitions from YAML or JSON files.

        Parameters
        ----------
        *paths : str
            Definition files.
        conflict : {'error', 'replace', 'extend'}, default='error'
            What to do if custom files were already given.
        replace_builtins : bool, default=False
            Let a custom definition replace a built-in of the same name.

        Returns
        -------
        FluentCleanPrompt
            A new builder.
        """
        builder = self._with(
            "custom", tuple(sorted(set(_names(paths, "custom")))), conflict
        )
        if replace_builtins:
            builder = builder._with("conflict", "replace", "replace")
        return builder

    # -- rules ------------------------------------------------------------

    def profile(self, name: str, conflict: str = "error") -> FluentCleanPrompt:
        """Choose the core-pattern profile: ``minimal``, ``balanced`` or ``strict``."""
        return self._with("profile", name, conflict)

    def core(self, enabled: bool = True, conflict: str = "error") -> FluentCleanPrompt:
        """Turn the built-in structural patterns on or off."""
        return self._with("core", bool(enabled), conflict)

    def style(self, name: str, conflict: str = "error") -> FluentCleanPrompt:
        """Choose ``placeholder`` (``[EMAIL-1]``) or ``surrogate`` stand-ins."""
        return self._with("style", name, conflict)

    def keep(self, *parts: str, conflict: str = "error") -> FluentCleanPrompt:
        """Send artefact parts removed by default: ``outputs``, ``figures``, ``paths``."""
        return self._with("keep", tuple(sorted(set(_names(parts, "keep")))), conflict)

    def roles(
        self,
        mapping: Mapping[str, str] | None = None,
        conflict: str = "error",
        **named: str,
    ) -> FluentCleanPrompt:
        """Declare column roles, e.g. ``roles(churned="target")``."""
        merged = dict(mapping or {})
        merged.update(named)
        return self._with("roles", tuple(sorted(merged.items())), conflict)

    def infer_roles(
        self, enabled: bool = True, conflict: str = "error"
    ) -> FluentCleanPrompt:
        """Guess column roles from their names; reported as inferred."""
        return self._with("infer_roles", bool(enabled), conflict)

    def ner(
        self, engine: str = "auto", language: str = "en", conflict: str = "error"
    ) -> FluentCleanPrompt:
        """Turn on entity detection with ``auto``, ``spacy``, ``nltk`` or ``both``."""
        return self._with("ner", engine, conflict)._with(
            "language", language, "replace"
        )

    def hide(self, *terms: str, conflict: str = "error") -> FluentCleanPrompt:
        """Hide these exact strings wherever they appear."""
        return self._with("hide", tuple(sorted(set(_names(terms, "hide")))), conflict)

    def allow(self, *terms: str, conflict: str = "error") -> FluentCleanPrompt:
        """Never hide these exact strings."""
        return self._with("allow", tuple(sorted(set(_names(terms, "allow")))), conflict)

    def remember(
        self, enabled: bool = True, conflict: str = "error"
    ) -> FluentCleanPrompt:
        """
        Hide a value everywhere once it has been hidden anywhere (default on).

        Notes
        -----
        **User notes.** A patient's name found in the ``name`` column of a CSV
        is then also hidden in the free-text note encoded after it, where no
        field rule could see it. Turn it off only when you need each file's
        output to be independent of what was encoded before.
        """
        return self._with("remember", bool(enabled), conflict)

    # -- terminal operations ---------------------------------------------

    def plan(self) -> CleanPlan:
        """Return the plan as it stands, without validating it."""
        return self._plan

    def validate(self) -> list[str]:
        """Return every problem with the plan; empty when it can run."""
        return self._plan.validate()

    def build(self) -> CleanPlan:
        """
        Return the plan, or raise listing every problem with it.

        Returns
        -------
        CleanPlan
            Validated.

        Raises
        ------
        CleanPromptError
            With one line per problem.
        """
        problems = self.validate()
        if problems:
            msg = "the plan is not valid:\n" + "\n".join(f"  - {p}" for p in problems)
            raise CleanPromptError(msg)
        return self._plan

    def materialize(self):
        """
        Return a :class:`~scikitplot.cleanprompt._runtime.Cleaner` for this plan.

        Returns
        -------
        Cleaner
            Ready to encode text, files, directories and archives.

        Raises
        ------
        CleanPromptError
            If the plan does not validate, or an entity engine it asks for is
            not usable.
        """
        from ._runtime import Cleaner  # ruff: ignore[import-outside-top-level]

        return Cleaner(self.build())

    def guard(  # noqa: A002 - public keyword
        self,
        format: str = "text",
        verify: bool = True,
    ):
        """
        Return a :class:`~scikitplot.cleanprompt._guard.Guard` for this plan.

        Parameters
        ----------
        format : str, default='text'
            How prompts are read.
        verify : bool, default=True
            Refuse to send when the independent leak check finds a value.

        Returns
        -------
        Guard
            Ready to put in front of any model client.
        """
        from ._guard import Guard  # ruff: ignore[import-outside-top-level]

        return Guard(self.materialize(), format=format, verify=verify)
