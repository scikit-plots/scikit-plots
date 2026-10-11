r"""
Detection packs: one domain's rules, as validated data.

A pack is everything cleanprompt needs to know about one kind of data — a
patient record, an address book, a pandas notebook, a deployment script — and
nothing about how to process a file. Packs are authored in YAML under
``_config/packs``, shipped compiled to JSON, and may be supplied by a user.

Notes
-----
**User notes.** A pack has up to four parts, each optional::

    name: patient
    version: 1
    summary: Clinical record identifiers.
    requires: [personal]
    fields:                     # a value under one of these keys is hidden
      - names: [mrn, medical_record_number]
        kind: MRN
        role: id
    patterns:                   # a value of this shape is hidden, anywhere
      - kind: NHS_NUMBER
        pattern: '\b\d{3}[ -]?\d{3}[ -]?\d{4}\b'
        intent: A UK NHS number.
        validate: nhs_mod11
        examples_yes: ['943 476 5919']
        examples_no: ['943 476 5918']
    code:                       # extra column-naming sites for notebooks
      column_keywords: [value_vars]

``fields`` is the part that matters most for records. ``"mrn": "00412345"``
is an eight-digit number to a pattern and a medical record number to anyone
who reads the key; a field rule reads the key.

**Developer notes — validation is total and runs the examples.**

:func:`pack_from_document` never accepts a partly valid pack. It collects
*every* problem in one pass — unknown keys, a malformed name, a regular
expression that does not compile, a validator that does not exist — and raises
once with the whole list, because a user fixing a pack one error per run is a
user who stops writing packs.

Every pattern's ``examples_yes`` must each be matched **in full** and every
``examples_no`` must produce **no** accepted match, and this is executed at
load time, not in a test suite somewhere. That is invariant ``I11``: a pack
cannot ship a regular expression that disagrees with its own documentation, and
a custom pack cannot either. A pattern with no positive examples is refused
for the same reason — an untested pattern is a claim nobody checked.

**User notes — an example that looks like a credential is written in pieces.**

An example may be a list of fragments instead of a string; the fragments are
joined, and the pattern is tested against the joined value::

    examples_yes: [["sk_live_", "0123456789abcdefABCDEF"]]

The file then holds no whole credential-shaped value. The built-in ``secrets``
pack is written this way throughout, and the catalog compiler refuses to build
if any built-in definition file matches one of that pack's own patterns
(invariant ``I14``). The reason is practical: a positive example for a key
pattern *is* a string a secret scanner blocks, so the file that documents how
keys are recognised could not be pushed or published.

**User notes — hide the value, not its label.** A pattern that needs a label
to be sure (``MRN: 00412345``) may wrap the part to hide in a group named
``value``: ``'(?i)\bMRN\s*:?\s*(?P<value>\d{4,12})\b'``. Only the group is
replaced, so the text keeps reading ``MRN: [MRN-1]``, and the same number
found by a field rule in a CSV gets the *same* label — the vault is keyed on
the value, and ``MRN: 00412345`` is not the value.

Unknown keys are errors rather than warnings. ``fields:`` is a typo that would
otherwise produce a pack that loads cleanly and hides nothing.

See Also
--------
scikitplot.cleanprompt._catalog : Loads, compiles and selects packs.
scikitplot.cleanprompt._hooks : The validators a pack may name.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ._canonical import detection_view
from ._detectors import RegexDetector
from ._exceptions import CleanPromptError
from ._hooks import get_validator, validator_names
from ._patterns import PatternSpec
from ._schema import ROLES
from ._types import Span

__all__ = [
    "VALUE_GROUP",
    "CodeSpec",
    "FieldSpec",
    "PackError",
    "PackPatternDetector",
    "PackSpec",
    "normalise_field",
    "pack_detectors",
    "pack_from_document",
]

#: The named group a pattern may use to mark the part to hide.
VALUE_GROUP = "value"

#: A pack name: lower case, starts with a letter, at most 32 characters.
_PACK_NAME = re.compile(r"^[a-z][a-z0-9_]{1,31}$")

#: A kind: upper case, starts with a letter, no grammar characters. The
#: placeholder separator ``-`` is excluded so a kind can never make a label
#: ambiguous.
_KIND = re.compile(r"^[A-Z][A-Z0-9_]{1,39}$")

#: A Python identifier, for the ``code`` section.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: The regular-expression flags a pack may request, by name.
_FLAGS = {
    "ASCII": re.ASCII,
    "DOTALL": re.DOTALL,
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "VERBOSE": re.VERBOSE,
}

#: A bound on a pattern's source length. Not a defence against catastrophic
#: backtracking — nothing short of a regex engine with a timeout is — but a
#: pack that needs more than this is a pack that should be several patterns.
_MAX_PATTERN = 2000

#: The fewest fragments a fragmented example may have. One would be a plain
#: string written as a list.
_MIN_FRAGMENTS = 2

#: The keys each section may carry. Anything else is an error.
_TOP_KEYS = frozenset(
    {"name", "version", "summary", "requires", "fields", "patterns", "code"}
)
_FIELD_KEYS = frozenset({"names", "kind", "role", "suffix", "span"})
_PATTERN_KEYS = frozenset(
    {
        "kind",
        "pattern",
        "intent",
        "priority",
        "flags",
        "validate",
        "confidence",
        "examples_yes",
        "examples_no",
        "risk",
        "risk_reason",
    }
)

#: The only value ``risk`` may take: the pack accepts the pattern's shape.
RISK_ACCEPTED = "accepted"

_CODE_KEYS = frozenset({"column_keywords", "column_methods", "dtype_roles"})

#: Splits camelCase so ``DateOfBirth`` and ``date_of_birth`` are one field.
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATORS = re.compile(r"[^a-z0-9]+")


class PackError(CleanPromptError):
    """
    A pack document is invalid.

    Parameters
    ----------
    source : str
        Where the document came from — a file name, or ``'<builtin>'``.
    problems : iterable of str
        Every problem found, in document order.

    Notes
    -----
    **Developer notes.** All problems travel together. The message is written
    for the person editing the file: one line per problem, each naming the
    path inside the document.
    """

    def __init__(self, source: str, problems: Iterable[str]) -> None:
        self.source = source
        self.problems = tuple(problems)
        listing = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(
            f"{source}: {len(self.problems)} problem(s) in definition\n{listing}"
        )


def normalise_field(name: str) -> str:
    """
    Return the canonical spelling of a field or column name.

    Parameters
    ----------
    name : str
        The name as written in the data.

    Returns
    -------
    str
        Lower case, camelCase split, every run of non-alphanumerics folded to a
        single underscore, trimmed.

    Notes
    -----
    **Developer notes.** A record written by one tool says ``DateOfBirth``, by
    another ``date_of_birth``, by a spreadsheet ``Date of Birth``. They are one
    field, and a rule that listed every spelling would be a rule that missed
    the next one.

    The name is read through the detection view first (``CP-103``): an
    invisible character inside it (``n\u200bame``) or full-width letters
    (``\uff4e\uff41\uff4d\uff45``) still name the field. Without that, one
    zero-width space in a header turned ``name`` into ``n_ame``, the field
    rule did not apply, and the column's values went out in the clear. The
    same function builds the index and looks names up, so the two cannot
    disagree.

    Examples
    --------
    >>> normalise_field("DateOfBirth")
    'date_of_birth'
    >>> normalise_field("  Date of-Birth ")
    'date_of_birth'
    >>> normalise_field("patientMRN")
    'patient_mrn'
    >>> normalise_field("n\u200bame")
    'name'
    """
    written = str(name)
    view = detection_view(written)
    split = _CAMEL.sub("_", (view.text if view is not None else written).strip())
    return _SEPARATORS.sub("_", split.lower()).strip("_")


@dataclass(frozen=True)
class FieldSpec:
    """
    A field whose value is sensitive wherever it appears in a record.

    Parameters
    ----------
    names : tuple of str
        Normalised names that identify the field.
    kind : str
        The kind its values are recorded under.
    role : str
        The schema role a column of this name carries, for the role-preserving
        stand-ins of notebooks and modules.
    suffix : bool, default=True
        Whether a key that *ends* with one of the names also matches, so that
        ``billing_phone`` is a ``phone``. Off for names too generic to trust
        as a suffix: ``state`` would otherwise match ``loading_state``.
    span : {'line', 'clause', 'token'}, default='line'
        How far a value runs in *prose* (``Key: value`` in running text).
        ``line`` hides to the end of the line — right for names, addresses,
        diagnoses and dates, which may contain a comma. ``clause`` stops at
        the first comma or semicolon — right for identifiers whose format
        never contains one but may contain spaces (a telephone number).
        ``token`` stops at the first whitespace — right only for identifiers
        whose format contains neither, so that ``MRN: 00412345 for
        ann@example.com`` hides the number as an MRN and leaves the address to
        the email rule. Choose by the identifier's *format*; when in doubt,
        ``line`` hides more and never less. Structured formats are unaffected:
        a CSV cell or a JSON value is always the whole value.
    """

    names: tuple[str, ...]
    kind: str
    role: str = "field"
    suffix: bool = True
    span: str = "line"


@dataclass(frozen=True)
class CodeSpec:
    """
    Library-specific column-naming sites, extending ``_code.py``.

    Parameters
    ----------
    column_keywords : tuple of str
        Keyword arguments whose string values name columns.
    column_methods : tuple of str
        Methods whose first positional argument names a column.
    dtype_roles : tuple of (str, str)
        dtype spellings and the role each establishes.
    """

    column_keywords: tuple[str, ...] = ()
    column_methods: tuple[str, ...] = ()
    dtype_roles: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class PackSpec:
    """
    One validated pack.

    Parameters
    ----------
    name : str
        Unique pack name.
    version : int
        The pack's own version, bumped when its rules change meaning.
    summary : str
        One line, shown by ``cleanprompt packs``.
    requires : tuple of str
        Packs this one builds on.
    fields : tuple of FieldSpec
        Field rules.
    patterns : tuple of PatternSpec
        Value-shape rules, every one already checked against its examples.
    code : CodeSpec
        Column-naming sites for code artefacts.
    source : str
        Where it was loaded from.

    Notes
    -----
    **Developer notes.** Frozen and hashable. ``source`` is excluded from
    equality, so the same pack loaded from a built-in and from a user file
    compares equal — which is what lets a catalog merge treat an identical
    redefinition as harmless and a *different* one as a conflict.
    """

    name: str
    version: int
    summary: str
    requires: tuple[str, ...] = ()
    fields: tuple[FieldSpec, ...] = ()
    patterns: tuple[PatternSpec, ...] = ()
    code: CodeSpec = CodeSpec()
    source: str = field(default="<builtin>", compare=False)

    def field_index(self) -> dict[str, FieldSpec]:
        """
        Return normalised field name to its rule.

        Returns
        -------
        dict
            One entry per name of every field.
        """
        return {name: rule for rule in self.fields for name in rule.names}


def _section(
    document: Mapping[str, Any], key: str, empty: Any, problems: list[str]
) -> Any:
    """
    Return an optional section, refusing one that is declared and left empty.

    Notes
    -----
    **Developer notes.** A missing key means "this pack has no such section".
    A key that is present with nothing in it (``fields: []``, ``patterns:``)
    says a section was intended and never written; treating it as absent made
    a half-finished pack load as if it were complete. The value is returned
    as written so the section's own parser reports a wrong type.
    """
    if key not in document:
        return empty
    value = document[key]
    if value is None or (isinstance(value, (list, Mapping)) and not value):
        problems.append(f"{key}: is declared but empty; add a rule or remove the key")
        return empty
    return value


def _example_list(value: Any, path: str, problems: list[str]) -> tuple[str, ...]:
    """
    Validate a list of examples, joining any written as fragments.

    Parameters
    ----------
    value : object
        The ``examples_yes`` or ``examples_no`` entry as parsed.
    path : str
        Where it is, for messages.
    problems : list of str
        Collects every problem found.

    Returns
    -------
    tuple of str
        One whole example per entry.

    Notes
    -----
    **User notes.** An example is a string, or a list of two or more strings
    that are joined with nothing between them::

        examples_yes: [["EMP-", "004121"]]  # the example is EMP-004121

    Write it as fragments when the whole value has the shape of a real
    credential. The pattern is still tested against the joined value, but the
    file itself never contains it, so a secret scanner reading the repository
    or a published package has nothing to find.

    **Developer notes.** The fragments, not the joined value, are what
    ``_compiled.json`` stores: the catalog compiler copies the document as
    written. Joining happens only here, in memory. A one-fragment list is
    refused because it is a plain string wearing a disguise, and an empty
    fragment because it splits nothing.
    """
    if not isinstance(value, list) or not value:
        problems.append(f"{path}: must be a non-empty list of examples")
        return ()
    out = []
    for index, item in enumerate(value):
        where = f"{path}[{index}]"
        if isinstance(item, str):
            if not item.strip():
                problems.append(f"{where}: must be a non-empty string")
                continue
            out.append(item)
        elif isinstance(item, list):
            if len(item) < _MIN_FRAGMENTS or not all(
                isinstance(part, str) and part for part in item
            ):
                problems.append(
                    f"{where}: a fragmented example is a list of "
                    f"{_MIN_FRAGMENTS} or more non-empty strings"
                )
                continue
            out.append("".join(item))
        else:
            problems.append(
                f"{where}: must be a string, or a list of string fragments to join"
            )
    return tuple(out)


def _string_list(
    value: Any, path: str, problems: list[str], identifiers: bool = False
) -> tuple[str, ...]:
    """Validate a list of non-empty strings, collecting problems."""
    if not isinstance(value, list) or not value:
        problems.append(f"{path}: must be a non-empty list of strings")
        return ()
    out = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            problems.append(f"{path}[{index}]: must be a non-empty string")
            continue
        if identifiers and not _IDENTIFIER.match(item):
            problems.append(f"{path}[{index}]: {item!r} is not a Python identifier")
            continue
        out.append(item)
    return tuple(out)


def _unknown(
    keys: Iterable[str], allowed: frozenset[str], path: str, problems: list[str]
) -> None:
    """Record every key not in ``allowed``."""
    for key in sorted(set(keys) - allowed):
        hint = ", ".join(sorted(allowed))
        problems.append(f"{path}: unknown key {key!r} (allowed: {hint})")


def _parse_fields(value: Any, problems: list[str]) -> tuple[FieldSpec, ...]:
    """Validate the ``fields`` section."""
    if not isinstance(value, list):
        problems.append("fields: must be a list")
        return ()
    out = []
    seen: dict[str, str] = {}
    for index, item in enumerate(value):
        path = f"fields[{index}]"
        if not isinstance(item, Mapping):
            problems.append(f"{path}: must be a mapping")
            continue
        _unknown(item.keys(), _FIELD_KEYS, path, problems)
        names = _string_list(item.get("names"), f"{path}.names", problems)
        kind = item.get("kind")
        if not isinstance(kind, str) or not _KIND.match(kind):
            problems.append(f"{path}.kind: {kind!r} must match {_KIND.pattern}")
            continue
        suffix = item.get("suffix", True)
        if not isinstance(suffix, bool):
            problems.append(f"{path}.suffix: must be true or false")
            continue
        span = item.get("span", "line")
        if span not in ("line", "clause", "token"):
            problems.append(
                f"{path}.span: {span!r} must be 'line', 'clause' or 'token'"
            )
            continue
        role = item.get("role", "field")
        if role not in ROLES:
            problems.append(
                f"{path}.role: {role!r} is not one of {', '.join(sorted(ROLES))}"
            )
            continue
        normalised = []
        for name in names:
            canonical = normalise_field(name)
            if not canonical:
                problems.append(f"{path}.names: {name!r} normalises to nothing")
                continue
            if canonical in seen and seen[canonical] != kind:
                problems.append(
                    f"{path}.names: {name!r} is already field kind {seen[canonical]}"
                )
                continue
            seen[canonical] = kind
            normalised.append(canonical)
        if normalised:
            out.append(
                FieldSpec(tuple(sorted(set(normalised))), kind, role, suffix, span)
            )
    return tuple(out)


def _check_examples(spec: PatternSpec, path: str, problems: list[str]) -> None:
    """Execute a pattern's examples: invariant I11."""
    compiled = re.compile(spec.pattern, spec.flags)

    def accepted(text: str) -> list[str]:
        return [
            match.group()
            for match in compiled.finditer(text)
            if match.group() and (spec.validate is None or spec.validate(match))
        ]

    for example in spec.examples_yes:
        if example not in accepted(example):
            problems.append(
                f"{path}.examples_yes: {example!r} is not matched in full "
                "(and accepted by its validator)"
            )
        elif VALUE_GROUP in compiled.groupindex:
            match = compiled.search(example)
            if match is None or not match.group(VALUE_GROUP):
                problems.append(
                    f"{path}.examples_yes: {example!r} leaves the {VALUE_GROUP!r} group empty"
                )
    for example in spec.examples_no:
        found = accepted(example)
        if found:
            problems.append(
                f"{path}.examples_no: {example!r} is matched as {found[0]!r}"
            )


def _risk_acceptance(
    item: Mapping[str, Any], path: str, problems: list[str]
) -> str | None:
    """
    Validate ``risk`` and ``risk_reason`` together; return the reason or None.

    Notes
    -----
    **User notes.** Accepting a pattern's backtracking risk takes both keys::

        risk: accepted
        risk_reason: inputs are single ticket ids, never prose

    **Developer notes.** The pair is all or nothing. ``risk`` without a reason
    is an acceptance nobody can review; a reason without ``risk: accepted``
    is a comment that silences nothing and would mislead the reader into
    thinking it did. ``accepted`` is the only value so that a future value
    (a per-pattern mode, say) is an explicit schema change, not a typo that
    happens to load.
    """
    has_risk, has_reason = "risk" in item, "risk_reason" in item
    if not (has_risk or has_reason):
        return None
    risk, reason = item.get("risk"), item.get("risk_reason")
    if has_risk and risk != RISK_ACCEPTED:
        problems.append(f"{path}.risk: {risk!r} must be {RISK_ACCEPTED!r}")
        return None
    if not has_risk:
        problems.append(
            f"{path}.risk_reason: add `risk: {RISK_ACCEPTED}` too, or remove the reason"
        )
        return None
    if not isinstance(reason, str) or not reason.strip():
        problems.append(
            f"{path}.risk_reason: `risk: {RISK_ACCEPTED}` needs a non-empty reason "
            "saying why this pattern's inputs are safe"
        )
        return None
    return reason.strip()


def _parse_patterns(  # ruff: ignore[too-many-branches]
    value: Any,
    problems: list[str],
) -> tuple[PatternSpec, ...]:
    """Validate the ``patterns`` section, executing every example."""
    if not isinstance(value, list):
        problems.append("patterns: must be a list")
        return ()
    out = []
    kinds: set[str] = set()
    for index, item in enumerate(value):
        path = f"patterns[{index}]"
        if not isinstance(item, Mapping):
            problems.append(f"{path}: must be a mapping")
            continue
        before = len(problems)
        _unknown(item.keys(), _PATTERN_KEYS, path, problems)
        kind = item.get("kind")
        if not isinstance(kind, str) or not _KIND.match(kind):
            problems.append(f"{path}.kind: {kind!r} must match {_KIND.pattern}")
        elif kind in kinds:
            problems.append(f"{path}.kind: {kind!r} is defined twice in this pack")
        source = item.get("pattern")
        if not isinstance(source, str) or not source:
            problems.append(f"{path}.pattern: must be a non-empty string")
        elif len(source) > _MAX_PATTERN:
            problems.append(f"{path}.pattern: longer than {_MAX_PATTERN} characters")
        intent = item.get("intent")
        if not isinstance(intent, str) or not intent.strip():
            problems.append(f"{path}.intent: say what this pattern is for")
        priority = item.get("priority", 50)
        if (
            isinstance(priority, bool)
            or not isinstance(priority, int)
            or not 0 <= priority <= 100  # ruff: ignore[magic-value-comparison]
        ):  # ruff: ignore[magic-value-comparison]
            problems.append(f"{path}.priority: must be an integer in 0..100")
        confidence = item.get("confidence", 1.0)
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= confidence <= 1.0
        ):
            problems.append(f"{path}.confidence: must be a number in 0..1")
        flags = 0
        for flag in item.get("flags", []) or []:
            if flag not in _FLAGS:
                problems.append(
                    f"{path}.flags: {flag!r} is not one of {', '.join(sorted(_FLAGS))}"
                )
            else:
                flags |= _FLAGS[flag]
        validator = None
        try:
            validator = get_validator(item.get("validate"))
        except KeyError:
            problems.append(
                f"{path}.validate: {item.get('validate')!r} is not one of "
                f"{', '.join(validator_names())}"
            )
        yes = _example_list(item.get("examples_yes"), f"{path}.examples_yes", problems)
        raw_no = item.get("examples_no", [])
        no = (
            ()
            if raw_no in (None, [])
            else _example_list(raw_no, f"{path}.examples_no", problems)
        )
        if isinstance(source, str) and source and len(source) <= _MAX_PATTERN:
            try:
                re.compile(source, flags)
            # A count such as {99999999999999999999} raises OverflowError, not
            # re.error; it is a problem in the pack like any other (CP-107).
            except (re.error, OverflowError) as exc:
                problems.append(f"{path}.pattern: does not compile: {exc}")
        risk_reason = _risk_acceptance(item, path, problems)
        if len(problems) != before:
            continue
        spec = PatternSpec(
            kind=kind,
            pattern=source,
            intent=intent.strip(),
            priority=priority,
            flags=flags,
            validate=validator,
            confidence=float(confidence),
            examples_yes=yes,
            examples_no=no,
            risk_reason=risk_reason,
        )
        _check_examples(spec, path, problems)
        if len(problems) == before:
            kinds.add(kind)
            out.append(spec)
    return tuple(out)


def _parse_code(value: Any, problems: list[str]) -> CodeSpec:
    """Validate the ``code`` section."""
    if not isinstance(value, Mapping):
        problems.append("code: must be a mapping")
        return CodeSpec()
    _unknown(value.keys(), _CODE_KEYS, "code", problems)
    keywords = ()
    methods = ()
    if "column_keywords" in value:
        keywords = _string_list(
            value["column_keywords"], "code.column_keywords", problems, True
        )
    if "column_methods" in value:
        methods = _string_list(
            value["column_methods"], "code.column_methods", problems, True
        )
    roles = []
    raw = value.get("dtype_roles", {}) or {}
    if not isinstance(raw, Mapping):
        problems.append("code.dtype_roles: must be a mapping of dtype to role")
    else:
        for dtype, role in raw.items():
            if role not in ROLES:
                problems.append(f"code.dtype_roles.{dtype}: {role!r} is not a role")
            else:
                roles.append((str(dtype).lower(), role))
    return CodeSpec(
        tuple(sorted(set(keywords))), tuple(sorted(set(methods))), tuple(sorted(roles))
    )


def pack_from_document(document: Any, source: str = "<builtin>") -> PackSpec:
    """
    Validate a parsed pack document and build its spec.

    Parameters
    ----------
    document : mapping
        The pack, as parsed from YAML or JSON.
    source : str, default='<builtin>'
        Where it came from, for messages.

    Returns
    -------
    PackSpec
        The validated pack.

    Raises
    ------
    PackError
        Listing every problem found. Nothing is returned for a partly valid
        document.

    Examples
    --------
    >>> pack = pack_from_document(
    ...     {
    ...         "name": "demo",
    ...         "version": 1,
    ...         "summary": "A demo.",
    ...         "fields": [{"names": ["DateOfBirth"], "kind": "DOB", "role": "date"}],
    ...     }
    ... )
    >>> pack.field_index()["date_of_birth"].kind
    'DOB'
    >>> pack_from_document({"name": "Bad Name", "version": 1, "summary": "x"})
    Traceback (most recent call last):
    ...
    scikitplot.cleanprompt._packs.PackError: <builtin>: 1 problem(s) in definition
      - name: 'Bad Name' must match ^[a-z][a-z0-9_]{1,31}$
    """
    problems: list[str] = []
    if not isinstance(document, Mapping):
        raise PackError(source, ["the document must be a mapping"])
    _unknown(document.keys(), _TOP_KEYS, "pack", problems)

    name = document.get("name")
    if not isinstance(name, str) or not _PACK_NAME.match(name):
        problems.append(f"name: {name!r} must match {_PACK_NAME.pattern}")
    version = document.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        problems.append(f"version: {version!r} must be a positive integer")
    summary = document.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        problems.append("summary: must be a non-empty string")
    requires = ()
    if document.get("requires"):
        requires = _string_list(document["requires"], "requires", problems)
        for other in requires:
            if not _PACK_NAME.match(other):
                problems.append(f"requires: {other!r} is not a pack name")
            if other == name:
                problems.append("requires: a pack cannot require itself")

    fields = _parse_fields(_section(document, "fields", [], problems), problems)
    patterns = _parse_patterns(_section(document, "patterns", [], problems), problems)
    code = _parse_code(_section(document, "code", {}, problems), problems)

    if not (fields or patterns or code != CodeSpec()) and not problems:
        problems.append("the pack defines no fields, patterns or code sites")

    if problems:
        raise PackError(source, problems)
    return PackSpec(
        name=name,
        version=version,
        summary=summary.strip(),
        requires=tuple(sorted(set(requires))),
        fields=fields,
        patterns=patterns,
        code=code,
        source=source,
    )


class PackPatternDetector(RegexDetector):
    """
    A pattern detector that carries the name of the pack it came from.

    Parameters
    ----------
    spec : PatternSpec
        The validated pattern.
    pack : str
        The pack's name.

    Notes
    -----
    **Developer notes.** :class:`RegexDetector` names itself ``regex:KIND``,
    and a registry refuses two detectors with one name. A pack is free to
    define a kind the core patterns also define — a stricter ``EMAIL`` for an
    internal domain, say — so pack detectors are named ``pack:NAME:KIND``,
    which is unique, stable across runs, and tells a report where a span came
    from.
    """

    def __init__(self, spec: PatternSpec, pack: str) -> None:
        super().__init__(spec)
        self.name = f"pack:{pack}:{spec.kind}"

    def _span(self, match: re.Match, start: int, end: int) -> Span:
        """
        Build the span, narrowed to the ``value`` group when the pattern has one.

        Notes
        -----
        **Developer notes.** The whole match is still what is found and
        validated; only what is *replaced* narrows. An empty or absent group
        falls back to the whole match, so narrowing can never hide less than
        the match would have minus its label.
        """
        if VALUE_GROUP in match.re.groupindex:
            low, high = match.span(VALUE_GROUP)
            if 0 <= low < high:
                return Span(
                    start=low,
                    end=high,
                    kind=self.kind,
                    text=match.group(VALUE_GROUP),
                    detector=self.name,
                    priority=self.priority,
                    confidence=self.confidence,
                )
        return super()._span(match, start, end)


def pack_detectors(packs: Iterable[PackSpec]) -> list[RegexDetector]:
    """
    Build one detector per distinct pattern across the given packs.

    Parameters
    ----------
    packs : iterable of PackSpec
        Already resolved and ordered.

    Returns
    -------
    list of RegexDetector
        Deduplicated by ``(kind, pattern, flags)``, in pack then pattern order.

    Notes
    -----
    **Developer notes.** Two packs may legitimately carry the same pattern —
    ``personal`` and ``patient`` both want a date of birth. Running it twice
    would double every span and then leave arbitration to discard one, which
    costs time and says nothing. Deduplication happens here, once.
    """
    seen: set[tuple[str, str, int]] = set()
    detectors = []
    for pack in packs:
        for spec in pack.patterns:
            key = (spec.kind, spec.pattern, spec.flags)
            if key in seen:
                continue
            seen.add(key)
            detectors.append(PackPatternDetector(spec, pack.name))
    return detectors
