"""
Redact a whole data-science artefact: a notebook, a module, a script.

Notes
-----
**User notes.** One call, or one command::

    cleanprompt encode --in analysis.ipynb --out analysis.clean.ipynb

What comes back is the same kind of file with the schema renamed, the paths
replaced, the rendered data removed and the figures dropped — and the code
still readable, so a model can actually help with it:

.. code-block:: python

    # before
    df = pd.read_parquet("/mnt/prod/exports/2026_q1_acme_pii.parquet")
    df = df[["customer_ssn", "acct_balance_usd", "region_code"]]

    # after
    df = pd.read_parquet("/data/dataset_1.parquet")
    df = df[["id_1", "amount_1", "category_1"]]

Paste that into any chat, and ``cleanprompt decode`` turns the answer — code
included, even code the model rewrote — back into your names.

**Developer notes — this module orchestrates, it does not redact.**

Nothing here re-implements detection, arbitration, rewriting or restoration.
The artefact layer is three decisions laid on top of the existing engine:

*Which parts to look at.* :mod:`~scikitplot.cleanprompt._documents` splits the
raw file into regions with roles, and the detectors below are built to respect
them. A base64 figure is redacted **whole**, so nothing inside it is ever
treated as prose.

Rendered outputs are redacted whole too, and that default deserves its
reasoning stated. An output is not a description of the data; it *is* the
data — ``df.head()`` is two hundred real rows, ``value_counts()`` is the real
segment names with their real sizes, and a count of one identifies a person.
Trying to redact *inside* a rendered table means parsing a repr that changes
between pandas versions, and every version it fails to parse is a silent leak.
Removing the region is decidable and complete. Tracebacks are deliberately
exempt and are scanned as text instead, because an error message is the single
most common reason to ask a model for help and is mostly structure rather than
data.

*What to look for.* :mod:`~scikitplot.cleanprompt._code` reads the schema out
of the source syntactically, and each discovered column becomes a
:class:`~scikitplot.cleanprompt.LiteralDetector` with identifier boundaries —
which is what makes ``df.acct_balance_usd`` get rewritten even though
attribute access is deliberately not a *discovery* site.

*What to call things instead.* :mod:`~scikitplot.cleanprompt._schema` chooses a
role-preserving identifier, and it is applied by **seeding the vault** with the
labels it produced. That is the whole trick: the engine already supports a seed
so an append-mode vault can keep its numbering, so the artefact layer needs no
change to the engine, no new tag style, and no second code path through which a
value could escape.

**Developer notes — what is deliberately not attempted.**

This layer does not execute the notebook, import pandas, or infer a dtype from
data it has not been shown. It reads what is written. A column whose role
nobody established is renamed to ``field_n`` and *reported* as unestablished,
because a confident wrong label is worse than an honest neutral one.

See Also
--------
scikitplot.cleanprompt._documents : Regions and roles.
scikitplot.cleanprompt._code : Syntactic schema discovery.
scikitplot.cleanprompt._schema : Role-preserving stand-ins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ._code import discover
from ._detectors import (
    DetectorRegistry,
    LiteralDetector,
    RegexDetector,
    default_registry,
)
from ._documents import Region, detect_format, regions_for
from ._engine import Redactor
from ._exceptions import CleanPromptError
from ._patterns import PatternSpec
from ._policy import DEFAULT_POLICY, RedactionPolicy
from ._schema import (
    infer_role,
    is_safe_to_rename,
    path_surrogate,
    role_for,
    schema_surrogate,
)
from ._types import Entry, RedactionResult

__all__ = [
    "PATH_PATTERN",
    "ArtifactPlan",
    "encode_artifact",
    "path_seed",
    "plan_artifact",
]

#: File extensions that make a bare name a dataset or an artefact rather than a
#: sentence with a full stop in it.
_DATA_EXTENSIONS = (
    "parquet|csv|tsv|json|jsonl|ndjson|xlsx|xls|feather|orc|avro|h5|hdf5|pkl|"
    "pickle|joblib|npy|npz|sav|dta|db|sqlite3|sqlite|zip|gz|txt|png|jpg|jpeg|"
    "svg|pdf|ipynb|py|yaml|yml|sql"
)

#: Characters that end a path. A path in source code is nearly always inside a
#: quoted string, so the closing quote, a comma and a bracket all terminate it.
_PATH_BODY = r"[^\s\"'`,;()\[\]{}<>\\|]"

#: A filesystem path, a URI, or a bare data file name.
#:
#: ``http`` and ``https`` are excluded on purpose: the ``URL`` pattern already
#: owns them, and two kinds competing for the same surface makes the report
#: harder to read without making the redaction any safer.
PATH_PATTERN = PatternSpec(
    kind="PATH",
    pattern=(
        r"(?<![\w/.\-])(?:"
        r"(?:(?!https?://)[a-z][a-z0-9+.\-]{1,15}://" + _PATH_BODY + r"+)"
        # A Windows path has one backslash between parts; the previous form
        # required two, so C:\\Users\\<name> went through unhidden (CP-075).
        r"|(?:[A-Za-z]:\\[^\s\"'`,;()\[\]{}<>|]*)"
        # Not the "/name>" of a closing tag such as </document> (CP-074); a
        # shell redirection like "sort </home/ann/in.txt" is still a path.
        r"|(?:(?:(?<!<)|(?!/[A-Za-z_][\w.:\-]*\s*>))~?/[A-Za-z0-9._\-]"
        + _PATH_BODY
        + r"*)"
        r"|(?:(?:[A-Za-z0-9._\-]+/)+[A-Za-z0-9._\-]*\.(?:" + _DATA_EXTENSIONS + r"))"
        r"|(?:[A-Za-z0-9][A-Za-z0-9._\-]*\.(?:" + _DATA_EXTENSIONS + r"))"
        r")\b"
    ),
    intent=(
        "A filesystem path, cloud URI or data file name. It discloses the "
        "organisation, the environment, the period and often the analyst: "
        "/home/marion.holt/work/acme-churn/ names a person and a client "
        "project without containing a single value any other pattern matches."
    ),
    priority=70,
    examples_yes=(
        "/mnt/prod/exports/2026_q1_customers.parquet",
        "s3://bucket-name/models/churn.pkl",
        "~/projects/train.py",
        "reports/figure.png",
        "customers_2026.parquet",
        "C:\\Users\\marion.holt\\Documents\\churn.xlsx",
        "sort </home/marion.holt/in.txt",
    ),
    examples_no=(
        "text/plain",
        "and/or",
        "</document>",
        "<doc>x</doc>",
        "TP/(TP+FN)",
        "50/50",
        "https://example.com/docs",
    ),
)


class _RegionDetector:
    """
    Report whole regions of a chosen role as single spans.

    Notes
    -----
    **Developer notes.** This is how a base64 figure is handled, and framing it
    as a *detector* rather than as a filter is the point. A rendered figure can
    show the data that was just removed from the table above it — axis labels,
    tick values, a legend of real segment names — so dropping it is a redaction
    like any other, and it should be counted, reported and restorable like any
    other.

    Because a region span covers everything inside it, longest-wins arbitration
    makes it beat any pattern that also matched within the payload, so nothing
    inside an image is ever separately labelled.
    """

    __slots__ = ("_regions", "confidence", "kind", "name", "priority")

    def __init__(self, regions: Iterable[Region], roles: Iterable[str], kind: str):
        self.name = f"region:{kind.lower()}"
        self.kind = kind
        self.priority = 95
        self.confidence = 1.0
        wanted = frozenset(roles)
        self._regions = tuple(one for one in regions if one.role in wanted)

    def detect(self, text: str, policy: RedactionPolicy):
        """Yield one span per region of the chosen roles."""
        del policy
        from ._types import Span  # ruff: ignore[import-outside-top-level]

        limit = len(text)
        for region in self._regions:
            if region.end > limit or region.end <= region.start:
                continue
            yield Span(
                start=region.start,
                end=region.end,
                kind=self.kind,
                text=text[region.start : region.end],
                detector=self.name,
                priority=self.priority,
                confidence=1.0,
            )


@dataclass
class ArtifactPlan:
    """
    What will be hidden in an artefact, and how it was decided.

    Parameters
    ----------
    fmt : str
        The detected format.
    regions : tuple of Region
        The document's regions.
    columns : dict
        Column name to ``(role, provenance, stand-in)``.
    refused : dict
        Column name to the reason it cannot be safely renamed.
    suggestions : tuple of str
        Names that look like a column list but were never used as one, offered
        for confirmation rather than acted upon.
    unparsed : tuple of str
        Fragments that did not parse, named so the report can say which cells
        were not read.
    identifiers : set of str
        Every identifier the artefact already uses.

    Notes
    -----
    **User notes.** ``plan_artifact`` gives you this without touching
    anything, which is what ``inspect`` prints. Read ``refused`` and
    ``unparsed`` before trusting the result: they are the two places where a
    column can exist and not be hidden.
    """

    fmt: str = "text"
    regions: tuple[Region, ...] = ()
    columns: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)
    suggestions: tuple[str, ...] = ()
    unparsed: tuple[str, ...] = ()
    identifiers: set[str] = field(default_factory=set)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view, with no original value in it."""
        return {
            "format": self.fmt,
            "regions": {
                role: sum(1 for one in self.regions if one.role == role)
                for role in sorted({one.role for one in self.regions})
            },
            "columns": {
                name: {"role": role, "established": provenance, "renamed_to": standin}
                for name, (role, provenance, standin) in sorted(self.columns.items())
            },
            "refused": dict(sorted(self.refused.items())),
            "suggestions": list(self.suggestions),
            "unparsed": list(self.unparsed),
        }


def plan_artifact(  # ruff: ignore[too-many-positional-arguments]
    text: str,
    fmt: str | None = None,
    path: str | None = None,
    declared_roles: dict[str, str] | None = None,
    extra_columns: Iterable[str] | None = None,
    infer_roles: bool = False,
    vocabulary: object | None = None,
    prior: Iterable[Entry] | None = None,
) -> ArtifactPlan:
    """
    Work out what an artefact's schema is, changing nothing.

    Parameters
    ----------
    text : str
        The artefact, exactly as read.
    fmt : str, optional
        ``'text'``, ``'python'`` or ``'notebook'``. Detected when omitted.
    path : str, optional
        The file name, used for format detection and reporting.
    declared_roles : dict, optional
        Roles the caller states, keyed by column name. Never second-guessed.
    extra_columns : iterable of str, optional
        Column names to hide that discovery could not establish — the answer to
        a ``suggestions`` entry.
    infer_roles : bool, default=False
        Whether to fall back to name-based role suggestion. Off by default;
        when on, the provenance of such a role is reported as ``'inferred'``.
    vocabulary : CodeSpec, optional
        Extra column-naming sites from the selected packs.
    prior : iterable of Entry, optional
        Entries already issued in this conversation or batch. A column seen
        before keeps its stand-in, and a new one never takes a stand-in already
        in use — which is what keeps ``amount_1`` meaning one column across a
        notebook and the module it imports.

    Returns
    -------
    ArtifactPlan
        The schema map and the two lists that qualify it.

    Raises
    ------
    CleanPromptError
        If the artefact does not parse as the format it claims to be.

    Notes
    -----
    **Developer notes.** Stand-ins are issued in sorted column order rather
    than in discovery order. Discovery order depends on how the notebook
    happens to be arranged, and a label that moves when a user reorders two
    cells would break the one property restoration depends on — that the same
    artefact redacts to the same text every time (``I3``).

    Examples
    --------
    >>> plan = plan_artifact("df = df[['region_code']]", "python")
    >>> plan.columns["region_code"][2]
    'field_1'
    >>> plan_artifact("df[['region_code']]", "python", infer_roles=True).columns
    {'region_code': ('category', 'inferred', 'category_1')}
    """
    resolved = fmt or detect_format(path, text)
    regions = regions_for(text, resolved, path)

    sources = _code_fragments(text, regions, resolved)

    found = discover(sources, vocabulary)
    names = set(found.columns) | {str(one) for one in extra_columns or ()}

    earlier = {entry.original: entry for entry in prior or () if entry.kind == "COLUMN"}
    columns: dict[str, tuple[str, str, str]] = {}
    refused: dict[str, str] = {}
    issued: set[str] = set(found.identifiers) | {entry.label for entry in prior or ()}
    per_role: dict[str, int] = {}

    for name in sorted(names):
        safe, why = is_safe_to_rename(name)
        if not safe:
            refused[name] = why
            continue
        role, provenance = role_for(name, declared_roles, found.observed_roles)
        if name in earlier:
            columns[name] = (role, provenance, earlier[name].label)
            continue
        if provenance == "none" and infer_roles:
            inferred = infer_role(name)
            if inferred is not None:
                role, provenance = inferred, "inferred"
        per_role[role] = per_role.get(role, 0) + 1
        standin = schema_surrogate(role, per_role[role], avoid=issued)
        issued.add(standin)
        columns[name] = (role, provenance, standin)

    # A binding is worth suggesting when it holds names that were *not*
    # otherwise established. The first version required the list to be wholly
    # uncovered, which silenced exactly the case that matters: NUMERIC = [a,
    # b, c] where only `a` is used as a selector, so `b` and `c` are hidden
    # from both the redaction and the report.
    suggestions = tuple(
        sorted(
            "{} ({})".format(name, ", ".join(sorted(set(values) - names)))
            for name, values in found.bindings.items()
            if len(values) > 1 and set(values) - names
        )
    )

    return ArtifactPlan(
        fmt=resolved,
        regions=regions,
        columns=columns,
        refused=refused,
        suggestions=suggestions,
        unparsed=found.unparsed,
        identifiers=set(found.identifiers),
    )


def _code_fragments(
    text: str, regions: Iterable[Region], fmt: str
) -> list[tuple[str, str]]:
    """
    Group code regions into the units that can actually be parsed.

    Parameters
    ----------
    text : str
        The original document.
    regions : iterable of Region
        The document's regions.
    fmt : str
        The artefact format.

    Returns
    -------
    list of (str, str)
        ``(label, source)`` pairs ready for
        :func:`~scikitplot.cleanprompt._code.discover`.

    Notes
    -----
    **Developer notes — the parse unit is a cell, not a line.**

    A notebook stores a cell's source as an *array of lines*, so the region
    splitter produces one region per line — which is right for rewriting,
    because each line is a separate JSON string with its own offsets, and
    wrong for parsing, because a line is rarely a complete statement. A
    ``df[[...]]`` selection written across two lines gives two fragments,
    neither of which is valid Python, and the columns inside it are never
    discovered.

    That is not a hypothetical: it is what the first version did, and the
    symptom was mild enough to miss — three of six columns hidden, and a
    truthful ``NOT read`` line in the report that looked like a notebook
    quirk rather than a defect.

    Regions are therefore concatenated back into their cell before parsing,
    while every offset used for rewriting stays per-line and untouched.
    """
    if fmt != "notebook":
        return [
            (region.path or "fragment", region.slice(text))
            for region in regions
            if region.role == "code"
        ]

    cells: dict[str, list[str]] = {}
    order: list[str] = []
    for region in regions:
        if region.role != "code":
            continue
        cell = region.path.split(".source", 1)[0] or "fragment"
        if cell not in cells:
            cells[cell] = []
            order.append(cell)
        # JSON escapes survive into a region's raw text; the parser needs the
        # source as Python, so each line is unescaped before it is joined.
        cells[cell].append(_unescape(region.slice(text)))
    return [(cell, "".join(cells[cell])) for cell in order]


def _unescape(body: str) -> str:
    r"""
    Return a JSON string body as the text it encodes.

    Notes
    -----
    **Developer notes.** A region of a notebook indexes the *raw* JSON, so its
    text still carries ``\n`` as two characters. Detection does not care —
    a column name contains no escapes — but :mod:`ast` does, so the code walk
    gets the decoded form while every offset keeps indexing the original.
    """
    import json  # ruff: ignore[import-outside-top-level]

    try:
        return json.loads(f'"{body}"')
    except ValueError:
        return body


def build_registry(
    plan: ArtifactPlan,
    base_kinds: Iterable[str] | None = None,
    hide_paths: bool = True,
    drop_binary: bool = True,
    drop_outputs: bool = True,
) -> DetectorRegistry:
    """
    Build the detector set an artefact needs.

    Parameters
    ----------
    plan : ArtifactPlan
        The plan from :func:`plan_artifact`.
    base_kinds : iterable of str, optional
        Structural pattern kinds to keep enabled. Defaults to all of them;
        an empty iterable enables none.
    hide_paths : bool, default=True
        Whether to hide filesystem paths and cloud URIs.
    drop_binary : bool, default=True
        Whether to redact base64 figure payloads whole.
    drop_outputs : bool, default=True
        Whether to redact rendered outputs whole. Tracebacks are **not**
        outputs for this purpose and are always scanned instead.

    Returns
    -------
    DetectorRegistry
        Ready to pass to :class:`~scikitplot.cleanprompt.Redactor`.

    Notes
    -----
    **Developer notes.** Column detectors use ``word_boundary=True``, which is
    what makes ``df.acct_balance_usd`` and ``X['acct_balance_usd']`` both get
    rewritten while ``acct_balance_usd_raw`` is left alone. It is also what
    keeps ``CP-001`` closed on this side: a column that is a prefix of another
    cannot claim part of it.
    """
    registry = default_registry(kinds=base_kinds)
    if plan.columns:
        registry.add(
            LiteralDetector(
                terms=sorted(plan.columns),
                kind="COLUMN",
                name="schema:columns",
                priority=80,
                word_boundary=True,
            )
        )
    if hide_paths:
        registry.add(RegexDetector(PATH_PATTERN))
    if drop_binary:
        registry.add(_RegionDetector(plan.regions, ("binary",), "FIGURE"))
    if drop_outputs:
        registry.add(_RegionDetector(plan.regions, ("output",), "OUTPUT"))
    return registry


def encode_artifact(  # ruff: ignore[too-many-positional-arguments]
    text: str,
    fmt: str | None = None,
    path: str | None = None,
    policy: RedactionPolicy | None = None,
    declared_roles: dict[str, str] | None = None,
    extra_columns: Iterable[str] | None = None,
    infer_roles: bool = False,
    hide_paths: bool = True,
    drop_binary: bool = True,
    drop_outputs: bool = True,
    extra_terms: Iterable[str] | None = None,
    extra_kind: str = "CUSTOM",
    word_boundary: bool = False,
    extra_detectors: Iterable[Any] | None = None,
    prior: Iterable[Entry] | None = None,
    vocabulary: object | None = None,
    core: bool = True,
) -> tuple[RedactionResult, ArtifactPlan]:
    """
    Redact a whole artefact, keeping it the kind of file it was.

    Parameters
    ----------
    text : str
        The artefact.
    fmt, path, declared_roles, extra_columns, infer_roles
        As for :func:`plan_artifact`.
    policy : RedactionPolicy, optional
        Defaults to :data:`~scikitplot.cleanprompt.DEFAULT_POLICY` widened to
        admit the artefact kinds.
    hide_paths, drop_binary, drop_outputs : bool
        As for :func:`build_registry`.
    extra_terms : iterable of str, optional
        Literal strings to hide as well — what ``--hide`` supplies. Threaded
        through because an artefact still contains the things prose does: an
        organisation nobody's pattern recognises, a project codename, a client.
    extra_kind : str, default='CUSTOM'
        The kind those literal terms are recorded under.
    word_boundary : bool, default=False
        Whether ``extra_terms`` match only at word boundaries.
    extra_detectors : iterable of Detector, optional
        Added to the registry — the selected packs' patterns and field rules.
    prior : iterable of Entry, optional
        Entries issued earlier in the same conversation or batch. They seed
        the vault, so a value keeps its stand-in across files.
    vocabulary : CodeSpec, optional
        Extra column-naming sites from the selected packs.
    core : bool, default=True
        Whether the built-in structural patterns run. ``False`` leaves only the
        schema, path, figure and output detectors plus ``extra_detectors`` —
        what a plan with ``.core(False)`` asks for.

    Returns
    -------
    tuple of (RedactionResult, ArtifactPlan)
        The result, and the plan that produced it, so a caller can report
        refusals and unread cells alongside what was hidden.

    Raises
    ------
    CleanPromptError
        If the artefact does not parse as its format.

    Notes
    -----
    **User notes.** The output is still a notebook, or still a module. Write it
    beside the original and send that.

    **Developer notes.** The vault is *seeded* with the plan's stand-ins, so
    the engine issues them instead of bracket labels without knowing anything
    about schemas. Restoration is unchanged and needs no special case, which is
    why a model's rewritten code restores as readily as the prompt does.

    Examples
    --------
    >>> result, plan = encode_artifact(
    ...     "df = df[['region_code']]", "python", infer_roles=True
    ... )
    >>> result.text
    "df = df[['category_1']]"
    >>> from scikitplot.cleanprompt import restore
    >>> restore(result.text, result.vault).text
    "df = df[['region_code']]"
    """
    earlier = tuple(prior or ())
    plan = plan_artifact(
        text,
        fmt=fmt,
        path=path,
        declared_roles=declared_roles,
        extra_columns=extra_columns,
        infer_roles=infer_roles,
        vocabulary=vocabulary,
        prior=earlier,
    )
    active = policy or DEFAULT_POLICY
    kinds = active.kinds
    registry = build_registry(
        plan,
        base_kinds=kinds if core else (),
        hide_paths=hide_paths,
        drop_binary=drop_binary,
        drop_outputs=drop_outputs,
    )
    for detector in extra_detectors or ():
        registry.add(detector)
    if kinds is not None:
        # The policy admits exactly the kinds this registry was built to run.
        # Widening by a fixed list instead (COLUMN, PATH, ...) admitted a kind
        # with no detector when the artefact had no columns, and the engine
        # refused it; and it left out every extra detector's kind, which the
        # engine then selected away without a word (CP-052).
        active = active.evolve(kinds=registry.kinds())

    seed = list(earlier)
    known = {entry.original for entry in earlier}
    column_base = max((e.ordinal for e in earlier if e.kind == "COLUMN"), default=0)
    for ordinal, (name, (_role, _provenance, standin)) in enumerate(
        sorted(plan.columns.items()), start=column_base + 1
    ):
        if name in known:
            continue
        seed.append(
            Entry(
                label=standin,
                kind="COLUMN",
                ordinal=ordinal,
                original=name,
                occurrences=0,
                detector="schema:columns",
                confidence=1.0,
            )
        )
    seed.extend(path_seed(text, seed) if hide_paths else ())
    result = Redactor(policy=active, registry=registry).redact(
        text,
        extra_terms=tuple(extra_terms) if extra_terms else None,
        extra_kind=extra_kind,
        word_boundary=word_boundary,
        seed=seed,
    )
    return result, plan


def path_seed(text: str, prior: Iterable[Entry] = ()) -> list[Entry]:
    """
    Return seed entries giving each new path in a text its shape-keeping stand-in.

    Parameters
    ----------
    text : str
        The document.
    prior : iterable of Entry
        Entries already issued; a path among them keeps its stand-in, and a new
        stand-in never repeats one of their labels.

    Returns
    -------
    list of Entry
        Only the paths not already in ``prior``, in first-appearance order.

    Notes
    -----
    **Developer notes.** Shared by every format rather than kept inside the
    notebook path, because a shell script and a traceback leak a home
    directory exactly as a notebook does.
    """
    earlier = list(prior)
    known = {entry.original for entry in earlier}
    labels = {entry.label for entry in earlier}
    ordinal = max((e.ordinal for e in earlier if e.kind == "PATH"), default=0)
    out: list[Entry] = []
    for original in _paths_in(text, None):
        if original in known:
            continue
        ordinal += 1
        label = path_surrogate(original, ordinal, avoid=labels)
        labels.add(label)
        known.add(original)
        out.append(
            Entry(
                label=label,
                kind="PATH",
                ordinal=ordinal,
                original=original,
                occurrences=0,
                detector="regex:PATH",
                confidence=1.0,
            )
        )
    return out


def _paths_in(text: str, plan: ArtifactPlan | None) -> tuple[str, ...]:
    """
    Return the distinct paths in an artefact, in first-appearance order.

    Notes
    -----
    **Developer notes.** Ordering by first appearance rather than
    alphabetically keeps ``dataset_1`` the one the reader meets first, which is
    the difference between a readable redacted notebook and a puzzle. It is
    still deterministic, because the text is.
    """
    del plan
    import re  # ruff: ignore[import-outside-top-level]

    seen = []
    for match in re.compile(PATH_PATTERN.pattern).finditer(text):
        value = match.group()
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def refuse_unknown_format(fmt: str) -> None:
    """
    Raise for a format this layer cannot handle.

    Parameters
    ----------
    fmt : str
        The requested format.

    Raises
    ------
    CleanPromptError
        Always; this helper exists so the message is written once.
    """
    raise CleanPromptError(
        f"cannot process an artefact of format {fmt!r}; pass --as text to treat "
        "it as plain prose"
    )
