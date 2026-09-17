# scikitplot/corpus/_plan.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Composable, order-independent configuration: :class:`CorpusPlan` and the fluent facade.

Two ways to configure a corpus, both compiling to the *same* canonical plan::

    # nested / explicit
    corpus = Corpus(plan=CorpusPlan.of(reader=..., embedder=..., storage=...))

    # fluent / ergonomic
    corpus = Corpus().reader(...).embedder(...).storage(...)

Notes
-----
**User-focused.**  Independent fragments commute, so these are identical::

    Corpus().embedder(A).storage(B)
    Corpus().storage(B).embedder(A)

Partial configuration is valid.  Configuring the *same* domain twice is an
error, because silently keeping the last one would discard the first without
saying so::

    Corpus().embedder(A).embedder(B)  # ValueError
    Corpus().embedder(A).replace_embedder(B)  # explicit, allowed
    Corpus().embedder(A).embedder(B, conflict="replace")

**Developer-focused.**  Three rules make this a *view* over configuration rather
than a second pipeline.

*Fluent call order never defines execution order.*  ``.reader()`` and
``.embedder()`` say **which** component, not **when** it runs.  The pipeline
order (read, normalize, chunk, enrich, embed, store, retrieve) is architectural.
If chained calls implied stage order, then ``.chunker().normalizer()`` would
silently build a different pipeline from ``.normalizer().chunker()`` -- a second
execution engine defined by call order.  Sequential composition therefore has
its own explicit API, :meth:`FluentCorpus.stages`.

*Conflict is an error by default.*  Last-call-wins would be another instance of
the shape this codebase has been bitten by three times already (F-R07-01,
F-R08-02, F-R09-01): an operation completing, producing plausible output, and
not reporting that part of the input was discarded.

*Nothing is constructed at configuration time.*  No network access, no model
loading, no backend initialisation happens because ``.embedder()`` was called.
A plan is data; :meth:`FluentCorpus.build` is where anything happens.

See Also
--------
scikitplot.corpus._diagnostics.ErrorRecord : what validation returns.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from ._canonical import canonical_digest

if TYPE_CHECKING:
    from ._registry import ComponentRegistry
    from ._runtime import RuntimeCorpus, RuntimePolicy

from ._diagnostics import ErrorCategory, ErrorRecord

__all__: list[str] = [
    "CONFIG_DOMAINS",
    "ConfigConflictError",
    "CorpusPlan",
    "FluentCorpus",
]

#: Configuration domains, one per component category.  All are **declarative**:
#: setting one says which component to use, never when it runs.
CONFIG_DOMAINS = (
    "source",
    "reader",
    "normalizer",
    "chunker",
    "enricher",
    "embedder",
    "storage",
    "index",
    "retrieval",
    "export",
)

#: Domains whose configuration decides *what gets built*. A change to any of
#: them invalidates a built index, so an artifact binds this scope.
INGEST_DOMAINS = (
    "source",
    "reader",
    "normalizer",
    "chunker",
    "enricher",
    "embedder",
    "storage",
    "index",
)

#: Domains whose configuration decides *how a build is queried*. A change here
#: cannot invalidate an artifact, which is the whole point of the split: one
#: fingerprint over every domain meant a retrieval budget change was
#: indistinguishable from a change that required a rebuild.
QUERY_DOMAINS = (
    "retrieval",
    "export",
)

#: Canonical pipeline stage order.  Architectural, and deliberately *not*
#: derived from the order in which configuration methods were called.
DEFAULT_STAGES = (
    "read",
    "normalize",
    "chunk",
    "enrich",
    "embed",
    "store",
    "retrieve",
)

_FIELD_SEP = "\x1f"
_PLAN_SCHEMA = "plan2"  # encoding changed in S-6; see _utils._canonical


class ConfigConflictError(ValueError):
    """Raised when one domain is configured twice without an explicit intent.

    Notes
    -----
    A subclass of :exc:`ValueError` so existing handlers still catch it, while
    callers can react specifically to a configuration conflict.
    """


def _describe(value: Any) -> str:
    """Return a stable textual identity for a config fragment."""
    for attr in ("fingerprint", "plan_id"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, str):
            return candidate
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        inner = ",".join(f"{f.name}={getattr(value, f.name)!r}" for f in fields)
        return f"{type(value).__name__}({inner})"
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return f"{type(value).__name__}({value!r})"


@dataclasses.dataclass(frozen=True)
class CorpusPlan:
    """An immutable, canonical description of a corpus configuration.

    Parameters
    ----------
    fragments : dict
        Mapping of domain name to configuration fragment.
    stages : tuple of str or None, optional
        Explicit pipeline stage order.  ``None`` means :data:`DEFAULT_STAGES`.

    Notes
    -----
    **Developer.**  ``fragments`` is stored sorted by domain, so two plans built
    by different call orders are *equal* rather than merely equivalent.  That is
    what makes order-independence checkable rather than aspirational.
    """

    fragments: dict[str, Any] = dataclasses.field(default_factory=dict)
    stages: tuple[str, ...] | None = None

    @classmethod
    def of(cls, **fragments: Any) -> CorpusPlan:
        """Build a plan from keyword fragments.

        Raises
        ------
        ValueError
            If a keyword is not a known configuration domain.
        """
        unknown = sorted(set(fragments) - set(CONFIG_DOMAINS))
        if unknown:
            raise ValueError(
                f"unknown configuration domain(s) {unknown}; "
                f"expected one of {list(CONFIG_DOMAINS)}"
            )
        return cls(fragments={k: v for k, v in fragments.items() if v is not None})

    @property
    def configured(self) -> list[str]:
        """Domains that have been configured, in canonical order."""
        return [d for d in CONFIG_DOMAINS if d in self.fragments]

    @property
    def effective_stages(self) -> tuple[str, ...]:
        """Stage order this plan will execute in."""
        return self.stages if self.stages is not None else DEFAULT_STAGES

    @property
    def fingerprint(self) -> str:
        """Stable content-derived identifier for this configuration.

        Notes
        -----
        **Developer.**  Two plans built by different call orders share a
        fingerprint; two plans differing in any fragment do not.  A config change
        that moves this value is exactly a change that should invalidate a built
        index.

        The derivation is :func:`scikitplot._utils._canonical.canonical_digest`,
        the one encoding every identity in this package uses, so a plan, an
        embedding generation and a build generation cannot disagree about what
        two equal configurations are. It replaced a ``repr``-based derivation
        that was not stable across interpreters: a fragment holding a plain
        object or a set of strings fingerprinted differently in a second
        process, which makes a persisted cache key unsound.

        The full SHA-256 is returned. Use :attr:`short_fingerprint` for display.

        Raises
        ------
        scikitplot._utils._canonical.CanonicalError
            If a fragment holds a value with no deterministic encoding. The
            message names the fragment and the field.
        """
        return canonical_digest(
            {
                "schema": _PLAN_SCHEMA,
                "fragments": {d: self.fragments[d] for d in self.configured},
                "stages": list(self.effective_stages),
            }
        )

    def _scope_digest(self, domains: tuple[str, ...], label: str) -> str:
        """Return the canonical digest of one identity scope.

        Parameters
        ----------
        domains : tuple of str
            Domains belonging to the scope, in canonical order.
        label : str
            Scope name, mixed into the digest so one scope's identity can never
            equal another's for the same fragments.

        Returns
        -------
        str
            Full SHA-256 hex digest.
        """
        configured = tuple(d for d in domains if d in self.fragments)
        payload = {
            "schema": _PLAN_SCHEMA,
            "scope": label,
            "fragments": {d: self.fragments[d] for d in configured},
        }
        if label == "ingest":
            payload["stages"] = list(self.effective_stages)
        return canonical_digest(payload)

    @property
    def ingest_fingerprint(self) -> str:
        """Identity of everything that decides what this plan builds.

        Notes
        -----
        **User.** This is the value an artifact binds. If it is unchanged, a
        previously built index is still valid for this plan, however the
        retrieval configuration has moved.

        **Developer.** Covers :data:`INGEST_DOMAINS` and the effective stage
        order, which changes what is produced rather than how it is read.
        """
        return self._scope_digest(INGEST_DOMAINS, "ingest")

    @property
    def query_fingerprint(self) -> str:
        """Identity of everything that decides how a build is queried.

        Notes
        -----
        **Developer.** Covers :data:`QUERY_DOMAINS`. A change here never
        invalidates an artifact; it is separated so that a cache keyed on the
        build cannot be discarded by a query-time edit.
        """
        return self._scope_digest(QUERY_DOMAINS, "query")

    @property
    def short_fingerprint(self) -> str:
        """First 16 characters of :attr:`fingerprint`, for display only.

        Notes
        -----
        **Developer.** Never persist this or use it as a key. The full digest is
        the identity; a prefix narrows the collision bound for no benefit the
        storage layer needs.
        """
        return self.fingerprint[:16]

    def get(self, domain: str) -> Any:
        """Return the fragment for ``domain``, or ``None``."""
        return self.fragments.get(domain)

    def validate(self) -> list[ErrorRecord]:
        """Check cross-fragment coherence.

        Returns
        -------
        list of ErrorRecord
            Empty when the plan is coherent.  Records rather than exceptions, so
            a caller sees every problem in one pass.

        Notes
        -----
        This is *phase 2* validation: fragment-level schema errors are raised at
        fragment construction, and capability errors (is this backend actually
        installed?) are deferred to build time.  Nothing here touches the
        network or loads a model.
        """
        problems: list[ErrorRecord] = []

        # The question is whether the configured index needs vectors, not
        # whether an index is configured at all: a lexical index needs none, and
        # refusing it meant a purely lexical corpus could not be expressed. A
        # fragment that does not say what it needs is treated as needing them,
        # so silence keeps the safe answer.
        index_fragment = self.fragments.get("index")
        needs_vectors = bool(getattr(index_fragment, "requires_vectors", True))
        if (
            "index" in self.fragments
            and needs_vectors
            and "embedder" not in self.fragments
        ):
            problems.append(
                ErrorRecord(
                    code="PLAN_INDEX_WITHOUT_EMBEDDER",
                    category=ErrorCategory.VALIDATION,
                    message=(
                        "a vector index is configured but no embedder is; the "
                        "index would have no vectors to build from. If this "
                        "index is lexical, declare requires_vectors=False on "
                        "the fragment."
                    ),
                    stage="plan",
                )
            )

        unknown_stages = [s for s in self.effective_stages if s not in DEFAULT_STAGES]
        if unknown_stages:
            problems.append(
                ErrorRecord(
                    code="PLAN_UNKNOWN_STAGE",
                    category=ErrorCategory.VALIDATION,
                    message=f"unknown pipeline stage(s): {unknown_stages}",
                    stage="plan",
                    details={"known": list(DEFAULT_STAGES)},
                )
            )

        duplicates = [
            s
            for s in set(self.effective_stages)
            if list(self.effective_stages).count(s) > 1
        ]
        if duplicates:
            problems.append(
                ErrorRecord(
                    code="PLAN_DUPLICATE_STAGE",
                    category=ErrorCategory.VALIDATION,
                    message=f"stage(s) listed more than once: {sorted(duplicates)}",
                    stage="plan",
                )
            )

        return problems

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible description of the plan."""
        return {
            "fingerprint": self.fingerprint,
            "configured": self.configured,
            "stages": list(self.effective_stages),
            "fragments": {d: _describe(self.fragments[d]) for d in self.configured},
        }

    def __eq__(self, other: object) -> bool:
        """Plans are equal when they describe the same configuration."""
        if not isinstance(other, CorpusPlan):
            return NotImplemented
        return self.fingerprint == other.fingerprint

    def __hash__(self) -> int:
        """Hash by fingerprint, consistent with :meth:`__eq__`."""
        return hash(self.fingerprint)


class FluentCorpus:
    """Immutable fluent builder over a :class:`CorpusPlan`.

    Notes
    -----
    Every method returns a **new** instance, so a partially-configured builder is
    safely reusable as a template::

        base = FluentCorpus().reader(r)
        strict, lenient = base.storage(a), base.storage(b)

    Examples
    --------
    >>> a = FluentCorpus().embedder("E").storage("S")
    >>> b = FluentCorpus().storage("S").embedder("E")
    >>> a.plan() == b.plan()
    True
    """

    __slots__ = ("_plan",)

    def __init__(self, plan: CorpusPlan | None = None) -> None:
        self._plan = plan if plan is not None else CorpusPlan()

    # -- construction --------------------------------------------------------

    def _with(self, domain: str, value: Any, conflict: str) -> FluentCorpus:
        """Return a new builder with ``domain`` set to ``value``."""
        if domain not in CONFIG_DOMAINS:
            raise ValueError(
                f"unknown configuration domain {domain!r}; "
                f"expected one of {list(CONFIG_DOMAINS)}"
            )
        if conflict not in ("error", "replace"):
            raise ValueError(f"conflict must be 'error' or 'replace', got {conflict!r}")
        if domain in self._plan.fragments and conflict == "error":
            existing = _describe(self._plan.fragments[domain])
            raise ConfigConflictError(
                f"{domain!r} is already configured as {existing}; refusing to "
                f"replace it with {_describe(value)} silently. Use "
                f".replace_{domain}(...) or conflict='replace' to substitute it "
                "deliberately."
            )
        fragments = dict(self._plan.fragments)
        fragments[domain] = value
        return FluentCorpus(CorpusPlan(fragments=fragments, stages=self._plan.stages))

    def config(
        self, domain: str, value: Any, *, conflict: str = "error"
    ) -> FluentCorpus:
        """Set one configuration domain by name."""
        return self._with(domain, value, conflict)

    def stages(self, *stages: str) -> FluentCorpus:
        """Set an explicit pipeline stage order.

        Notes
        -----
        **Developer.**  This is the *only* way to express execution order.
        Chained configuration calls deliberately do not, because a chain whose
        order silently changed the pipeline would be a second execution engine
        defined by call order.
        """
        return FluentCorpus(
            CorpusPlan(fragments=dict(self._plan.fragments), stages=tuple(stages))
        )

    # -- inspection ----------------------------------------------------------

    def plan(self) -> CorpusPlan:
        """Return the canonical plan this builder describes."""
        return self._plan

    def validate(self) -> list[ErrorRecord]:
        """Return cross-fragment problems, empty when coherent."""
        return self._plan.validate()

    def build(self) -> CorpusPlan:
        """Validate and return the plan, raising on any problem.

        Raises
        ------
        ValueError
            Naming every validation problem found.

        Notes
        -----
        **Developer.**  ``build()`` remains the validated-plan boundary for
        backward compatibility.  Runtime component construction is explicit via
        :meth:`materialize`, so a caller can still inspect, compare and serialise
        a plan without constructing operational state.
        """
        problems = self.validate()
        if problems:
            joined = "; ".join(str(p) for p in problems)
            raise ValueError(f"invalid corpus plan: {joined}")
        return self._plan

    def materialize(
        self,
        *,
        policy: RuntimePolicy | None = None,
        registry: ComponentRegistry | None = None,
    ) -> RuntimeCorpus:
        """Validate this plan and construct a :class:`RuntimeCorpus`.

        Unlike the fluent setters, materialization is an operational boundary:
        it may construct storage/index/component objects, but it does not read
        the configured source.  Source processing starts only when
        :meth:`RuntimeCorpus.run` or :meth:`RuntimeCorpus.add` is called.
        """
        from ._runtime import materialize_plan  # noqa: PLC0415

        return materialize_plan(self.build(), policy=policy, registry=registry)

    def __repr__(self) -> str:
        """Return a readable summary."""
        configured = ", ".join(self._plan.configured) or "-"
        return f"<FluentCorpus configured=[{configured}] {self._plan.fingerprint}>"


def _make_setter(domain: str):
    """Build the declarative setter for one domain."""

    def setter(self, value: Any, *, conflict: str = "error") -> FluentCorpus:
        return self._with(domain, value, conflict)

    setter.__name__ = domain
    setter.__qualname__ = f"FluentCorpus.{domain}"
    setter.__doc__ = (
        f"Configure the {domain}.\n\n"
        "        Parameters\n"
        "        ----------\n"
        "        value : Any\n"
        f"            Configuration fragment or component for the {domain} domain.\n"
        "        conflict : {{'error', 'replace'}}, optional\n"
        f"            What to do if the {domain} is already configured. Default\n"
        "            ``'error'``: silently discarding the previous value would be\n"
        "            a configuration that looks applied but was overwritten.\n\n"
        "        Returns\n"
        "        -------\n"
        "        FluentCorpus\n"
        "            A new builder; the original is unchanged.\n\n"
        "        Raises\n"
        "        ------\n"
        "        ConfigConflictError\n"
        f"            If the {domain} is already configured and ``conflict`` is\n"
        "            ``'error'``.\n"
    )
    return setter


def _make_replacer(domain: str):
    """Build the explicit replacement setter for one domain."""

    def replacer(self, value: Any) -> FluentCorpus:
        return self._with(domain, value, "replace")

    replacer.__name__ = f"replace_{domain}"
    replacer.__qualname__ = f"FluentCorpus.replace_{domain}"
    replacer.__doc__ = (
        f"Replace the {domain}, whether or not one is already configured.\n\n"
        "        The explicit counterpart to the error-by-default rule: the\n"
        "        intent to substitute is stated in the method name.\n\n"
        "        Returns\n"
        "        -------\n"
        "        FluentCorpus\n"
    )
    return replacer


for _domain in CONFIG_DOMAINS:
    setattr(FluentCorpus, _domain, _make_setter(_domain))
    setattr(FluentCorpus, f"replace_{_domain}", _make_replacer(_domain))
del _domain
