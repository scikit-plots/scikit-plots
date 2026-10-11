"""
Deciding whether a test of an optional tier runs, and saying why not.

Notes
-----
**Developer notes.** "The tier is unavailable" hides the one distinction that
matters. A tier whose distribution is *not installed* is a fair reason to
skip. A tier whose distribution *is installed* and that the library still
refuses (out of the declared range, or failing its probe) is a defect, either
in the environment or in the declared range, and a skip buries it: a suite
then reports success without having run the tests it installed the
dependency for. That happened with ``cryptography`` 50 against a declared
``<50`` (``CP-092``).

So the reason given for a skip always carries the probe's own account, and
``test__capabilities.TestInstalledTiers`` fails when an installed tier is
refused.
"""

from __future__ import annotations

from .. import _capabilities as caps

#: Statuses that mean "nothing is installed", the only fair reason to skip.
NOT_INSTALLED = frozenset({caps.CapabilityStatus.ABSENT})


def available(tier):
    """
    Say whether the tests of ``tier`` can run.

    Parameters
    ----------
    tier : str
        One of the keys of ``_capabilities.TIERS``.

    Returns
    -------
    bool
        ``True`` when the tier's status is ``AVAILABLE``.
    """
    return caps.probe(tier).available


def skip_reason(tier):
    """
    Return the reason to give when the tests of ``tier`` are skipped.

    Parameters
    ----------
    tier : str
        One of the keys of ``_capabilities.TIERS``.

    Returns
    -------
    str
        The tier, its status and the probe's detail, for example
        ``the 'crypto' tier is ABSENT: cryptography is not installed``.
    """
    report = caps.probe(tier)
    return f"the {tier!r} tier is {report.status.value}: {report.detail}"


def installed_but_refused(tier):
    """
    Return the probe's report when ``tier`` is installed and not usable.

    Parameters
    ----------
    tier : str
        One of the keys of ``_capabilities.TIERS``.

    Returns
    -------
    CapabilityReport or None
        The report when the status is neither ``AVAILABLE`` nor ``ABSENT``;
        ``None`` otherwise.
    """
    report = caps.probe(tier)
    if report.available or report.status in NOT_INSTALLED:
        return None
    return report


def engine_ready(engine="spacy", language="en"):
    """
    Say whether an entity engine can actually run here: package, language, data.

    Parameters
    ----------
    engine : str, default='spacy'
        ``"spacy"`` or ``"nltk"``.
    language : str, default='en'
        Language code.

    Returns
    -------
    bool
        :func:`~scikitplot.cleanprompt._engines.engine_readiness` ``.ready``
        with the data checked.

    Notes
    -----
    **Developer notes.** A live-model test gated on the tier alone failed on a
    machine with spaCy installed and no model (six tests, round 25): the
    ``CP-093`` confusion of installed with ready, in the suite. A missing model
    or data package is an absent prerequisite, like an absent package, so it is
    a fair reason to skip — and :func:`engine_skip_reason` says which command
    supplies it. A *broken* engine is not skipped silently: its status is in
    the reason, and ``TestInstalledTiers`` still fails on a refused tier.
    """
    from .._engines import engine_readiness

    return engine_readiness(engine, language, check_assets=True).ready


def engine_skip_reason(engine="spacy", language="en"):
    """Return the reason to give when a live ``engine`` test is skipped."""
    from .._engines import engine_readiness

    result = engine_readiness(engine, language, check_assets=True)
    remedy = f" (fix: {result.remedy})" if result.remedy else ""
    return f"the {engine!r} engine is not ready ({result.status}): {result.reason}{remedy}"
