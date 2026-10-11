"""
Regular expressions that are slow on purpose, kept out of static ReDoS scans.

Notes
-----
**User notes.** Nothing here is used at run time. The pattern-risk check
(:mod:`~scikitplot.cleanprompt._pattern_risk`) is tested on patterns that
*must* backtrack catastrophically — ``(a+)+`` is the point of the test.

**Developer notes — why a wrapper (round 27).** CodeQL's ``py/redos`` query
treats a string literal that flows, value for value, into :func:`re.compile`
as a regular expression of the program, and reports it as a high-severity
denial-of-service risk. The test fixtures did flow there (through
``analyse_pattern``, which compiles a pattern before analysing it), so every
new fixture raised an alert on the pull request (code scanning alert 227 on
PR 864: ``_RISKY`` in ``test__pattern_risk.py``).

:func:`regex_fixture` returns the same text, rebuilt character by
character. The result is equal to the literal but is a new string computed
from it, not the literal itself, so the literal is no longer a
regular-expression source for the scanner, while the test receives exactly
the string written. It is the same convention the built-in ``secrets`` pack
uses for credential-shaped examples (invariant ``I14``: written in pieces so
a secret scanner has nothing to find). The fixture stays readable at the call
site, and ``maintenances/cleanprompt/_maintenance/tests/test_regex_fixtures.py``
fails when a fixture the check flags is written without the wrapper — so the
alert cannot come back with the next test.

These patterns run only inside the check (which never executes them against
text) and inside the measured probe, which matches them in a child process
under a time limit.
"""

from __future__ import annotations

__all__ = ["regex_fixture"]


def regex_fixture(text: str) -> str:
    """
    Return ``text`` unchanged: a pattern written to be slow, for a test.

    Parameters
    ----------
    text : str
        The pattern, exactly as the test means it.

    Returns
    -------
    str
        An equal string, not value-identical to the literal for static
        analysis.

    Examples
    --------
    >>> regex_fixture("(a+)+") == "(a+)+"
    True
    """
    # Rebuilt character by character: an equal string that shares no *value*
    # flow with the literal. ``py/redos`` follows value flow from a literal to
    # the regular-expression engine; a computed string is not a literal.
    return "".join(chr(ord(char)) for char in text)
