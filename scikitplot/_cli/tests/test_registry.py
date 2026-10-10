"""
Tests for :mod:`scikitplot._cli.registry`.

Notes
-----
**Developer notes.** The ``cleanprompt`` install hint names the submodule's
optional extras. It predated the ``nltk`` tier and described the ``crypto``
extra as needed for "vault encryption" after encryption had become base tier,
so a user was sent towards a larger install and away from the NLTK-only path.
The check below reads cleanprompt's tier table from source text, without
importing that submodule, so the two cannot drift apart again.

``scikitplot._cli`` also ships alone, in the ``scikit-plots-skinny`` partial
distribution, where ``scikitplot/cleanprompt`` is not installed. The hint is
checked against the extras it promises everywhere; the comparison with
cleanprompt's own tier table runs where that table exists and is skipped,
with the reason, where it does not. Reading a sibling part's file
unconditionally failed every skinny leg of the partial-distributions workflow
with ``FileNotFoundError``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ..registry import BY_NAME

_CAPABILITIES = Path(__file__).resolve().parents[2] / "cleanprompt" / "_capabilities.py"


def _cleanprompt_tiers():
    """Return the keys of ``TIERS`` in cleanprompt's ``_capabilities.py``."""
    tree = ast.parse(_CAPABILITIES.read_text(encoding="utf-8"))
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else (
            node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None
        )
        if isinstance(target, ast.Name) and target.id == "TIERS" and isinstance(node.value, ast.Dict):
            return {key.value for key in node.value.keys if isinstance(key, ast.Constant)}
    raise AssertionError("TIERS not found in cleanprompt/_capabilities.py")


#: The optional extras the hint promises: ``scikit-plots[cleanprompt-<tier>]``.
_EXTRAS = frozenset({"ner", "nltk", "web", "crypto"})


class TestCleanpromptInstallHint:
    """The hint names every optional extra and does not overstate one."""

    def test_every_extra_is_named(self):
        hint = BY_NAME["cleanprompt"].install_hint
        for tier in _EXTRAS:
            assert f"[cleanprompt-{tier}]" in hint, tier

    @pytest.mark.skipif(
        not _CAPABILITIES.is_file(),
        reason=(
            "scikitplot.cleanprompt is not installed (a partial distribution "
            "such as scikit-plots-skinny); its tier table cannot be compared"
        ),
    )
    def test_the_extras_are_cleanprompts_tiers(self):
        assert _cleanprompt_tiers() == _EXTRAS

    def test_encryption_is_not_described_as_needing_the_crypto_extra(self):
        hint = BY_NAME["cleanprompt"].install_hint
        assert "(vault encryption)" not in hint
        assert "Fernet" in hint
        assert "encryption included" in hint

    def test_the_command_stays_ungated(self):
        """An optional extra must never make the base command unavailable."""
        assert not BY_NAME["cleanprompt"].capabilities
