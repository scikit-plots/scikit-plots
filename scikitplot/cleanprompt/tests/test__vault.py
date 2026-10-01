"""Tests for :mod:`scikitplot.cleanprompt._vault`."""

from __future__ import annotations

import json

import pytest

from .. import PolicyError, Vault


class TestConstruction:
    """Building a vault."""

    def test_empty(self):
        vault = Vault()
        assert len(vault) == 0
        assert vault.labels() == ()
        assert vault.grammar_fingerprint is None

    def test_from_mapping(self):
        vault = Vault({"[A-1]": "x", "[B-1]": "y"})
        assert len(vault) == 2
        assert vault["[A-1]"] == "x"

    def test_does_not_alias_the_caller_dict(self):
        source = {"[A-1]": "x"}
        vault = Vault(source)
        source["[A-1]"] = "tampered"
        assert vault["[A-1]"] == "x"

    def test_carries_a_grammar_fingerprint(self):
        assert Vault({}, grammar_fingerprint="abc").grammar_fingerprint == "abc"


class TestAdd:
    """Append-only semantics."""

    def test_add_then_read(self):
        vault = Vault()
        vault.add("[A-1]", "secret")
        assert vault["[A-1]"] == "secret"

    def test_identical_re_add_is_a_no_op(self):
        vault = Vault()
        vault.add("[A-1]", "secret")
        vault.add("[A-1]", "secret")
        assert len(vault) == 1

    def test_conflicting_re_add_is_refused(self):
        """Two secrets under one label would make restoration ambiguous."""
        vault = Vault({"[A-1]": "one"})
        with pytest.raises(PolicyError, match="already bound"):
            vault.add("[A-1]", "two")

    def test_empty_label_is_refused(self):
        with pytest.raises(PolicyError, match="non-empty"):
            Vault().add("", "x")

    def test_add_to_a_cleared_vault_is_refused(self):
        vault = Vault({"[A-1]": "x"})
        vault.clear()
        with pytest.raises(PolicyError, match="cleared"):
            vault.add("[A-2]", "y")


class TestReads:
    """Lookup surface."""

    def test_get_with_default(self):
        vault = Vault({"[A-1]": "x"})
        assert vault.get("[A-1]") == "x"
        assert vault.get("[Z-9]") is None
        assert vault.get("[Z-9]", "fallback") == "fallback"

    def test_contains_and_iter(self):
        vault = Vault({"[A-1]": "x", "[B-1]": "y"})
        assert "[A-1]" in vault
        assert "[Z-1]" not in vault
        assert list(vault) == ["[A-1]", "[B-1]"]

    def test_labels_preserve_insertion_order(self):
        vault = Vault()
        for index in range(5):
            vault.add("[A-{0}]".format(index), str(index))
        assert vault.labels() == tuple("[A-{0}]".format(i) for i in range(5))

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            Vault()["[A-1]"]


class TestSecrecy:
    """Nothing incidental may disclose a value."""

    def test_repr_hides_values(self):
        vault = Vault({"[A-1]": "topsecret"})
        assert repr(vault) == "Vault(entries=1, closed=False)"
        assert "topsecret" not in repr(vault)

    def test_str_hides_values(self):
        assert "topsecret" not in str(Vault({"[A-1]": "topsecret"}))

    def test_format_hides_values(self):
        assert "topsecret" not in "{0}".format(Vault({"[A-1]": "topsecret"}))

    def test_traceback_message_hides_values(self):
        vault = Vault({"[A-1]": "topsecret"})
        try:
            raise RuntimeError("context: {0!r}".format(vault))
        except RuntimeError as exc:
            assert "topsecret" not in str(exc)

    def test_export_is_the_only_named_way_out(self):
        vault = Vault({"[A-1]": "topsecret"})
        assert vault.export() == {"[A-1]": "topsecret"}
        assert "topsecret" in json.dumps(vault.export())

    def test_export_returns_a_copy(self):
        vault = Vault({"[A-1]": "x"})
        exported = vault.export()
        exported["[A-1]"] = "tampered"
        assert vault["[A-1]"] == "x"


class TestLifecycle:
    """Clearing and the context manager."""

    def test_clear_empties_and_closes(self):
        vault = Vault({"[A-1]": "x"})
        vault.clear()
        assert len(vault) == 0
        assert vault.closed is True

    def test_clear_overwrites_before_dropping(self):
        """Any surviving reference must observe no plaintext."""
        vault = Vault()
        vault.add("[A-1]", "topsecret")
        inner = vault._mapping  # noqa: SLF001 - the property under test
        vault.clear()
        assert "topsecret" not in inner.values()

    def test_clear_is_idempotent(self):
        vault = Vault({"[A-1]": "x"})
        vault.clear()
        vault.clear()
        assert vault.closed is True

    def test_export_after_clear_is_refused(self):
        vault = Vault({"[A-1]": "x"})
        vault.clear()
        with pytest.raises(PolicyError, match="cleared"):
            vault.export()

    def test_context_manager_clears_on_exit(self):
        with Vault({"[A-1]": "x"}) as vault:
            assert len(vault) == 1
        assert vault.closed is True

    def test_context_manager_clears_on_exception(self):
        vault = Vault({"[A-1]": "x"})
        with pytest.raises(ValueError):
            with vault:
                raise ValueError("boom")
        assert vault.closed is True


class TestSlots:
    """The vault must not grow arbitrary attributes."""

    def test_has_slots(self):
        assert hasattr(Vault, "__slots__")

    def test_cannot_attach_an_attribute(self):
        with pytest.raises(AttributeError):
            Vault().stash = "secret"
