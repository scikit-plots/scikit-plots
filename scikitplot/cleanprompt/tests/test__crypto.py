"""Tests for :mod:`scikitplot.cleanprompt._crypto`."""

from __future__ import annotations

import pytest

from .. import CapabilityError, CleanPromptError
from .. import _capabilities as caps
from .._crypto import decrypt_mapping, encrypt_mapping, new_key

CRYPTO = caps.probe("crypto").available
needs_crypto = pytest.mark.skipif(not CRYPTO, reason="the 'crypto' tier is unavailable")


class TestUnavailableTier:
    """The failure is actionable and arrives before the heavy import."""

    def test_new_key_raises_when_absent(self, monkeypatch):
        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        with pytest.raises(CapabilityError) as caught:
            new_key()
        assert caught.value.tier == "crypto"
        assert "pip install" in str(caught.value)

    def test_encrypt_raises_when_absent(self, monkeypatch):
        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        with pytest.raises(CapabilityError):
            encrypt_mapping({"[A-1]": "x"}, b"k")


@needs_crypto
class TestRoundTrip:
    """Encrypting and decrypting a vault mapping."""

    def test_round_trip(self):
        key = new_key().encode()
        mapping = {"[EMAIL-1]": "ada@example.com", "[PHONE-1]": "+1 555 010 4477"}
        assert decrypt_mapping(encrypt_mapping(mapping, key), key) == mapping

    def test_labels_are_not_encrypted(self):
        """A label is already public; encrypting it buys nothing."""
        key = new_key().encode()
        box = encrypt_mapping({"[EMAIL-1]": "secret"}, key)
        assert list(box) == ["[EMAIL-1]"]

    def test_values_are_not_readable(self):
        key = new_key().encode()
        box = encrypt_mapping({"[EMAIL-1]": "ada@example.com"}, key)
        assert "ada@example.com" not in "".join(box.values())

    def test_each_value_is_encrypted_separately(self):
        key = new_key().encode()
        box = encrypt_mapping({"[A-1]": "same", "[A-2]": "same"}, key)
        assert box["[A-1]"] != box["[A-2]"], "Fernet embeds a random IV per token"

    def test_unicode_survives(self):
        key = new_key().encode()
        mapping = {"[NE-1]": "Mustafa Kemal Atatürk", "[NE-2]": "Ünïcodé 🎉"}
        assert decrypt_mapping(encrypt_mapping(mapping, key), key) == mapping

    def test_empty_mapping(self):
        key = new_key().encode()
        assert encrypt_mapping({}, key) == {}
        assert decrypt_mapping({}, key) == {}

    def test_keys_are_distinct(self):
        assert new_key() != new_key()

    def test_key_is_a_fernet_key(self):
        assert len(new_key()) == 44


@needs_crypto
class TestFailures:
    """Wrong key, tampered file, malformed document."""

    def test_wrong_key_is_refused(self):
        box = encrypt_mapping({"[A-1]": "x"}, new_key().encode())
        with pytest.raises(CleanPromptError, match="failed authentication"):
            decrypt_mapping(box, new_key().encode())

    def test_the_failure_names_the_label_not_the_value(self):
        box = encrypt_mapping({"[SECRET-1]": "topsecret"}, new_key().encode())
        with pytest.raises(CleanPromptError) as caught:
            decrypt_mapping(box, new_key().encode())
        assert "[SECRET-1]" in str(caught.value)
        assert "topsecret" not in str(caught.value)

    def test_tampering_is_detected(self):
        key = new_key().encode()
        box = encrypt_mapping({"[A-1]": "x"}, key)
        token = box["[A-1]"]
        box["[A-1]"] = token[:-4] + ("AAAA" if not token.endswith("AAAA") else "BBBB")
        with pytest.raises(CleanPromptError, match="failed authentication"):
            decrypt_mapping(box, key)

    def test_malformed_key_is_reported_with_guidance(self):
        with pytest.raises(CleanPromptError, match="not a valid Fernet key"):
            encrypt_mapping({"[A-1]": "x"}, b"not-a-key")

    @pytest.mark.parametrize("bad", [[], "string", {"[A-1]": 5}, {5: "x"}, None])
    def test_malformed_document_is_refused(self, bad):
        with pytest.raises(CleanPromptError, match="malformed"):
            decrypt_mapping(bad, new_key().encode())


@needs_crypto
class TestVaultIntegration:
    """End to end through the CLI's vault document."""

    def test_encrypted_vault_round_trips(self, tmp_path, monkeypatch):
        import io

        from .._cli import main

        key = new_key()
        monkeypatch.setenv("CLEANPROMPT_VAULT_KEY", key)
        source = tmp_path / "t.txt"
        text = "mail ada@example.com"
        source.write_text(text, encoding="utf-8")
        vault = tmp_path / "v.json"

        out = io.StringIO()
        main(
            ["redact", "--in", str(source), "--vault", str(vault), "--encrypt",
             "--quiet", "--color", "never"],
            stdout=out, stderr=io.StringIO(),
        )
        redacted = out.getvalue().rstrip("\n")
        assert "ada@example.com" not in vault.read_text(encoding="utf-8")

        reply = tmp_path / "r.txt"
        reply.write_text(redacted, encoding="utf-8")
        back = io.StringIO()
        main(
            ["restore", "--in", str(reply), "--vault", str(vault), "--quiet"],
            stdout=back, stderr=io.StringIO(),
        )
        assert back.getvalue().rstrip("\n") == text

    def test_encrypt_without_a_key_is_refused(self, tmp_path, monkeypatch):
        import io

        from .._cli import main

        monkeypatch.delenv("CLEANPROMPT_VAULT_KEY", raising=False)
        source = tmp_path / "t.txt"
        source.write_text("mail a@b.co", encoding="utf-8")
        err = io.StringIO()
        code = main(
            ["redact", "--in", str(source), "--vault", str(tmp_path / "v.json"),
             "--encrypt", "--quiet"],
            stdout=io.StringIO(), stderr=err,
        )
        assert code == 1
        assert "CLEANPROMPT_VAULT_KEY" in err.getvalue()

    def test_the_document_records_that_it_is_encrypted(self, tmp_path, monkeypatch):
        import io
        import json

        from .._cli import main

        monkeypatch.setenv("CLEANPROMPT_VAULT_KEY", new_key())
        source = tmp_path / "t.txt"
        source.write_text("mail a@b.co", encoding="utf-8")
        vault = tmp_path / "v.json"
        main(
            ["redact", "--in", str(source), "--vault", str(vault), "--encrypt",
             "--quiet", "--color", "never"],
            stdout=io.StringIO(), stderr=io.StringIO(),
        )
        assert json.loads(vault.read_text(encoding="utf-8"))["encrypted"] is True
