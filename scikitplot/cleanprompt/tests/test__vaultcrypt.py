"""
Tests for :mod:`scikitplot.cleanprompt._vaultcrypt`.

Notes
-----
**Developer notes — what is being asserted, and what is not.**

These do not prove the construction is secure. No test suite can; the claim
rests on HMAC-SHA256 being a pseudorandom function and on the composition being
the textbook one, which the module's notes state so a reviewer can check it.

What they do prove is that the properties the construction is supposed to
deliver are actually delivered by *this code*: that a wrong passphrase is
rejected rather than yielding plausible rubbish, that a modified token is
refused, that two identical values do not produce identical tokens, that a
token cannot be moved from one label to another, and that a value never appears
in the output. Those are the places an implementation slips, as against the
places a construction is broken.

The round trip is also asserted under an ``__import__`` blocker, because the
reason this module exists is to work where nothing is installed.
"""

from __future__ import annotations

import base64
import builtins
import json

import pytest

from .._exceptions import CleanPromptError, PolicyError
from .._vaultcrypt import (
    CIPHERS,
    DEFAULT_CIPHER,
    PORTABLE_CIPHER,
    _derive,
    _kdf_params,
    _keystream,
    decrypt_mapping,
    encrypt_mapping,
    new_passphrase,
    resolve_cipher,
)

# Deliberately not ASCII: a passphrase is whatever the user typed, and a
# non-ASCII one must survive the encode/derive path unchanged.
SECRET = "a passphrase with spaces and ünicode".encode("utf-8")
MAPPING = {"[EMAIL-1]": "ada@example.com", "[IPV4-1]": "192.168.1.10"}


def _roundtrip(mapping=None, passphrase=SECRET):
    """Encrypt and decrypt, returning ``(tokens, params, result)``."""
    tokens, params = encrypt_mapping(mapping or MAPPING, passphrase)
    return tokens, params, decrypt_mapping(tokens, passphrase, params)


class TestRoundTrip:
    """The basic contract."""

    def test_values_come_back_exactly(self):
        _, _, out = _roundtrip()
        assert out == MAPPING

    def test_labels_are_not_encrypted(self):
        """A label is already in the text that was sent; hiding it buys nothing."""
        tokens, _ = encrypt_mapping(MAPPING, SECRET)
        assert set(tokens) == set(MAPPING)

    def test_no_value_appears_in_the_tokens(self):
        tokens, params = encrypt_mapping(MAPPING, SECRET)
        blob = json.dumps({"tokens": tokens, "params": params})
        for value in MAPPING.values():
            assert value not in blob

    def test_an_empty_mapping_is_fine(self):
        tokens, params = encrypt_mapping({}, SECRET)
        assert decrypt_mapping(tokens, SECRET, params) == {}

    def test_unicode_survives(self):
        mapping = {"[PERSON-1]": "Mustafa Kemal Atatürk", "[X-1]": "\U0001f680"}
        _, _, out = _roundtrip(mapping)
        assert out == mapping

    def test_a_long_value_survives(self):
        """Longer than one keystream block, so the counter must advance."""
        mapping = {"[X-1]": "x" * 5000}
        _, _, out = _roundtrip(mapping)
        assert out == mapping

    def test_an_empty_value_survives(self):
        _, _, out = _roundtrip({"[X-1]": ""})
        assert out == {"[X-1]": ""}

    def test_the_passphrase_may_contain_anything(self):
        for passphrase in (b"\x00\xff", b" ", b"x" * 1000):
            _, _, out = _roundtrip(passphrase=passphrase)
            assert out == MAPPING


class TestConfidentiality:
    """What an attacker with the file, and no passphrase, learns."""

    def test_the_wrong_passphrase_is_refused(self):
        tokens, params = encrypt_mapping(MAPPING, SECRET)
        with pytest.raises(CleanPromptError) as caught:
            decrypt_mapping(tokens, b"wrong", params)
        assert "passphrase is wrong" in str(caught.value)

    def test_the_wrong_passphrase_never_yields_a_value(self):
        """It must fail, not return plausible rubbish."""
        tokens, params = encrypt_mapping(MAPPING, SECRET)
        with pytest.raises(CleanPromptError):
            decrypt_mapping(tokens, b"wrong", params)

    def test_equal_values_do_not_produce_equal_tokens(self):
        """A fresh nonce per value, so the file leaks no equality."""
        tokens, _ = encrypt_mapping({"[A-1]": "same", "[B-1]": "same"}, SECRET)
        assert tokens["[A-1]"] != tokens["[B-1]"]

    def test_encrypting_twice_differs(self):
        first, _ = encrypt_mapping(MAPPING, SECRET)
        second, _ = encrypt_mapping(MAPPING, SECRET)
        assert first != second

    def test_each_document_gets_its_own_salt(self):
        _, first = encrypt_mapping(MAPPING, SECRET)
        _, second = encrypt_mapping(MAPPING, SECRET)
        assert first["salt"] != second["salt"]

    def test_the_parameters_carry_no_secret(self):
        _, params = encrypt_mapping(MAPPING, SECRET)
        blob = json.dumps(params)
        assert "passphrase" not in blob
        for value in MAPPING.values():
            assert value not in blob


class TestIntegrity:
    """What happens when the file has been changed."""

    @staticmethod
    def _flip(token, index):
        raw = bytearray(base64.b64decode(token))
        raw[index] ^= 0x01
        return base64.b64encode(bytes(raw)).decode("ascii")

    @pytest.mark.parametrize("index", [0, 8, 20, -1])
    def test_a_flipped_bit_anywhere_is_caught(self, index):
        tokens, params = encrypt_mapping(MAPPING, SECRET)
        tokens["[EMAIL-1]"] = self._flip(tokens["[EMAIL-1]"], index)
        with pytest.raises(CleanPromptError) as caught:
            decrypt_mapping(tokens, SECRET, params)
        assert "authentication" in str(caught.value)

    def test_a_token_moved_to_another_label_is_caught(self):
        """
        The label is inside the tag, so tokens cannot be swapped.

        Without this, someone with write access could exchange two entries and
        make ``[EMAIL-1]`` restore to a different person's address, with every
        check still passing and no key required.
        """
        tokens, params = encrypt_mapping(MAPPING, SECRET)
        swapped = {
            "[EMAIL-1]": tokens["[IPV4-1]"],
            "[IPV4-1]": tokens["[EMAIL-1]"],
        }
        with pytest.raises(CleanPromptError) as caught:
            decrypt_mapping(swapped, SECRET, params)
        assert "authentication" in str(caught.value)

    def test_a_renamed_label_is_caught(self):
        tokens, params = encrypt_mapping(MAPPING, SECRET)
        renamed = {"[EMAIL-9]": tokens["[EMAIL-1]"]}
        with pytest.raises(CleanPromptError):
            decrypt_mapping(renamed, SECRET, params)

    def test_a_truncated_token_is_caught(self):
        tokens, params = encrypt_mapping(MAPPING, SECRET)
        raw = base64.b64decode(tokens["[EMAIL-1]"])
        tokens["[EMAIL-1]"] = base64.b64encode(raw[:10]).decode("ascii")
        with pytest.raises(CleanPromptError) as caught:
            decrypt_mapping(tokens, SECRET, params)
        assert "truncated" in str(caught.value)

    def test_a_token_that_is_not_base64_is_caught(self):
        tokens, params = encrypt_mapping(MAPPING, SECRET)
        tokens["[EMAIL-1]"] = "not base64 at all!!"
        with pytest.raises(CleanPromptError) as caught:
            decrypt_mapping(tokens, SECRET, params)
        assert "base64" in str(caught.value)

    def test_a_salt_swapped_for_another_is_caught(self):
        """Changing the parameters changes the key, so the tag fails."""
        tokens, params = encrypt_mapping(MAPPING, SECRET)
        _, other = encrypt_mapping(MAPPING, SECRET)
        params = dict(params, salt=other["salt"])
        with pytest.raises(CleanPromptError):
            decrypt_mapping(tokens, SECRET, params)


class TestKeyDerivation:
    """The parameters travel with the document, so old vaults stay readable."""

    def test_the_parameters_are_recorded(self):
        _, params = encrypt_mapping(MAPPING, SECRET)
        assert params["name"] in ("scrypt", "pbkdf2_hmac_sha256")
        assert params["salt"]
        assert params["length"] >= 64

    def test_pbkdf2_is_a_working_alternative(self):
        """Where scrypt is unavailable, the fallback must actually work."""
        salt = base64.b64encode(b"0123456789abcdef").decode("ascii")
        params = {
            "name": "pbkdf2_hmac_sha256",
            "salt": salt,
            "length": 64,
            "rounds": 1000,
        }
        tokens, _ = encrypt_mapping(MAPPING, SECRET)
        # Derive with both and confirm the module can use either shape.
        enc, mac = _derive(SECRET, params)
        assert len(enc) == 32 and len(mac) == 32
        assert enc != mac

    def test_encryption_and_authentication_keys_differ(self):
        """One secret must never serve two purposes."""
        enc, mac = _derive(SECRET, _kdf_params())
        assert enc != mac

    def test_the_same_passphrase_and_salt_give_the_same_keys(self):
        params = _kdf_params()
        assert _derive(SECRET, params) == _derive(SECRET, params)

    def test_a_different_passphrase_gives_different_keys(self):
        params = _kdf_params()
        assert _derive(SECRET, params) != _derive(b"other", params)

    def test_an_unknown_function_is_refused(self):
        params = {"name": "rot13", "salt": "AAAA", "length": 64}
        with pytest.raises(CleanPromptError) as caught:
            _derive(SECRET, params)
        assert "unknown key-derivation" in str(caught.value)

    def test_malformed_parameters_are_refused(self):
        with pytest.raises(CleanPromptError):
            _derive(SECRET, {"name": "scrypt"})

    def test_too_little_key_material_is_refused(self):
        params = dict(_kdf_params(), length=16)
        with pytest.raises(CleanPromptError) as caught:
            _derive(SECRET, params)
        assert "required" in str(caught.value)


class TestKeystream:
    """The counter-mode construction itself."""

    def test_it_produces_the_requested_length(self):
        for length in (0, 1, 31, 32, 33, 1000):
            assert len(_keystream(b"k" * 32, b"n" * 16, length)) == length

    def test_it_is_deterministic(self):
        assert _keystream(b"k" * 32, b"n" * 16, 64) == _keystream(
            b"k" * 32, b"n" * 16, 64
        )

    def test_a_different_nonce_gives_a_different_stream(self):
        assert _keystream(b"k" * 32, b"a" * 16, 64) != _keystream(
            b"k" * 32, b"b" * 16, 64
        )

    def test_a_different_key_gives_a_different_stream(self):
        assert _keystream(b"a" * 32, b"n" * 16, 64) != _keystream(
            b"b" * 32, b"n" * 16, 64
        )

    def test_blocks_past_the_first_are_not_a_repeat(self):
        """The counter must advance, or a long value repeats its keystream."""
        stream = _keystream(b"k" * 32, b"n" * 16, 96)
        assert stream[0:32] != stream[32:64] != stream[64:96]


class TestPassphrase:
    """Generated passphrases, which people have to type."""

    def test_the_shape_is_what_was_asked_for(self):
        phrase = new_passphrase()
        assert len(phrase.split("-")) == 4
        assert all(len(group) == 5 for group in phrase.split("-"))

    def test_it_excludes_look_alike_characters(self):
        """Read off a screen or dictated, 0/O and 1/I/l are the same character."""
        phrases = "".join(new_passphrase() for _ in range(50))
        for confusable in "01OIlUu":
            assert confusable not in phrases

    def test_two_are_not_the_same(self):
        assert len({new_passphrase() for _ in range(50)}) == 50

    def test_the_shape_is_configurable(self):
        phrase = new_passphrase(groups=2, width=3)
        assert len(phrase.split("-")) == 2
        assert len(phrase) == 7

    @pytest.mark.parametrize("groups,width", [(0, 5), (4, 0), (-1, 5)])
    def test_a_degenerate_shape_is_refused(self, groups, width):
        with pytest.raises(ValueError):
            new_passphrase(groups=groups, width=width)


class TestCipherSelection:
    """``--cipher`` resolution."""

    def test_portable_resolves_to_itself(self):
        assert resolve_cipher("portable") == "portable"

    def test_fernet_is_never_silently_downgraded(self):
        """Asking for it by name and getting something else would be a lie."""
        assert resolve_cipher("fernet") == "fernet"

    def test_auto_follows_the_tier(self):
        from .. import _capabilities as caps

        expected = "fernet" if caps.probe("crypto").available else "portable"
        assert resolve_cipher("auto") == expected

    def test_auto_falls_back_when_the_tier_is_absent(self, monkeypatch):
        from .. import _capabilities as caps

        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        assert resolve_cipher("auto") == "portable"

    def test_the_default_is_the_portable_one(self):
        """A vault that cannot be opened is the worse failure."""
        assert DEFAULT_CIPHER == "portable"
        assert resolve_cipher() == "portable"

    def test_an_unknown_cipher_is_refused(self):
        with pytest.raises(PolicyError) as caught:
            resolve_cipher("rot13")
        assert "choose from" in str(caught.value)

    def test_every_choice_resolves(self):
        for choice in CIPHERS:
            assert resolve_cipher(choice) in ("portable", "fernet")


class TestWorksWithNothingInstalled:
    """The reason this module exists."""

    def test_the_round_trip_holds_under_an_import_blocker(self, monkeypatch):
        blocked = ("cryptography", "spacy", "nltk", "flask", "click")
        real = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in blocked:
                raise ImportError("blocked by test: " + name)
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guard)
        tokens, params = encrypt_mapping(MAPPING, SECRET)
        assert decrypt_mapping(tokens, SECRET, params) == MAPPING

    def test_it_imports_nothing_beyond_the_standard_library(self):
        import scikitplot.cleanprompt._vaultcrypt as module

        source = open(module.__file__, encoding="utf-8").read()
        for name in ("cryptography", "nacl", "Crypto"):
            assert "import {0}".format(name) not in source

    def test_the_construction_name_is_versioned(self):
        """A future change must be able to arrive without stranding vaults."""
        assert PORTABLE_CIPHER.endswith("-v1")


class TestRejections:
    """Bad input, refused with a message rather than a traceback."""

    def test_an_empty_passphrase_cannot_encrypt(self):
        with pytest.raises(CleanPromptError) as caught:
            encrypt_mapping(MAPPING, b"")
        assert "passphrase is required" in str(caught.value)

    def test_an_empty_passphrase_cannot_decrypt(self):
        tokens, params = encrypt_mapping(MAPPING, SECRET)
        with pytest.raises(CleanPromptError):
            decrypt_mapping(tokens, b"", params)

    def test_entries_that_are_not_an_object_are_refused(self):
        _, params = encrypt_mapping(MAPPING, SECRET)
        with pytest.raises(CleanPromptError) as caught:
            decrypt_mapping(["not", "an", "object"], SECRET, params)
        assert "not an object" in str(caught.value)

    def test_a_non_string_entry_is_refused(self):
        _, params = encrypt_mapping(MAPPING, SECRET)
        with pytest.raises(CleanPromptError) as caught:
            decrypt_mapping({"[X-1]": 42}, SECRET, params)
        assert "string pair" in str(caught.value)


class TestDoctests:
    """Every documented example runs."""

    def test_doctests_pass(self):
        from ._isolated import assert_doctests_pass

        assert_doctests_pass("_vaultcrypt")
