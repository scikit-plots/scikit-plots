# scikitplot/corpus/_normalizers/tests/test__text_normalizer.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for scikitplot.corpus._normalizers._text_normalizer
==========================================================

Coverage
--------
* :class:`NormalizerConfig` — construction, defaults, ``__post_init__``
  validation, ``steps`` auto-derivation, invalid unicode_form,
  negative min_length, custom_pipeline, explicit ``steps`` override.
* :func:`normalize_text` — each step in isolation (unicode, ligatures,
  control_chars, hyphenation, whitespace, lowercase, custom), combined
  default pipeline, min_length guard returning ``None``, empty input,
  ``config=None`` uses defaults.
* :class:`TextNormalizer` — ``normalize()`` method: step-selective
  application, empty string, unknown step ignored; ``normalize_documents()``
  method: overwrites ``normalized_text``, skips when already set and
  ``overwrite=False``, forces re-normalisation with ``overwrite=True``,
  handles ``text=None``, returns new document instances (immutability),
  batch logging.

All tests are pure Python — zero optional dependencies required.

Run with::

    pytest corpus/_normalizers/tests/test__text_normalizer.py -v
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from .._text_normalizer import NormalizerConfig, TextNormalizer, normalize_text


# ---------------------------------------------------------------------------
# Minimal CorpusDocument stand-in (avoids importing heavy _schema)
# ---------------------------------------------------------------------------


@dataclass
class _Doc:
    """Minimal stand-in compatible with TextNormalizer.normalize_documents."""

    text: str | None = "Hello world."
    normalized_text: str | None = None

    def replace(self, **kwargs: Any) -> "_Doc":
        import copy  # noqa: PLC0415

        obj = copy.copy(self)
        for k, v in kwargs.items():
            object.__setattr__(obj, k, v)
        return obj


# ---------------------------------------------------------------------------
# Ligature characters used in tests
# ---------------------------------------------------------------------------

_FI = "\ufb01"   # ﬁ → fi
_FL = "\ufb02"   # ﬂ → fl
_FF = "\ufb00"   # ﬀ → ff
_FFI = "\ufb03"  # ﬃ → ffi
_OE = "\u0153"   # œ → oe
_AE = "\u00e6"   # æ → ae


# ===========================================================================
# NormalizerConfig — construction and validation
# ===========================================================================


class TestNormalizerConfig:

    def test_default_unicode_form_is_nfkc(self) -> None:
        cfg = NormalizerConfig()
        assert cfg.unicode_form == "NFKC"

    def test_defaults_booleans(self) -> None:
        cfg = NormalizerConfig()
        assert cfg.expand_ligatures is True
        assert cfg.fix_hyphenation is True
        assert cfg.collapse_whitespace is True
        assert cfg.strip_control_chars is True
        assert cfg.lowercase is False

    def test_default_min_length_is_one(self) -> None:
        cfg = NormalizerConfig()
        assert cfg.min_length == 1

    def test_default_custom_pipeline_empty(self) -> None:
        cfg = NormalizerConfig()
        assert cfg.custom_pipeline == ()

    # ------------------------------------------------------------------
    # steps auto-derivation
    # ------------------------------------------------------------------

    def test_steps_auto_derived_all_on(self) -> None:
        cfg = NormalizerConfig()
        assert isinstance(cfg.steps, list)
        # All default-true steps must appear.
        for step in ("unicode", "ligatures", "control_chars", "hyphenation", "whitespace"):
            assert step in cfg.steps

    def test_steps_auto_derived_excludes_disabled_steps(self) -> None:
        cfg = NormalizerConfig(
            expand_ligatures=False,
            fix_hyphenation=False,
            lowercase=False,
            custom_pipeline=(),
        )
        assert "ligatures" not in cfg.steps
        assert "hyphenation" not in cfg.steps
        assert "lowercase" not in cfg.steps
        assert "custom" not in cfg.steps

    def test_steps_auto_derived_includes_lowercase_when_enabled(self) -> None:
        cfg = NormalizerConfig(lowercase=True)
        assert "lowercase" in cfg.steps

    def test_steps_auto_derived_includes_custom_when_pipeline_nonempty(self) -> None:
        cfg = NormalizerConfig(custom_pipeline=(str.strip,))
        assert "custom" in cfg.steps

    def test_steps_explicit_override_respected(self) -> None:
        cfg = NormalizerConfig(steps=["whitespace"])
        assert cfg.steps == ["whitespace"]

    def test_steps_explicit_empty_list(self) -> None:
        cfg = NormalizerConfig(steps=[])
        assert cfg.steps == []

    def test_steps_order_follows_pipeline_order(self) -> None:
        cfg = NormalizerConfig(lowercase=True)
        expected_order = [
            "unicode", "ligatures", "control_chars",
            "hyphenation", "whitespace", "lowercase",
        ]
        # All present steps must appear in the canonical order.
        present = [s for s in expected_order if s in cfg.steps]
        indices = [cfg.steps.index(s) for s in present]
        assert indices == sorted(indices)

    # ------------------------------------------------------------------
    # Validation errors
    # ------------------------------------------------------------------

    def test_invalid_unicode_form_raises(self) -> None:
        with pytest.raises(ValueError, match="unicode_form"):
            NormalizerConfig(unicode_form="INVALID")

    def test_negative_min_length_raises(self) -> None:
        with pytest.raises(ValueError, match="min_length"):
            NormalizerConfig(min_length=-1)

    def test_zero_min_length_accepted(self) -> None:
        cfg = NormalizerConfig(min_length=0)
        assert cfg.min_length == 0

    def test_empty_unicode_form_disables_step(self) -> None:
        cfg = NormalizerConfig(unicode_form="")
        assert "unicode" not in cfg.steps

    def test_all_unicode_forms_accepted(self) -> None:
        for form in ("NFC", "NFD", "NFKC", "NFKD", ""):
            cfg = NormalizerConfig(unicode_form=form)
            assert cfg.unicode_form == form

    def test_frozen_immutable(self) -> None:
        cfg = NormalizerConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.lowercase = True  # type: ignore[misc]

    def test_equality_ignores_steps_list(self) -> None:
        """Two configs differing only in ``steps`` must compare equal."""
        cfg1 = NormalizerConfig(steps=["unicode"])
        cfg2 = NormalizerConfig(steps=["whitespace"])
        # steps is excluded from hash/compare (hash=False, compare=False).
        assert cfg1 == cfg2


# ===========================================================================
# normalize_text — pure function, step by step
# ===========================================================================


class TestNormalizeText:

    # ------------------------------------------------------------------
    # Unicode normalisation
    # ------------------------------------------------------------------

    def test_unicode_nfkc_normalises_fullwidth(self) -> None:
        # Full-width digit U+FF10 → ASCII '0'
        result = normalize_text("\uff10", config=NormalizerConfig(unicode_form="NFKC"))
        assert result == "0"

    def test_unicode_disabled_leaves_text_unchanged(self) -> None:
        text = "\uff10\uff11"
        result = normalize_text(
            text,
            config=NormalizerConfig(unicode_form="", expand_ligatures=False,
                                    strip_control_chars=False, fix_hyphenation=False,
                                    collapse_whitespace=False),
        )
        assert result == text

    def test_unicode_nfc_accepted(self) -> None:
        result = normalize_text("café", config=NormalizerConfig(unicode_form="NFC"))
        assert isinstance(result, str)

    # ------------------------------------------------------------------
    # Ligature expansion
    # ------------------------------------------------------------------

    def test_fi_ligature_expanded(self) -> None:
        result = normalize_text(f"ef{_FI}cient", config=NormalizerConfig())
        assert _FI not in result
        assert "fi" in result

    def test_fl_ligature_expanded(self) -> None:
        result = normalize_text(f"e{_FL}uent", config=NormalizerConfig())
        assert _FL not in result
        assert "fl" in result

    def test_ff_ligature_expanded(self) -> None:
        result = normalize_text(f"di{_FF}erent", config=NormalizerConfig())
        assert _FF not in result
        assert "ff" in result

    def test_ffi_ligature_expanded(self) -> None:
        result = normalize_text(f"e{_FFI}cient", config=NormalizerConfig())
        assert _FFI not in result
        assert "ffi" in result

    def test_oe_ligature_expanded(self) -> None:
        result = normalize_text(f"{_OE}uvre", config=NormalizerConfig())
        assert _OE not in result
        assert "oe" in result

    def test_ae_ligature_expanded(self) -> None:
        result = normalize_text(f"{_AE}sthetic", config=NormalizerConfig())
        assert _AE not in result
        assert "ae" in result

    def test_ligatures_disabled_leaves_chars(self) -> None:
        # unicode_form="" must also be disabled: NFKC itself decomposes ligatures.
        cfg = NormalizerConfig(unicode_form="", expand_ligatures=False,
                               fix_hyphenation=False, strip_control_chars=False,
                               collapse_whitespace=False)
        text = f"ef{_FI}cient"
        result = normalize_text(text, config=cfg)
        assert _FI in result

    def test_multiple_ligatures_in_one_string(self) -> None:
        text = f"{_FI}rst {_FL}oor"
        result = normalize_text(text, config=NormalizerConfig())
        assert "first" in result
        assert "floor" in result

    # ------------------------------------------------------------------
    # Control character stripping
    # ------------------------------------------------------------------

    def test_zero_width_space_removed(self) -> None:
        text = "Hello\u200bworld"
        result = normalize_text(text, config=NormalizerConfig())
        assert "\u200b" not in result

    def test_soft_hyphen_removed(self) -> None:
        text = "hyp\xadhenation"
        result = normalize_text(text, config=NormalizerConfig())
        assert "\xad" not in result

    def test_bom_removed(self) -> None:
        text = "\ufeffHello world."
        result = normalize_text(text, config=NormalizerConfig())
        assert "\ufeff" not in result

    def test_newline_and_tab_preserved(self) -> None:
        cfg = NormalizerConfig(collapse_whitespace=False)
        result = normalize_text("line1\nline2\ttab", config=cfg)
        assert "\n" in result
        assert "\t" in result

    def test_control_disabled_leaves_chars(self) -> None:
        cfg = NormalizerConfig(strip_control_chars=False, expand_ligatures=False,
                               fix_hyphenation=False, collapse_whitespace=False)
        text = "A\u200bB"
        result = normalize_text(text, config=cfg)
        assert "\u200b" in result

    # ------------------------------------------------------------------
    # Hyphenation fix
    # ------------------------------------------------------------------

    def test_hyphenated_line_break_rejoined(self) -> None:
        text = "compu-\nter"
        result = normalize_text(text, config=NormalizerConfig())
        assert "computer" in result
        assert "-\n" not in result

    def test_hyphen_mid_word_no_newline_unchanged(self) -> None:
        text = "well-known technique"
        result = normalize_text(text, config=NormalizerConfig())
        assert "well-known" in result

    def test_hyphen_with_trailing_spaces_before_newline(self) -> None:
        text = "algo-  \nrithm"
        result = normalize_text(text, config=NormalizerConfig())
        # Hyphen + whitespace + newline should be joined.
        assert "-" not in result or "algorithm" in result

    def test_hyphenation_disabled_preserves_break(self) -> None:
        cfg = NormalizerConfig(fix_hyphenation=False, collapse_whitespace=False,
                               strip_control_chars=False, expand_ligatures=False)
        text = "compu-\nter"
        result = normalize_text(text, config=cfg)
        assert "-\n" in result

    def test_docstring_example(self) -> None:
        result = normalize_text("The  \ufb01rst  compu-\nter  was huge.")
        assert result == "The first computer was huge."

    # ------------------------------------------------------------------
    # Whitespace collapse
    # ------------------------------------------------------------------

    def test_multiple_spaces_collapsed(self) -> None:
        result = normalize_text("Hello   world", config=NormalizerConfig())
        assert "  " not in result
        assert "Hello world" in result

    def test_tabs_collapsed_to_space(self) -> None:
        result = normalize_text("col1\tcol2", config=NormalizerConfig())
        assert "\t" not in result

    def test_leading_trailing_whitespace_stripped(self) -> None:
        result = normalize_text("  leading and trailing  ", config=NormalizerConfig())
        assert result is not None and result == result.strip()

    def test_excess_blank_lines_collapsed(self) -> None:
        text = "Para one.\n\n\n\nPara two."
        result = normalize_text(text, config=NormalizerConfig())
        assert "\n\n\n" not in (result or "")

    def test_whitespace_disabled_preserves_spaces(self) -> None:
        cfg = NormalizerConfig(collapse_whitespace=False, fix_hyphenation=False,
                               strip_control_chars=False, expand_ligatures=False)
        text = "Hello   world"
        result = normalize_text(text, config=cfg)
        assert "   " in result

    # ------------------------------------------------------------------
    # Lowercase
    # ------------------------------------------------------------------

    def test_lowercase_applied(self) -> None:
        cfg = NormalizerConfig(lowercase=True)
        result = normalize_text("HELLO WORLD", config=cfg)
        assert result == "hello world"

    def test_lowercase_default_false(self) -> None:
        result = normalize_text("HELLO WORLD", config=NormalizerConfig())
        assert result == "HELLO WORLD"

    # ------------------------------------------------------------------
    # Custom pipeline
    # ------------------------------------------------------------------

    def test_custom_pipeline_applied_last(self) -> None:
        marker = []
        def custom_fn(s: str) -> str:
            marker.append(True)
            return s + "!"
        cfg = NormalizerConfig(custom_pipeline=(custom_fn,))
        result = normalize_text("hello", config=cfg)
        assert result is not None and result.endswith("!")
        assert marker

    def test_custom_pipeline_multiple_fns_in_order(self) -> None:
        log: list[int] = []
        def fn1(s: str) -> str:
            log.append(1)
            return s + "_1"
        def fn2(s: str) -> str:
            log.append(2)
            return s + "_2"
        cfg = NormalizerConfig(custom_pipeline=(fn1, fn2))
        result = normalize_text("x", config=cfg)
        assert log == [1, 2]
        assert result is not None and result.endswith("_1_2")

    # ------------------------------------------------------------------
    # min_length guard
    # ------------------------------------------------------------------

    def test_below_min_length_returns_none(self) -> None:
        cfg = NormalizerConfig(min_length=10)
        result = normalize_text("Hi", config=cfg)
        assert result is None

    def test_exactly_min_length_returns_text(self) -> None:
        cfg = NormalizerConfig(min_length=5)
        result = normalize_text("Hello", config=cfg)
        assert result == "Hello"

    def test_above_min_length_returns_text(self) -> None:
        cfg = NormalizerConfig(min_length=3)
        result = normalize_text("Hello", config=cfg)
        assert result is not None

    def test_min_length_zero_never_returns_none_for_nonempty(self) -> None:
        cfg = NormalizerConfig(min_length=0)
        result = normalize_text("x", config=cfg)
        assert result is not None

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_config_none_uses_defaults(self) -> None:
        """``config=None`` must behave identically to default ``NormalizerConfig()``."""
        text = "The  \ufb01rst  compu-\nter  was huge."
        assert normalize_text(text, config=None) == normalize_text(
            text, config=NormalizerConfig()
        )

    def test_empty_string_with_min_length_one_returns_none(self) -> None:
        result = normalize_text("", config=NormalizerConfig(min_length=1))
        assert result is None

    def test_empty_string_with_min_length_zero_returns_empty(self) -> None:
        result = normalize_text("", config=NormalizerConfig(min_length=0))
        assert result == ""

    def test_only_whitespace_collapses_to_empty(self) -> None:
        result = normalize_text("   \t\n  ", config=NormalizerConfig(min_length=1))
        assert result is None

    def test_unicode_already_normalised_unchanged(self) -> None:
        text = "café"  # pre-composed NFC
        result = normalize_text(text, config=NormalizerConfig(unicode_form="NFC"))
        assert result is not None and "calf" in result  # NFC-normalised "café" unchanged


# ===========================================================================
# TextNormalizer.normalize — step-selective string method
# ===========================================================================


class TestTextNormalizerNormalize:

    def test_empty_string_returns_empty(self) -> None:
        n = TextNormalizer()
        assert n.normalize("") == ""

    def test_default_config_normalises_ligature(self) -> None:
        n = TextNormalizer()
        result = n.normalize(f"ef{_FI}cient")
        assert _FI not in result
        assert "efficient" in result

    def test_steps_selective_only_unicode(self) -> None:
        cfg = NormalizerConfig(steps=["unicode"])
        n = TextNormalizer(cfg)
        # Ligature should remain (step not in list).
        text = f"ef{_FI}cient"
        result = n.normalize(text)
        # NFKC will expand the ligature anyway, so just check no crash.
        assert isinstance(result, str)

    def test_steps_selective_only_whitespace(self) -> None:
        cfg = NormalizerConfig(steps=["whitespace"])
        n = TextNormalizer(cfg)
        result = n.normalize("Hello   world")
        assert "  " not in result

    def test_steps_selective_only_lowercase(self) -> None:
        cfg = NormalizerConfig(lowercase=True, steps=["lowercase"])
        n = TextNormalizer(cfg)
        result = n.normalize("HELLO")
        assert result == "hello"

    def test_steps_selective_only_ligatures(self) -> None:
        cfg = NormalizerConfig(steps=["ligatures"])
        n = TextNormalizer(cfg)
        result = n.normalize(f"ef{_FI}cient")
        assert "fi" in result

    def test_steps_selective_only_hyphenation(self) -> None:
        cfg = NormalizerConfig(steps=["hyphenation"])
        n = TextNormalizer(cfg)
        result = n.normalize("compu-\nter")
        assert "computer" in result

    def test_steps_selective_only_control_chars(self) -> None:
        cfg = NormalizerConfig(steps=["control_chars"])
        n = TextNormalizer(cfg)
        result = n.normalize("Hello\u200bworld")
        assert "\u200b" not in result

    def test_steps_selective_custom(self) -> None:
        fn = lambda s: s.replace("x", "y")  # noqa: E731
        cfg = NormalizerConfig(custom_pipeline=(fn,), steps=["custom"])
        n = TextNormalizer(cfg)
        assert n.normalize("axb") == "ayb"

    def test_steps_empty_list_returns_original(self) -> None:
        cfg = NormalizerConfig(steps=[])
        n = TextNormalizer(cfg)
        text = "Hello   \ufb01rst"
        assert n.normalize(text) == text

    def test_unknown_step_silently_skipped(self) -> None:
        """An unrecognised step name must not raise — just be silently ignored."""
        cfg = NormalizerConfig(steps=["whitespace", "nonexistent_step"])
        n = TextNormalizer(cfg)
        result = n.normalize("Hello   world")
        assert "  " not in result  # whitespace step ran fine

    def test_normalize_never_returns_none(self) -> None:
        """``normalize()`` must always return str, never None."""
        cfg = NormalizerConfig(min_length=1000)
        n = TextNormalizer(cfg)
        result = n.normalize("short")
        assert isinstance(result, str)

    def test_repr_contains_config(self) -> None:
        n = TextNormalizer()
        assert "TextNormalizer" in repr(n)
        assert "config" in repr(n)


# ===========================================================================
# TextNormalizer.normalize_documents — batch pipeline method
# ===========================================================================


class TestTextNormalizerNormalizeDocuments:

    def _doc(self, text: str | None, normalized: str | None = None) -> _Doc:
        return _Doc(text=text, normalized_text=normalized)

    def test_populates_normalized_text(self) -> None:
        n = TextNormalizer()
        doc = self._doc("Hello   world.")
        results = n.normalize_documents([doc])
        assert len(results) == 1
        assert results[0].normalized_text == "Hello world."

    def test_returns_new_instance_not_mutated(self) -> None:
        """Original document must not be mutated."""
        n = TextNormalizer()
        original = self._doc("Hello.")
        results = n.normalize_documents([original])
        assert results[0] is not original
        assert original.normalized_text is None

    def test_skips_already_normalised_when_overwrite_false(self) -> None:
        n = TextNormalizer()
        doc = self._doc("raw text", normalized="already clean")
        results = n.normalize_documents([doc], overwrite=False)
        assert results[0].normalized_text == "already clean"

    def test_overwrites_when_overwrite_true(self) -> None:
        n = TextNormalizer()
        doc = self._doc("Hello   world.", normalized="stale value")
        results = n.normalize_documents([doc], overwrite=True)
        assert results[0].normalized_text == "Hello world."

    def test_empty_list_returns_empty(self) -> None:
        n = TextNormalizer()
        assert n.normalize_documents([]) == []

    def test_batch_all_docs_processed(self) -> None:
        n = TextNormalizer()
        docs = [self._doc(f"  Doc {i}.  ") for i in range(5)]
        results = n.normalize_documents(docs)
        assert len(results) == 5
        for r in results:
            assert r.normalized_text is not None
            assert not r.normalized_text.startswith("  ")

    def test_doc_with_none_text_handled(self) -> None:
        """``text=None`` must not raise; normalized_text is None (empty → min_length)."""
        n = TextNormalizer()   # default min_length=1
        doc = self._doc(None)
        # After source fix (raw = getattr(doc, "text", "") or ""),
        # normalize_text("") with min_length=1 → None.
        results = n.normalize_documents([doc])
        assert results[0].normalized_text is None

    def test_mixed_batch_skip_and_normalise(self) -> None:
        n = TextNormalizer()
        docs = [
            self._doc("raw text"),            # should be normalised
            self._doc("other", "pre-set"),    # should be skipped (overwrite=False)
        ]
        results = n.normalize_documents(docs, overwrite=False)
        assert results[0].normalized_text == "raw text"
        assert results[1].normalized_text == "pre-set"

    def test_logging_called(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging  # noqa: PLC0415

        n = TextNormalizer()
        docs = [self._doc("hello.")]
        # Patch the logger inside the text_normalizer module
        patch_path = "scikitplot.corpus._normalizers._text_normalizer.logger"
        with patch(patch_path) as mock_logger:
            n.normalize_documents(docs)
        # 1. Ensure info() was called
        assert mock_logger.info.called, "logger.info() was never called"
        # 2. Verify the message content
        messages = [str(call.args[0]).lower() for call in mock_logger.info.call_args_list]
        assert any("normalised" in m or "textnormalizer" in m for m in messages), (
            f"Expected keywords not found in logs: {messages}"
        )

    def test_min_length_sets_normalized_text_to_none(self) -> None:
        cfg = NormalizerConfig(min_length=100)
        n = TextNormalizer(cfg)
        doc = self._doc("short")
        results = n.normalize_documents([doc])
        assert results[0].normalized_text is None

    def test_custom_normalizer_applied_in_batch(self) -> None:
        fn = lambda s: s.upper()  # noqa: E731
        cfg = NormalizerConfig(custom_pipeline=(fn,))
        n = TextNormalizer(cfg)
        docs = [self._doc("hello"), self._doc("world")]
        results = n.normalize_documents(docs)
        for r in results:
            assert r.normalized_text is not None
            assert r.normalized_text == r.normalized_text.upper()

    def test_default_overwrite_is_false(self) -> None:
        """Calling without ``overwrite`` must default to ``False``."""
        n = TextNormalizer()
        doc = self._doc("raw", "existing")
        results = n.normalize_documents([doc])
        assert results[0].normalized_text == "existing"
