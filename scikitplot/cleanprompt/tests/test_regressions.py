"""
One named regression per defect reproduced against the upstream project.

Notes
-----
**Developer notes.** Each test names the defect identifier from
``maintenances/cleanprompt/REVIEW.json`` and asserts the *desired* behaviour,
never the historical one. A test that encodes a known defect as expected
behaviour turns a green suite into evidence that the implementation and the
tests agree, which is not the same as evidence that the contract is right.

The reproductions themselves are in
``maintenances/cleanprompt/_maintenance/evidence/upstream-defects.log``. These
tests are the permanent gate; the log is the provenance.
"""

from __future__ import annotations

import re

import pytest

from .. import (
    DEFAULT_POLICY,
    LiteralDetector,
    Redactor,
    default_registry,
    get_pattern,
    restore,
)


def _matches(kind, text):
    """Return the accepted matches of one pattern against ``text``."""
    spec = get_pattern(kind)
    return [
        match.group()
        for match in spec.compiled().finditer(text)
        if spec.validate is None or spec.validate(match)
    ]


class TestCP001SubstringCollision:
    """``CP-001`` — a shorter secret must not corrupt a longer one."""

    def test_overlapping_literal_terms_do_not_corrupt(self, redactor):
        """Upstream produced ``'[ADDITIONAL-1] met [ADDITIONAL-1]a'``."""
        result = redactor.redact("Ann met Anna", extra_terms=["Ann", "Anna"])
        assert result.text == "[CUSTOM-1] met [CUSTOM-2]"

    def test_no_fragment_of_a_secret_survives(self, redactor):
        """The trailing ``a`` of ``Anna`` leaked upstream; it must not here."""
        result = redactor.redact("Anna", extra_terms=["Ann", "Anna"])
        assert result.text == "[CUSTOM-1]"
        assert "a" not in result.text

    def test_round_trip_is_exact(self, redactor):
        original = "Ann met Anna and Ann again"
        result = redactor.redact(original, extra_terms=["Ann", "Anna"])
        assert restore(result.text, result.vault).text == original

    def test_literal_detector_prefers_the_longer_term(self):
        """Longest-first alternation, independent of the order given."""
        for terms in (["Ann", "Anna"], ["Anna", "Ann"]):
            spans = list(
                LiteralDetector(terms).detect("Ann met Anna", DEFAULT_POLICY)
            )
            assert [span.text for span in spans] == ["Ann", "Anna"]


class TestCP002AdditionalTermsDropDetection:
    """``CP-002`` — supplying extra terms must never reduce what is detected."""

    def test_extra_terms_do_not_disable_structural_detection(self, redactor):
        """Upstream skipped email, phone and URL once extra terms were given."""
        text = "mail a@example.com about Acme"
        result = redactor.redact(text, extra_terms=["Acme"])
        assert "a@example.com" not in result.text
        assert "[EMAIL-1]" in result.text
        assert "[CUSTOM-1]" in result.text

    def test_adding_a_term_is_monotone(self, redactor):
        """Adding a term can only add entries, never remove one."""
        text = "mail a@example.com about Acme"
        without = redactor.redact(text)
        with_term = redactor.redact(text, extra_terms=["Acme"])
        assert {entry.kind for entry in without.entries} <= {
            entry.kind for entry in with_term.entries
        }
        assert with_term.stats.entries > without.stats.entries


class TestCP003SharedMutableState:
    """``CP-003`` — no state may carry between documents or callers."""

    def test_numbering_restarts_for_each_document(self, redactor):
        """Upstream's second document continued the first one's counters."""
        first = redactor.redact("mail a@example.com")
        second = redactor.redact("mail b@example.com")
        assert first.text == second.text == "mail [EMAIL-1]"

    def test_vaults_are_independent(self, redactor):
        first = redactor.redact("mail a@example.com")
        second = redactor.redact("mail b@example.com")
        assert first.vault["[EMAIL-1]"] == "a@example.com"
        assert second.vault["[EMAIL-1]"] == "b@example.com"

    def test_clearing_one_vault_leaves_the_other_intact(self, redactor):
        first = redactor.redact("mail a@example.com")
        second = redactor.redact("mail b@example.com")
        first.vault.clear()
        assert second.vault["[EMAIL-1]"] == "b@example.com"

    def test_concurrent_use_is_safe(self, redactor):
        """One redactor, many threads, no interference."""
        import threading

        outcomes = {}

        def work(index):
            text = "mail user{0}@example.com".format(index)
            result = redactor.redact(text)
            outcomes[index] = (result.text, result.vault["[EMAIL-1]"])

        threads = [threading.Thread(target=work, args=(i,)) for i in range(16)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(outcomes) == 16
        for index, (text, secret) in outcomes.items():
            assert text == "mail [EMAIL-1]"
            assert secret == "user{0}@example.com".format(index)


class TestCP004UrlCharacterRange:
    """``CP-004`` — the URL class must not be an accidental range."""

    def test_upstream_class_was_a_range(self):
        """Documents the defect: ``[$-_@.&+]`` spans 0x24 to 0x5F."""
        upstream = re.compile(r"[$-_@.&+]")
        assert upstream.match("5") and upstream.match("A") and upstream.match("<")

    def test_current_url_pattern_stops_at_the_url(self):
        assert _matches("URL", "see https://example.com/a?b=1#c now") == [
            "https://example.com/a?b=1#c"
        ]

    def test_bare_host_is_not_a_url(self):
        assert _matches("URL", "example.com") == []


class TestCP005EmailTopLevelLabel:
    """``CP-005`` — the top-level label must not admit a literal pipe."""

    def test_upstream_class_admitted_a_pipe(self):
        upstream = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )
        assert upstream.search("a@b.a|b") is not None

    def test_current_pattern_rejects_it(self):
        assert "a@b.a|b" not in _matches("EMAIL", "a@b.a|b")

    def test_ordinary_addresses_still_match(self):
        assert _matches("EMAIL", "a.b+tag@example.co.uk") == ["a.b+tag@example.co.uk"]


class TestCP006DetectorsSeeRewrittenText:
    """``CP-006`` — no detector may observe an inserted placeholder."""

    def test_redaction_is_idempotent(self, redactor, sample_text):
        once = redactor.redact(sample_text)
        twice = redactor.redact(once.text)
        assert twice.text == once.text
        assert twice.stats.entries == 0

    def test_placeholders_are_never_consumed(self, redactor):
        """Upstream turned ``'Contact [EMAIL-1] now'`` into ``'[[ORG-1]-1]'``."""
        result = redactor.redact("Contact [EMAIL-1] now")
        assert "[EMAIL-1]" in result.text

    def test_pre_existing_label_does_not_alias_a_new_one(self, redactor):
        """A fresh label must never collide with one already in the input."""
        original = "[EMAIL-1] and bob@example.com"
        result = redactor.redact(original)
        assert result.text == "[EMAIL-1] and [EMAIL-2]"
        assert restore(result.text, result.vault).text == original


class TestCP008MutableDefault:
    """``CP-008`` — no mutable default arguments."""

    def test_no_public_callable_has_a_mutable_default(self):
        import inspect

        from .. import _detectors, _engine, _policy, _render, _vault

        offenders = []
        for module in (_detectors, _engine, _policy, _render, _vault):
            for name, obj in vars(module).items():
                if name.startswith("_") or not callable(obj):
                    continue
                targets = [obj]
                if inspect.isclass(obj):
                    targets = [
                        value
                        for key, value in vars(obj).items()
                        if inspect.isfunction(value) and not key.startswith("__")
                    ]
                for target in targets:
                    try:
                        signature = inspect.signature(target)
                    except (TypeError, ValueError):  # pragma: no cover
                        continue
                    for parameter in signature.parameters.values():
                        if isinstance(parameter.default, (list, dict, set)):
                            offenders.append(
                                "{0}.{1}:{2}".format(
                                    module.__name__, target.__name__, parameter.name
                                )
                            )
        assert offenders == []


class TestCP012ColourInData:
    """``CP-012`` — presentation must not be baked into returned values."""

    def test_restoration_returns_plain_text(self, redactor):
        result = redactor.redact("mail ada@example.com")
        restored = restore(result.text, result.vault)
        assert "\033" not in restored.text
        assert restored.text == "mail ada@example.com"

    def test_highlighting_is_opt_in_and_separate(self, redactor):
        from .. import highlight_placeholders

        result = redactor.redact("mail ada@example.com")
        assert "\033" not in result.text
        assert "\033" in highlight_placeholders(result.text, color=True)
        assert highlight_placeholders(result.text, color=False) == result.text


class TestCP013PhoneFalsePositives:
    """``CP-013`` — a date or an order number is not a phone number."""

    @pytest.mark.parametrize(
        "text", ["2024-01-15", "12345678", "numpy 1.26.4", "Revenue rose 1234"]
    )
    def test_rejected(self, text):
        assert _matches("PHONE", text) == []

    @pytest.mark.parametrize(
        "text", ["+1 555 010 4477", "(020) 7946 0958", "555-010-4477"]
    )
    def test_accepted(self, text):
        assert _matches("PHONE", text) == [text]

    def test_iso_date_survives_a_full_redaction(self, redactor):
        result = redactor.redact("released 2024-01-15")
        assert result.text == "released 2024-01-15"


class TestCP015ValidatorShadowing:
    """
    ``CP-015`` — a rejected candidate must not consume the text it covered.

    Notes
    -----
    **Developer notes.** Found during this rewrite's own scale verification, not
    inherited from upstream. :meth:`re.Pattern.finditer` hands back each match
    *after* consuming it, so a match the validator rejects has already swallowed
    everything it spanned, and shorter candidates starting inside it are never
    offered. The consequence is a leak: real values survive into the redacted
    output because a longer rejected candidate hid them.
    """

    def test_shorter_candidate_is_found_after_a_rejection(self, redactor):
        """
        ``4242 4242`` must not be stranded by the rejected longer match.

        Notes
        -----
        **Developer notes.** The assertion is the guarantee, not a particular
        decomposition. Here the surviving detections overlap partially, so the
        documented merge rule folds them into one placeholder covering the whole
        region. That is coarser than two labels but it is the safe outcome, and
        it is what the design chose: discarding a partially overlapping span
        would leave the characters it alone covered in the output.
        """
        result = redactor.redact("123-45-6789 4242 4242 4242 4242")
        assert "4242" not in result.text
        assert "123-45-6789" not in result.text
        assert result.text == "[CREDIT_CARD-1]"

    def test_longest_accepted_match_wins_at_a_start(self, redactor):
        """Not the longest match — the longest match the validator accepts."""
        result = redactor.redact("4477 555 010 4477 2024-01-15")
        assert result.text == "[PHONE-1] 2024-01-15"

    def test_two_glued_numbers_are_separated(self, redactor):
        result = redactor.redact("4242 4242 4242 555 010 4477")
        assert result.text == "[PHONE-1] [PHONE-2]"

    def test_no_detected_value_survives(self, redactor):
        for text in (
            "123-45-6789 4242 4242 4242 4242",
            "4477 555 010 4477 2024-01-15",
            "010 4477 4242 4242 4242",
            "1234 5698 7654 555 010 4477",
        ):
            result = redactor.redact(text)
            for entry in result.entries:
                assert entry.original not in result.text

    def test_rejection_scanning_stays_fast(self, redactor):
        """The retry loop must not make a digit-heavy document quadratic."""
        import time

        text = " ".join(["4242"] * 4000)
        started = time.monotonic()
        redactor.redact(text)
        assert time.monotonic() - started < 5.0


class TestCP016WindowBoundary:
    """
    ``CP-016`` — an ``endpos`` window must not fabricate a boundary.

    Notes
    -----
    **Developer notes.** Also found during this rewrite's verification, and a
    direct consequence of the ``CP-015`` fix. Limiting a match with ``endpos``
    makes the engine treat that offset as end-of-string, so a trailing ``\\b``
    or ``(?!\\w)`` succeeds there even when the text continues with a word
    character. Without re-verification the scanner would accept a truncated
    match and leave the tail of a real value in the clear.
    """

    def test_a_value_is_never_cut_in_half(self, redactor):
        result = redactor.redact("4242 4242 4242 4242x")
        for entry in result.entries:
            assert entry.original not in result.text
        assert "[PHONE-1] 4242x" == result.text

    def test_trailing_word_character_blocks_a_false_boundary(self, redactor):
        """The lookahead must be judged against the real next character."""
        result = redactor.redact("call 555-010-4477abc")
        assert result.text == "call 555-010-4477abc"

    def test_a_genuine_boundary_is_still_accepted(self, redactor):
        assert redactor.redact("call 555-010-4477.").text == "call [PHONE-1]."

    def test_round_trip_survives_the_boundary_logic(self, redactor):
        for text in ("4242 4242 4242 4242x", "call 555-010-4477.", "a 010 4477 b"):
            result = redactor.redact(text)
            assert restore(result.text, result.vault).text == text


class TestPlaceholderPrefixCollision:
    """Investigated and rejected as a defect; kept as a standing guarantee."""

    def test_ordinal_ten_does_not_collide_with_ordinal_one(self, email_redactor):
        """``[EMAIL-1]`` is not a prefix of ``[EMAIL-11]``: the suffix decides."""
        text = " ".join("user{0}@example.com".format(i) for i in range(1, 13))
        result = email_redactor.redact(text)
        assert "[EMAIL-11]" in result.text
        assert restore(result.text, result.vault).text == text

    def test_restoring_many_labels_is_order_independent(self, email_redactor):
        text = " ".join("user{0}@example.com".format(i) for i in range(1, 30))
        result = email_redactor.redact(text)
        assert restore(result.text, result.vault).text == text


class TestNoSecretsInDiagnostics:
    """Summaries, reprs and serializers must not disclose values."""

    def test_summary_names_no_secret(self, redactor):
        result = redactor.redact("mail ada@example.com")
        assert "ada@example.com" not in result.summary()

    def test_vault_repr_names_no_secret(self, redactor):
        result = redactor.redact("mail ada@example.com")
        assert "ada@example.com" not in repr(result.vault)

    def test_entry_repr_names_no_secret(self, redactor):
        result = redactor.redact("mail ada@example.com")
        assert "ada@example.com" not in repr(result.entries[0])

    def test_as_dict_names_no_secret(self, redactor):
        import json

        from .. import as_dict

        result = redactor.redact("mail ada@example.com")
        assert "ada@example.com" not in json.dumps(as_dict(result))

    def test_summary_table_hides_values_by_default(self, redactor):
        from .. import summary_table

        result = redactor.redact("mail ada@example.com")
        assert "ada@example.com" not in summary_table(result, color=False)
        assert "ada@example.com" in summary_table(result, reveal=True, color=False)


class TestCP007And009WebTierHardening:
    """``CP-007`` and ``CP-009`` — the web tier's data handling."""

    def test_no_module_uses_eval_or_exec(self):
        """Upstream evaluated a string rebuilt from the session cookie."""
        import pathlib

        package = pathlib.Path(__file__).resolve().parent.parent
        offenders = []
        for path in sorted(package.rglob("*.py")):
            if "tests" in path.parts:
                continue
            source = path.read_text(encoding="utf-8")
            for name in ("eval(", "exec(", "pickle.loads", "os.system"):
                if name in source:
                    offenders.append("{0}: {1}".format(path.name, name))
        assert offenders == []

    def test_app_factory_refuses_an_unset_secret_key(self, monkeypatch):
        pytest.importorskip("flask")
        from .. import PolicyError, create_app

        monkeypatch.delenv("CLEANPROMPT_SECRET_KEY", raising=False)
        with pytest.raises(PolicyError) as caught:
            create_app()
        assert "CLEANPROMPT_SECRET_KEY" in str(caught.value)


class TestCP023SilentEngineDegradation:
    """
    ``CP-024`` — ``--ner`` must not succeed having run no engine.

    Notes
    -----
    **Developer notes.** Found by the suite during this rewrite's third round,
    when a stale assertion about exit codes stopped matching. The ``auto``
    engine mode degrades by design: it is the default, and a default that
    refused to run on a base installation would make the base tier unusable.
    But an explicit request is not a default. With neither spaCy nor NLTK
    usable, ``--ner`` added no detector, detected no entities and exited ``0``,
    telling the user their text had been scanned for names when nothing had
    looked at it.

    That is the same failure the whole rewrite was commissioned to remove,
    reached from the opposite direction: the first time it was a missing
    capability reported as a clean result, this time an unmeetable request
    reported as a met one. The fix separates *resolution* from *obligation*:
    :func:`~scikitplot.cleanprompt._engines.build_detectors` takes
    ``required``, and the two call sites that act on an explicit ask pass it.
    """

    def test_explicit_request_with_no_usable_engine_raises(self, monkeypatch):
        from .. import CapabilityError
        from .. import _capabilities as caps
        from .._engines import build_detectors

        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        with pytest.raises(CapabilityError) as caught:
            build_detectors(mode="auto", required=True)
        assert caught.value.tier == "ner"

    def test_the_message_names_every_engine_and_its_own_reason(self, monkeypatch):
        """A combined hint would send an NLTK user off to install spaCy."""
        from .. import CapabilityError
        from .. import _capabilities as caps
        from .._engines import build_detectors

        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        with pytest.raises(CapabilityError) as caught:
            build_detectors(mode="auto", required=True)
        message = str(caught.value)
        assert "spacy" in message and "nltk" in message
        assert "--ner-engine none" in message

    def test_structural_detection_is_named_as_unaffected(self, monkeypatch):
        """Panic is not the goal: the base tier still works and must say so."""
        from .. import CapabilityError
        from .. import _capabilities as caps
        from .._engines import build_detectors

        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        with pytest.raises(CapabilityError) as caught:
            build_detectors(mode="auto", required=True)
        assert "Structural detectors" in str(caught.value)

    def test_mode_none_is_exempt(self, monkeypatch):
        """``--ner-engine none`` is an instruction, not an unmet request."""
        from .. import _capabilities as caps
        from .._engines import build_detectors

        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        assert build_detectors(mode="none", required=True) == []

    def test_an_unrequired_call_still_degrades_quietly(self, monkeypatch):
        """The default path must stay usable on a base installation."""
        from .. import _capabilities as caps
        from .._engines import build_detectors

        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        assert build_detectors(mode="auto") == []

    def test_a_named_engine_still_fails_at_the_point_of_use(self, monkeypatch):
        """Naming an engine is already loud; ``required`` must not change it."""
        from .. import CapabilityError
        from .. import _capabilities as caps
        from .._engines import build_detectors

        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        detectors = build_detectors(mode="nltk", required=True)
        assert len(detectors) == 1
        with pytest.raises(CapabilityError):
            list(detectors[0].detect("Ada Lovelace", DEFAULT_POLICY))

    def test_the_cli_reports_it_as_unavailable_not_as_success(self, tmp_path, monkeypatch):
        from .. import _capabilities as caps
        from .._cli import main

        monkeypatch.setattr(caps, "_installed_version", lambda _n: None)
        import io
        import contextlib

        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            monkeypatch.setattr("sys.stdin", io.StringIO("Ada Lovelace"))
            status = main(["redact", "--vault", str(tmp_path / "v.json"), "--ner"])
        # 69 is EX_UNAVAILABLE: "install something", not "it broke".
        assert status == 69
        assert "hint:" in err.getvalue()


class TestCP024EntityDetectorIdentifiedByName:
    """
    ``CP-025`` — a second engine must not be invisible to the diagnostics.

    Notes
    -----
    **Developer notes.** ``diagnose`` decided whether entity detection was
    running by testing ``detector.name.startswith("ner:")``, which is the
    spaCy detector's naming scheme. The NLTK detector is named ``nltk``, so a
    registry actively detecting names reported the high-severity blind spot
    "names are NOT being detected" — a warning raised against working
    software.

    A false alarm is not the harmless direction of this error. The banner
    exists so that a real gap is believed; one that cries wolf whenever the
    lighter engine is used trains people to dismiss it, which costs exactly
    the case it was built for.

    The test is now on the *kind*, which both detectors declare as ``"NE"``
    and which is part of the detector protocol rather than a naming habit.
    """

    def test_an_nltk_registry_is_not_reported_as_blind(self):
        from .. import diagnose
        from .._nltk import nltk_detector

        registry = default_registry(kinds=["EMAIL"])
        registry.add(nltk_detector())
        report = diagnose(DEFAULT_POLICY, registry)
        assert report.ner_active is True
        assert not any("named entities" in spot.category for spot in report.blind_spots)

    def test_an_empty_registry_is_still_reported_as_blind(self):
        from .. import diagnose

        report = diagnose(DEFAULT_POLICY, default_registry(kinds=["EMAIL"]))
        assert report.ner_active is False
        assert any("named entities" in spot.category for spot in report.blind_spots)

    def test_every_entity_detector_declares_the_same_kind(self):
        """The property the diagnosis now relies on, asserted where it is set."""
        from .._nltk import nltk_detector

        kinds = {nltk_detector().kind}
        from .. import _capabilities as caps

        if caps.probe("ner").status.value != "ABSENT":
            from .._ner import spacy_detector

            kinds.add(spacy_detector().kind)
        assert kinds == {"NE"}

    def test_the_remedy_matches_the_installed_engines(self):
        """It hard-coded a model name that the default had already moved off."""
        from .._diagnostics import _entity_remedy

        remedy = _entity_remedy()
        assert "en_core_web_lg" not in remedy
        assert "--ner" in remedy or "pip install" in remedy


class TestCP025BrokenTierReportedAsDetectorCrash:
    """
    ``CP-026`` — installed-but-unimportable is ``BROKEN``, not a crash.

    Notes
    -----
    **Developer notes.** Found by the import-isolation probe once the probe was
    strengthened. Capability probing reads distribution *metadata*, so a
    half-finished upgrade, a wheel built for another interpreter, or a local
    file shadowing the package all satisfy ``require`` and then fail at
    ``import``. The raw :class:`ImportError` escaped into the detector's broad
    handler and arrived as ``detector 'ner:en_core_web_sm' failed: ImportError:
    …`` — a message that sends the reader to audit this submodule for a fault
    that is in their environment.

    ``BROKEN`` already existed in the capability vocabulary for exactly this
    state; nothing was mapping onto it. The remedy uses
    ``--force-reinstall`` deliberately: a plain install is a no-op against a
    distribution pip already believes is satisfied, which is the very state
    being reported.
    """

    @staticmethod
    def _unimportable(monkeypatch, name, version):
        """
        Make ``name`` claim to be installed, in range, and refuse to import.

        The version must be inside the tier's supported range: an out-of-range
        one is ``INCOMPATIBLE`` and ``require`` rejects it before the import is
        ever attempted, so the probe would never reach the state under test.
        """
        import builtins

        from .. import _capabilities as caps, _ner, _nltk

        # Both engines cache their loaded machinery, and the cache is consulted
        # before the import. Left populated by an earlier live test, these
        # assertions pass in isolation and fail in a full run — so the caches
        # are emptied here rather than the ordering being relied upon.
        monkeypatch.setattr(_ner, "_PIPELINES", {})
        monkeypatch.setattr(_nltk, "_RESOURCES", {})
        monkeypatch.setattr(caps, "_installed_version", lambda _n: version)
        real = builtins.__import__

        def guard(module, *args, **kwargs):
            if module.split(".")[0] == name:
                raise ImportError("blocked: " + module)
            return real(module, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", guard)

    def test_spacy_reports_broken_not_a_detector_failure(self, monkeypatch):
        from .. import CapabilityError
        from .._ner import spacy_detector

        detector = spacy_detector()
        self._unimportable(monkeypatch, "spacy", "3.7.2")
        with pytest.raises(CapabilityError) as caught:
            list(detector.detect("Ada Lovelace", DEFAULT_POLICY))
        assert caught.value.status == "BROKEN"
        assert "--force-reinstall" in caught.value.install_hint

    def test_nltk_reports_broken_not_a_detector_failure(self, monkeypatch):
        from .. import CapabilityError
        from .._nltk import nltk_detector

        detector = nltk_detector()
        self._unimportable(monkeypatch, "nltk", "3.9")
        with pytest.raises(CapabilityError) as caught:
            list(detector.detect("Ada Lovelace", DEFAULT_POLICY))
        assert caught.value.status == "BROKEN"
        assert "--force-reinstall" in caught.value.install_hint

    def test_the_message_distinguishes_broken_from_absent(self, monkeypatch):
        """'Install it' is wrong advice for something already installed."""
        from .. import CapabilityError
        from .._nltk import nltk_detector

        detector = nltk_detector()
        self._unimportable(monkeypatch, "nltk", "3.9")
        with pytest.raises(CapabilityError) as caught:
            list(detector.detect("Ada Lovelace", DEFAULT_POLICY))
        assert "cannot be imported" in str(caught.value)

    def test_it_is_still_a_capability_error_for_existing_handlers(self, monkeypatch):
        """CapabilityError subclasses ImportError; that must keep holding."""
        from .._nltk import nltk_detector

        detector = nltk_detector()
        self._unimportable(monkeypatch, "nltk", "3.9")
        with pytest.raises(ImportError):
            list(detector.detect("Ada Lovelace", DEFAULT_POLICY))


class TestCP026ChunkerRebuiltPerSentence:
    """
    ``CP-027`` — the NLTK chunker must be built once, not once per sentence.

    Notes
    -----
    **Developer notes.** Found while writing the live-engine probe, which ran
    two orders of magnitude slower than expected. :func:`nltk.ne_chunk`
    constructs a fresh ``Maxent_NE_Chunker`` on every call and this detector
    calls it once per sentence. Measured on NLTK 3.10.3: constructing the
    chunker takes 460 ms, parsing a sentence with it takes 1 ms. A
    ten-sentence prompt therefore spent 4.6 seconds rebuilding a model that
    never changes.

    The consequence is not only slowness. The lighter engine exists for the
    machine that cannot take spaCy; being 200x slower than the heavy one
    inverts the reason to choose it, and in the web interface a multi-second
    redaction reads as a hang, which invites a second submission of the same
    unredacted text.

    The cache is keyed by nothing because the chunker is stateless with
    respect to the text — the same property that makes caching safe is the one
    :class:`TestCP003SharedMutableState` guards for the pipeline as a whole.

    These tests measure the live engine, so they need NLTK and its data; where
    the tier is not usable they skip with the capability's own message rather
    than fail (measured in a bare environment, round 12).
    """

    @pytest.fixture(autouse=True)
    def _live_nltk(self):
        from .. import CapabilityError
        from .._nltk import nltk_detector

        try:
            list(nltk_detector().detect("Ada Lovelace.", DEFAULT_POLICY))
        except CapabilityError as error:
            pytest.skip(str(error))

    def test_the_chunker_is_cached_across_documents(self):
        from .._nltk import _RESOURCES, nltk_detector

        detector = nltk_detector()
        list(detector.detect("Ada Lovelace went to London.", DEFAULT_POLICY))
        first = _RESOURCES.get("chunker")
        list(detector.detect("Charles Babbage went to Paris.", DEFAULT_POLICY))
        assert _RESOURCES.get("chunker") is first

    def test_a_multi_sentence_document_is_not_quadratic_in_sentences(self):
        """One rebuild per sentence showed up as a linear cost in sentences."""
        import time

        from .._nltk import nltk_detector

        detector = nltk_detector()
        one = "Ada Lovelace went to London. "
        list(detector.detect(one, DEFAULT_POLICY))  # pay the one-off build

        start = time.monotonic()
        list(detector.detect(one * 20, DEFAULT_POLICY))
        elapsed = time.monotonic() - start
        # Twenty rebuilds cost ~9 s; twenty parses cost ~0.03 s. A second is
        # far above the fixed cost and far below the defect, so this fails on
        # a regression without being sensitive to machine speed.
        assert elapsed < 1.0, "20 sentences took {0:.2f}s".format(elapsed)

    def test_the_fallback_still_detects(self, monkeypatch):
        """No pre-built chunker is a slow path, never a broken one."""
        from .. import _nltk

        monkeypatch.setattr(_nltk, "_RESOURCES", {})
        monkeypatch.setattr(_nltk, "_build_chunker", lambda: None)
        spans = list(_nltk.nltk_detector().detect("Ada Lovelace wrote.", DEFAULT_POLICY))
        assert any(span.kind == "PERSON" for span in spans)

    def test_the_fallback_produces_the_same_spans(self, monkeypatch):
        from .. import _nltk

        text = "Ada Lovelace wrote to Charles Babbage from London."
        fast = [(s.start, s.end, s.kind) for s in _nltk.nltk_detector().detect(text, DEFAULT_POLICY)]

        monkeypatch.setattr(_nltk, "_RESOURCES", {})
        monkeypatch.setattr(_nltk, "_build_chunker", lambda: None)
        slow = [(s.start, s.end, s.kind) for s in _nltk.nltk_detector().detect(text, DEFAULT_POLICY)]
        assert fast == slow

    def test_an_unbuildable_chunker_does_not_disable_the_engine(self, monkeypatch):
        """Returning None must be a fallback, not a refusal to run."""
        from .. import _nltk

        monkeypatch.setattr(_nltk, "_RESOURCES", {})

        def explode():
            raise RuntimeError("no chunker here")

        monkeypatch.setattr("nltk.chunk.ne_chunker", explode)
        assert _nltk._build_chunker() is None


class TestCP027And028PunctuationBoundaries:
    """
    ``CP-028`` and ``CP-029`` — punctuation around a value, handled both ways.

    Notes
    -----
    **Developer notes.** Found by checking every pattern's own examples in
    every ordinary sentence position, which is a thing the suite had never
    done: each pattern was tested against its examples in isolation, and in
    isolation there is no punctuation.

    Two opposite errors, both from the same blind spot.

    ``CP-028`` is a **leak**. ``IPV4``'s guard was ``(?![\\w.])`` — no word
    character and no dot may follow. That correctly rejects ``1.2.3.4.5``, and
    it also rejects ``192.168.1.10.`` at the end of a sentence, which is where
    addresses most often appear. Sentence-final addresses were not detected at
    all and went to the model in the clear. ``IPV6`` had the same guard and the
    same hole. The fix narrows the guard to what it was actually for: a
    trailing dot disqualifies the match only when a digit follows it.

    ``CP-029`` is **over-capture**. ``URL``'s trailing classes admit ``.``,
    ``,`` and ``)``, all legal inside a URL, so ``see https://example.com/x.``
    stored ``https://example.com/x.`` as the secret. The text still round-trips
    — the dot goes back where it came from — so no test caught it, but the
    value in the vault is wrong and the model receives a sentence with no end
    to it.

    The class-level test below is the real gate: it runs every pattern's own
    ``examples_yes`` through twelve sentence positions, so the next pattern
    added is checked the same way without anyone remembering to.
    """

    TEMPLATES = (
        "{0}", "see {0}", "see {0} now", "see {0}.", "see {0},", "see {0};",
        "see {0}!", "see {0})", "see {0}\n", "{0}.", "({0})", "x {0}. y",
    )

    def _detector(self, kind):
        from .._detectors import RegexDetector

        return RegexDetector(get_pattern(kind))

    def test_every_example_is_found_in_every_sentence_position(self):
        from .. import PATTERNS

        missed = []
        for kind in sorted(PATTERNS):
            spec = get_pattern(kind)
            detector = self._detector(kind)
            for example in spec.examples_yes:
                for template in self.TEMPLATES:
                    text = template.format(example)
                    found = [s.text for s in detector.detect(text, DEFAULT_POLICY)]
                    if example not in found:
                        missed.append((kind, template, example, found))
        assert missed == [], "not found in context: {0}".format(missed)

    def test_no_example_captures_the_surrounding_punctuation(self):
        from .. import PATTERNS

        over = []
        for kind in sorted(PATTERNS):
            spec = get_pattern(kind)
            detector = self._detector(kind)
            for example in spec.examples_yes:
                for template in self.TEMPLATES:
                    text = template.format(example)
                    for span in detector.detect(text, DEFAULT_POLICY):
                        if span.text != example and example in span.text:
                            over.append((kind, template, span.text))
        assert over == [], "captured punctuation: {0}".format(over)

    def test_every_counterexample_is_still_rejected(self):
        """The fix must not have been bought by loosening the guards."""
        from .. import PATTERNS

        accepted = []
        for kind in sorted(PATTERNS):
            spec = get_pattern(kind)
            detector = self._detector(kind)
            for example in spec.examples_no:
                for span in detector.detect(example, DEFAULT_POLICY):
                    if span.text == example:
                        accepted.append((kind, example))
        assert accepted == []

    def test_a_sentence_final_ipv4_is_redacted(self, redactor):
        """The reported shape: an address ending a sentence."""
        result = redactor.redact("he mailed a@b.co from 192.168.1.10.")
        assert "192.168.1.10" not in result.text
        assert result.text.endswith(".")

    def test_a_longer_dotted_run_is_still_rejected(self, redactor):
        """The guard that CP-028 narrowed must still do its original job."""
        assert _matches("IPV4", "1.2.3.4.5") == []

    def test_a_sentence_final_ipv6_is_redacted(self, redactor):
        text = "the host is fe80:1:2:3:4:5:6:7."
        result = redactor.redact(text)
        assert "fe80:1:2:3:4:5:6:7" not in result.text
        assert restore(result.text, result.vault).text == text

    def test_a_url_does_not_swallow_the_full_stop(self, redactor):
        result = redactor.redact("see https://example.com/path?q=1#frag.")
        assert result.text.endswith(".")
        assert result.entries[0].original == "https://example.com/path?q=1#frag"

    def test_a_bare_host_url_does_not_swallow_the_full_stop(self, redactor):
        result = redactor.redact("see https://example.com.")
        assert result.entries[0].original == "https://example.com"

    def test_url_round_trip_is_still_exact(self, redactor):
        for text in (
            "see https://example.com/path?q=1#frag.",
            "see (https://example.com/a) here",
            "https://example.com/p?q=1&r=2!",
        ):
            result = redactor.redact(text)
            assert restore(result.text, result.vault).text == text


class TestCP030TextCouldNotBeGivenDirectly:
    """
    ``CP-030`` — the obvious way to pass text was rejected without a remedy.

    Notes
    -----
    **Developer notes.** Reported from a real session. A user ran
    ``cleanprompt inspect --ner`` followed by a paragraph and got, in order:
    ``bash: syntax error near unexpected token `('`` from the shell, then
    ``Error: No such option '-M'`` when they tried ``-"text"``, then
    ``Error: Got unexpected extra argument`` when they quoted it properly.

    That last one is ours, and it is the worst of the three: the text is named
    as the problem, and nothing says what to do instead. The commands accepted
    ``--in PATH`` and standard input, both documented in a single line of
    ``--help`` that a person mid-paste does not read.

    Two things were wrong and both are fixed. The obvious invocation now works,
    because a variadic positional costs nothing and matches what everyone
    types. And a command that falls through to standard input on a terminal
    says so, with the other routes in, instead of sitting silent while the user
    concludes it has hung.

    The remaining failure is the shell's, and no program can fix it — a shell
    consumes ``(`` before this code runs. What we can do is document the
    heredoc, which is the form that survives arbitrary prose, and the CLI's own
    hint now prints it.
    """

    def _run(self, args, stdin="", tty=False):
        import io

        from .._cli import main

        class _Tty(io.StringIO):
            def isatty(self):
                return True

        stream = (_Tty if tty else io.StringIO)(stdin)
        out, err = io.StringIO(), io.StringIO()
        status = main(args, stdin=stream, stdout=out, stderr=err)
        return status, out.getvalue(), err.getvalue()

    def test_the_invocation_that_was_rejected_now_works(self):
        status, out, _ = self._run(["inspect", "Mail ada@example.com about it"])
        assert status == 0
        assert "[EMAIL-1]" in out

    def test_every_text_command_accepts_it(self, tmp_path):
        assert self._run(["inspect", "mail a@b.co"])[0] == 0
        assert self._run(["scan", "mail a@b.co"])[0] == 3
        assert self._run(["roundtrip", "mail a@b.co"])[0] == 0
        vault = str(tmp_path / "v.json")
        assert self._run(["redact", "--vault", vault, "mail a@b.co"])[0] == 0
        assert self._run(["restore", "--vault", vault, "see [EMAIL-1]"])[0] == 0

    def test_a_terminal_is_never_left_guessing(self):
        _, _, err = self._run(["inspect"], stdin="mail a@b.co", tty=True)
        assert "reading from standard input" in err
        assert "Ctrl-D" in err

    def test_the_hint_shows_the_heredoc_that_survives_prose(self):
        """The form that works for the paragraph that started this."""
        _, _, err = self._run(["inspect"], stdin="x", tty=True)
        assert "<<'END'" in err

    def test_two_sources_at_once_is_refused_not_guessed(self, tmp_path):
        path = tmp_path / "p.txt"
        path.write_text("from the file", encoding="utf-8")
        status, _, err = self._run(["inspect", "--in", str(path), "direct"])
        assert status == 1
        assert "one or the other" in err


class TestCP031DecodeWasNeverDemonstrated:
    """
    ``CP-031`` — the half people doubt was the half nothing showed.

    Notes
    -----
    **Developer notes.** ``redact`` and ``restore`` are separate commands with
    a vault file between them, which is right for real work: the reply may come
    back minutes or days later, in another process. It also meant that seeing
    the values return took two commands, a file path and an understanding of
    what the vault is for — before which the tool looks like it destroys your
    text.

    ``roundtrip`` shows all five stages in one command and writes nothing to
    disk. The stand-in reply is a fixed string, and is labelled as one on every
    run: presenting generated text as a model's answer would be a lie told by a
    tool whose whole purpose is trust.
    """

    def _run(self, args):
        import io

        from .._cli import main

        out, err = io.StringIO(), io.StringIO()
        status = main(args, stdin=io.StringIO(""), stdout=out, stderr=err)
        return status, out.getvalue(), err.getvalue()

    def test_one_command_shows_redaction_and_restoration(self):
        status, out, _ = self._run(["roundtrip", "Mail ada@example.com"])
        assert status == 0
        assert "[EMAIL-1]" in out  # it was removed
        assert "ada@example.com" in out.split("5 ·")[1]  # and it came back

    def test_the_value_is_absent_from_what_would_be_sent(self):
        _, out, _ = self._run(["roundtrip", "Mail ada@example.com"])
        assert "ada@example.com" not in out.split("2 ·")[1].split("3 ·")[0]

    def test_the_stand_in_reply_never_poses_as_a_model(self):
        _, out, _ = self._run(["roundtrip", "Mail ada@example.com"])
        assert "NOT from a model" in out

    def test_a_real_reply_replaces_it(self):
        _, out, _ = self._run(
            ["roundtrip", "Mail ada@example.com", "--reply", "sent to [EMAIL-1]"]
        )
        assert "NOT from a model" not in out
        assert "sent to ada@example.com" in out

    def test_nothing_is_written_to_disk(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._run(["roundtrip", "Mail ada@example.com"])
        assert list(tmp_path.iterdir()) == []


class TestCP032OptionAbbreviationDivergence:
    """
    ``CP-032`` — an abbreviated long option must not depend on the frontend.

    Notes
    -----
    **Developer notes.** argparse accepts any unambiguous prefix of a long
    option by default; click accepts none. So ``--form json`` reached
    ``--format`` on a machine without click and failed with exit 2 on a machine
    with it. Same command line, same version, different answer — the ``CP-021``
    class, found by probing the option grammar after a user asked whether
    ``--`` was handled.

    argparse conforms, as it did for ``CP-021``, because the strict behaviour
    is the one that can be relied on. An abbreviation is a latent break even on
    a single machine: it is unambiguous only until a new option shares its
    prefix, so a script that worked for a year fails on an upgrade that added a
    feature it never used.
    """

    def _status(self, frontend, argv):
        import io

        from .._cli import main

        return main(
            ["--frontend", frontend, *argv],
            stdin=io.StringIO(""),
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )

    def test_the_reported_abbreviation_fails_in_both(self):
        argv = ["inspect", "--form", "json", "x"]
        assert self._status("argparse", argv) == 2
        assert self._status("click", argv) == 2

    def test_the_full_spelling_succeeds_in_both(self):
        argv = ["inspect", "--format", "json", "x"]
        assert self._status("argparse", argv) == 0
        assert self._status("click", argv) == 0

    def test_an_abbreviated_flag_fails_in_both(self):
        for argv in (["inspect", "--rev", "x"], ["inspect", "--no-sug", "x"]):
            assert self._status("argparse", argv) == 2
            assert self._status("click", argv) == 2


class TestCP033StrayOptionSwallowedAsText:
    """
    ``CP-033`` — a mistyped option must never be redacted as if it were text.

    Notes
    -----
    **Developer notes.** argparse treats any token containing a space as a
    positional, whatever it starts with, so ``inspect "--secret is a@b.co"``
    was accepted as text and quietly processed. Click rejected the same command
    line as an unknown option.

    The divergence matters, and the direction matters more. A tool that decides
    what leaves the machine must not silently do something other than what was
    asked: a user who mistyped ``--secrets`` as ``--secret`` would have had the
    flag itself treated as their prompt, with no indication that the option
    they meant was never applied.

    argparse decides this inside ``_parse_optional``, which is private and has
    moved between releases, so the conforming is a check after parsing rather
    than an override: a dash-leading positional is refused unless a ``--``
    delimiter authorised it. Past a ``--`` nothing is checked, because that is
    precisely what the delimiter means.
    """

    def _run(self, frontend, argv):
        import io

        from .._cli import main

        out, err = io.StringIO(), io.StringIO()
        status = main(
            ["--frontend", frontend, *argv],
            stdin=io.StringIO(""),
            stdout=out,
            stderr=err,
        )
        return status, out.getvalue(), err.getvalue()

    def test_a_stray_option_is_refused_by_both(self):
        argv = ["inspect", "--secret is a@b.co"]
        assert self._run("argparse", argv)[0] == 2
        assert self._run("click", argv)[0] == 2

    def test_it_is_never_silently_treated_as_text(self):
        _, out, _ = self._run("argparse", ["inspect", "--secret is a@b.co"])
        assert "[EMAIL-1]" not in out

    def test_the_message_names_the_delimiter(self):
        _, _, err = self._run("argparse", ["inspect", "--secret is a@b.co"])
        assert "--" in err
        assert "end the options" in err

    def test_the_delimiter_makes_it_text_again(self):
        """The user says it is text, and then it is text."""
        for frontend in ("argparse", "click"):
            status, out, _ = self._run(
                frontend,
                ["roundtrip", "--format", "json", "--", "--secret is a@b.co"],
            )
            assert status == 0
            import json as _json

            assert _json.loads(out)["original"] == "--secret is a@b.co"

    def test_a_lone_dash_is_still_a_valid_positional(self):
        """'-' conventionally means standard input and must not be refused."""
        import io

        from .._cli import main

        status = main(
            ["--frontend", "argparse", "inspect", "--in", "-"],
            stdin=io.StringIO("mail a@b.co"),
            stdout=io.StringIO(),
            stderr=io.StringIO(),
        )
        assert status == 0


class TestCP034To036VaultDefaultsAndCleanCommand:
    """
    ``CP-034`` … ``CP-036`` — the vault you never have to name.

    Notes
    -----
    **Developer notes.** Three reported frictions with one shape: the tool
    worked, and the shortest path through it was not reachable.

    ``CP-034`` — ``--vault`` was required, so the simplest useful command
    carried a path the user had to invent and keep consistent. The default now
    resolves to the platform state directory. Not the working directory: a
    vault holds the removed values in clear text, this tool is used inside
    checkouts, and a secrets file in the working directory is one ``git add .``
    from a commit. That is the worst outcome this submodule can produce, so the
    location is a security decision rather than a convenience one.

    ``CP-035`` — every run rewrote the vault, so numbering restarted and the
    same address could be ``[EMAIL-1]`` in one turn and ``[EMAIL-2]`` in
    another. A reply quoting an earlier placeholder then restored to the wrong
    value. Append mode seeds the redactor from the existing vault, which the
    engine has supported since round one and the CLI never used.

    ``CP-036`` — ``redact --vault v.json 2>/dev/null`` already produced a bare
    clean prompt. Nobody found it, and the reported session shows a user
    reaching for ``inspect`` instead. ``clean`` is the same pipeline with the
    defaults someone pasting into a chat window actually wants.
    """

    def _run(self, argv, stdin="", env=None, monkeypatch=None):
        import io

        from .._cli import main

        out, err = io.StringIO(), io.StringIO()
        status = main(argv, stdin=io.StringIO(stdin), stdout=out, stderr=err)
        return status, out.getvalue(), err.getvalue()

    @pytest.fixture()
    def state(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CLEANPROMPT_VAULT", raising=False)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        return tmp_path / "state" / "cleanprompt" / "vault.json"

    def test_the_default_vault_is_outside_the_working_directory(self, tmp_path, monkeypatch):
        """The whole point: 'git add .' must not be able to reach it."""
        from .._cli import default_vault_path

        monkeypatch.delenv("CLEANPROMPT_VAULT", raising=False)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        monkeypatch.chdir(tmp_path)
        resolved = default_vault_path()
        assert not resolved.startswith(str(tmp_path) + "/vault")
        assert "state" in resolved

    def test_the_whole_loop_needs_no_path(self, state):
        """clean, paste, restore — and no argument typed anywhere."""
        _, safe, _ = self._run(["clean", "-q", "mail ada@example.com"])
        assert safe.strip() == "mail [EMAIL-1]"
        _, back, _ = self._run(["restore", "-q", "I mailed [EMAIL-1]."])
        assert back.strip() == "I mailed ada@example.com."

    def test_clean_puts_nothing_but_the_prompt_on_stdout(self, state):
        _, out, _ = self._run(["clean", "mail ada@example.com from 192.168.1.10."])
        assert out == "mail [EMAIL-1] from [IPV4-1].\n"

    def test_a_placeholder_survives_across_turns(self, state):
        self._run(["clean", "-q", "mail ada@example.com"])
        self._run(["clean", "-q", "cc bob@example.com"])
        _, out, _ = self._run(["clean", "-q", "remind ada@example.com"])
        assert out.strip() == "remind [EMAIL-1]"

    def test_no_placeholder_maps_to_two_values(self, state):
        import json as _json

        for text in ("mail ada@example.com", "cc bob@example.com", "and eve@example.com"):
            self._run(["clean", "-q", text])
        entries = _json.loads(state.read_text(encoding="utf-8"))["entries"]
        assert len(set(entries.values())) == len(entries) == 3

    def test_the_vault_is_not_world_readable(self, state):
        import os
        import stat as _stat

        self._run(["clean", "-q", "mail ada@example.com"])
        assert _stat.S_IMODE(os.stat(state).st_mode) & 0o077 == 0

    def test_writing_the_vault_is_never_silent(self, state):
        """A file of secrets appearing with no mention would be a side effect."""
        _, _, err = self._run(["clean", "mail ada@example.com"])
        assert "vault:" in err

    def test_appending_onto_an_index_less_vault_is_refused(self, tmp_path):
        import json as _json

        old = tmp_path / "old.json"
        old.write_text(
            _json.dumps(
                {
                    "format": 1,
                    "encrypted": False,
                    "entries": {"[EMAIL-1]": "old@example.com"},
                    "grammar_fingerprint": None,
                }
            ),
            encoding="utf-8",
        )
        status, _, err = self._run(
            ["clean", "--vault", str(old), "--vault-mode", "append", "a@b.co"]
        )
        assert status == 1
        assert "collide" in err


class TestCP041EntitySpanWithAnUnmatchedBracket:
    """
    ``CP-041`` — an entity span must not carry an unmatched bracket.

    Notes
    -----
    **Developer notes.** Measured, not supposed. On ``Mustafa Kemal
    Atatürk[e] (c. 1881)`` spaCy returns the person as ``'Mustafa Kemal
    Atatürk[e'``: it swallows the opening bracket and the footnote letter and
    leaves the closing one behind.

    Two things went wrong at once, and the visible one was the lesser. The
    prompt carried ``[PERSON-1]]``, a malformed placeholder — untidy, and more
    likely to be rewritten by the model, which is the failure the lenient
    matcher then has to repair. The worse one was silent: the vault recorded
    the person's name *as* ``Mustafa Kemal Atatürk[e``, so a restoration into
    any other text would have produced that string as somebody's name.

    The root cause is an asymmetry. Every structural pattern carries a
    validator — Luhn for a card, mod-97 for an IBAN, octet ranges for a dotted
    quad — and entity spans, which come from a third-party model, were taken
    verbatim with nothing checked at all.
    """

    def _spans(self, text, mode="spacy"):
        from .. import build_detectors

        return list(build_detectors(mode=mode)[0].detect(text, DEFAULT_POLICY))

    @pytest.fixture(autouse=True)
    def _needs_an_engine(self):
        from .. import _capabilities as caps

        if not caps.probe("ner").available:
            pytest.skip("the 'ner' tier is unavailable")

    def test_the_reported_span_is_trimmed(self):
        text = "Mustafa Kemal Atatürk[e] founded it"
        people = [s for s in self._spans(text) if s.kind == "PERSON"]
        assert people
        assert "[" not in people[0].text
        assert people[0].text == "Mustafa Kemal Atatürk"

    def test_the_vault_records_the_real_name(self):
        from .. import Redactor, build_detectors, default_registry

        registry = default_registry()
        registry.add(build_detectors(mode="spacy")[0])
        result = Redactor(registry=registry).redact("Mustafa Kemal Atatürk[e] led it")
        assert all("[" not in entry.original for entry in result.entries)

    def test_the_placeholder_is_well_formed(self):
        from .. import Redactor, build_detectors, default_registry

        registry = default_registry()
        registry.add(build_detectors(mode="spacy")[0])
        result = Redactor(registry=registry).redact("Mustafa Kemal Atatürk[e] led it")
        assert "[PERSON-1]]" not in result.text

    def test_the_round_trip_is_still_exact(self):
        from .. import Redactor, build_detectors, default_registry

        registry = default_registry()
        registry.add(build_detectors(mode="spacy")[0])
        text = "Mustafa Kemal Atatürk[e] (c. 1881[f]) founded the Republic of Turkey."
        result = Redactor(registry=registry).redact(text)
        assert restore(result.text, result.vault).text == text


class TestCP041TrimRule:
    """The trimming rule itself, without needing an engine installed."""

    def test_an_unmatched_opener_truncates_the_span(self):
        from .._engines import trim_entity_span

        text = "Mustafa Kemal Atatürk[e] founded it"
        start, end = trim_entity_span(text, 0, 23)
        assert text[start:end] == "Mustafa Kemal Atatürk"

    def test_balanced_brackets_are_left_alone(self):
        """Acme (Europe) Ltd is not malformed and must not be shortened."""
        from .._engines import trim_entity_span

        text = "Acme (Europe) Ltd is here"
        start, end = trim_entity_span(text, 0, 17)
        assert text[start:end] == "Acme (Europe) Ltd"

    def test_an_unmatched_closer_moves_the_start(self):
        from .._engines import trim_entity_span

        text = "e] Acme Corporation"
        start, end = trim_entity_span(text, 0, len(text))
        assert text[start:end] == "Acme Corporation"

    def test_a_span_of_only_brackets_collapses(self):
        from .._engines import trim_entity_span

        text = "[[[ ]]]"
        start, end = trim_entity_span(text, 0, 3)
        assert start == end  # the caller drops it

    def test_a_trailing_full_stop_is_not_trimmed(self):
        """Inc. and St. end in one legitimately; guessing is not this tool's job."""
        from .._engines import trim_entity_span

        text = "Acme Inc. filed"
        start, end = trim_entity_span(text, 0, 9)
        assert text[start:end] == "Acme Inc."

    def test_a_plain_span_is_unchanged(self):
        from .._engines import trim_entity_span

        text = "the Republic of Turkey"
        assert trim_entity_span(text, 0, len(text)) == (0, len(text))


class TestCP042ModelRewrittenPlaceholders:
    """
    ``CP-042`` — a placeholder the model rewrote must still restore.

    Notes
    -----
    **Developer notes — the root cause is that the channel is not lossless.**

    Every invariant this submodule proves holds across the parts it controls:
    round trips are exact, spans index their own text, nothing leaks.
    Restoration is the exception, because between redaction and restoration the
    text passes through a language model, and the design treated that hop as if
    it returned the bytes it was given.

    It does not. Measured against realistic replies, ``[EMAIL-1]`` comes back
    lower-cased, with an underscore or a space for the hyphen, with a Unicode
    dash, with the brackets escaped for Markdown, and wrapped across a line.
    Exact matching restored the first spelling and missed every other one —
    and the report said ``restored 0 placeholder(s)`` with exit status 0, so a
    user could paste a half-restored answer onward without noticing.

    The fix has three parts because one is not enough. The grammar now
    recognises that bounded set of rewrites; a lenient match is acted on only
    when it resolves to a label the vault actually holds, so ordinary prose
    like ``[note 2]`` is untouched; and every repair is reported rather than
    performed quietly.
    """

    @staticmethod
    def _vault():
        return Redactor().redact("Mail ada@example.com and bob@example.com").vault

    @pytest.mark.parametrize(
        "reply",
        [
            "I mailed [EMAIL-1].",
            "I mailed [email-1].",
            "I mailed [EMAIL_1].",
            "I mailed [EMAIL 1].",
            "I mailed [EMAIL‑1].",
            r"I mailed \[EMAIL-1\].",
            "I mailed [EMAIL-\n1].",
            "I mailed **[EMAIL-1]**.",
            "I mailed `[EMAIL-1]`.",
        ],
    )
    def test_every_measured_rewrite_restores(self, reply):
        assert "ada@example.com" in restore(reply, self._vault()).text

    def test_a_repair_is_reported(self):
        result = restore("I mailed [EMAIL_1].", self._vault())
        assert result.repaired == (("[EMAIL_1]", "[EMAIL-1]"),)

    def test_an_exact_match_is_not_reported_as_repaired(self):
        assert restore("I mailed [EMAIL-1].", self._vault()).repaired == ()

    def test_ordinary_prose_is_untouched(self):
        """A lenient match that resolves to nothing must change nothing."""
        text = "See [note 2] and [1] and [figure 3] there."
        result = restore(text, self._vault())
        assert result.text == text
        assert result.unknown == ()

    def test_a_genuinely_unknown_label_is_still_reported(self):
        result = restore("See [EMAIL-9].", self._vault())
        assert result.unknown == ("[EMAIL-9]",)

    def test_exact_mode_restores_the_old_behaviour(self):
        result = restore("I mailed [EMAIL_1].", self._vault(), lenient=False)
        assert "ada@example.com" not in result.text

    def test_a_silent_nothing_is_now_loud(self):
        from .. import restoration_note

        note = restoration_note(restore("Nothing here.", self._vault()))
        assert "nothing was restored" in note

    def test_two_different_rewrites_in_one_reply(self):
        result = restore("[EMAIL_1] and [EMAIL 2] both.", self._vault())
        assert "ada@example.com" in result.text
        assert "bob@example.com" in result.text
        assert len(result.repaired) == 2

    def test_a_rewritten_label_cannot_resolve_to_the_wrong_entry(self):
        """Normalisation must preserve the ordinal, not just the category."""
        result = restore("[EMAIL_2] only.", self._vault())
        assert "bob@example.com" in result.text
        assert "ada@example.com" not in result.text


class TestCP044EscapedSeparator:
    """
    ``CP-044`` — a Markdown-escaped separator must restore like an escaped
    bracket.

    Notes
    -----
    **Developer notes — the root cause is a list checked against itself.**

    ``CP-042`` enumerated the shapes a model returns a label in and admitted
    each one. Two of those shapes compose: a model writing ``[EMAIL_1]`` inside
    Markdown escapes the underscore, because an unescaped one opens emphasis.
    The enumeration never said so, and the implementation was read back against
    the enumeration rather than against the behaviour, so ``[EMAIL\\_1]``
    matched nothing and restored nothing.

    The failure mode is the one ``CP-042`` exists to prevent, arriving by a
    different door: a reply visibly full of placeholders, reported as
    ``restored 0 placeholder(s)`` at exit status 0.

    The fix is one allowance — an optional backslash before the separator,
    exactly the allowance the delimiters already had. What keeps it honest is
    unchanged and is asserted here: a lenient match is acted on only when it
    resolves to a label the vault holds, so this widens the set of *labels*
    recognised and the set of *prose* recognised by nothing.

    **User notes.** Nothing to do. A reply that previously came back with
    ``[EMAIL\\_1]`` still in it now restores, and the repair is named in the
    summary like every other one.
    """

    @staticmethod
    def _vault():
        return Redactor().redact("Mail ada@example.com and bob@example.com").vault

    @pytest.mark.parametrize(
        "reply",
        [
            r"I mailed [EMAIL\_1].",
            r"I mailed [email\_1].",
            r"I mailed \[EMAIL\_1\].",
            r"I mailed [EMAIL\-1].",
            r"I mailed **[EMAIL\_1]**.",
        ],
    )
    def test_an_escaped_separator_restores(self, reply):
        assert "ada@example.com" in restore(reply, self._vault()).text

    def test_the_repair_is_reported_with_the_escape_intact(self):
        result = restore(r"I mailed [EMAIL\_1].", self._vault())
        assert result.repaired == ((r"[EMAIL\_1]", "[EMAIL-1]"),)

    def test_the_ordinal_still_selects_the_right_entry(self):
        result = restore(r"[EMAIL\_2] only.", self._vault())
        assert "bob@example.com" in result.text
        assert "ada@example.com" not in result.text

    @pytest.mark.parametrize(
        "prose",
        [r"See [see\_4] there.", r"See [note\_2] there.", r"See [fig\-3] there."],
    )
    def test_escaped_prose_lookalikes_are_untouched(self, prose):
        """The escape must not become a licence to rewrite the model's words."""
        result = restore(prose, self._vault())
        assert result.text == prose
        assert result.unknown == ()

    def test_exact_mode_still_refuses(self):
        result = restore(r"I mailed [EMAIL\_1].", self._vault(), lenient=False)
        assert "ada@example.com" not in result.text

    def test_the_documented_shape_list_matches_the_implementation(self):
        """
        The regression that produced this finding: the docstring listed the
        shapes and nothing checked the list against the pattern.
        """
        from .._policy import TagStyle

        style = TagStyle()
        documented = [
            line.strip().split("  ")[0].strip()
            for line in (style.lenient_pattern.__doc__ or "").splitlines()
            if line.strip().startswith(("[EMAIL", "[email", "\\[EMAIL"))
        ]
        assert len(documented) >= 8, "the shape table shrank or disappeared"
        for shape in documented:
            # The docstring is raw and the table is literal, so the only
            # substitution is the one the table itself declares.
            candidate = shape.replace("<U+2011>", "‑").replace("<NL>", "\n")
            assert style.lenient_pattern().fullmatch(candidate), (
                "documented shape {0!r} does not match the pattern".format(shape)
            )


class TestCP045BrokenPipeReportedAsAnError:
    """
    ``CP-045`` — a reader closing the pipe is not this program's error.

    Notes
    -----
    **Developer notes — the root cause is an over-broad ``except``.**

    :func:`~scikitplot.cleanprompt._cli.main` translates :class:`OSError` into
    ``error: ...`` and status ``1``, which is right for a missing vault, an
    unreadable input file or a full disk. :class:`BrokenPipeError` is an
    ``OSError`` and is none of those things: it means the *reader* went away,
    which is what ``| head -1`` does by design.

    So ``cleanprompt kinds | head -1`` — a correct command line, doing exactly
    what the user asked — printed ``error: [Errno 32] Broken pipe`` into the
    terminal and exited non-zero for a reason the user could not act on. On the
    reports that made that visible it also produced a second, worse message:
    the interpreter's shutdown flush hit the same closed pipe and printed
    ``Exception ignored in: <_io.TextIOWrapper name='<stdout>' ...>``.

    Both halves have to be fixed together, and the second is why simply
    swallowing the exception is not enough: the flush that raises happens after
    every handler has returned. Descriptor ``1`` is therefore pointed at
    ``os.devnull`` before returning, which is CPython's own recipe.

    The status is ``128 + SIGPIPE``, the number a shell reports for a process
    the kernel killed with that signal, so a pipeline cannot tell the two cases
    apart. There is no ``SIGPIPE`` on Windows, and there the ordinary error
    status is the honest answer.

    **User notes.** Piping any report into ``head``, ``less`` or ``grep -m1``
    is quiet and ordinary again.
    """

    @staticmethod
    def _main_with_closed_stdout(argv):
        """Run the real entry point against a stdout that refuses writes."""
        import io

        from .._cli import main

        class _ClosedPipe(io.StringIO):
            def write(self, _text):
                raise BrokenPipeError(32, "Broken pipe")

        errors = io.StringIO()
        status = main(
            list(argv),
            stdin=io.StringIO(""),
            stdout=_ClosedPipe(),
            stderr=errors,
        )
        return status, errors.getvalue()

    def test_the_status_is_the_sigpipe_convention(self):
        from .._cli import EXIT_BROKEN_PIPE

        status, _ = self._main_with_closed_stdout(["kinds"])
        assert status == EXIT_BROKEN_PIPE

    def test_nothing_is_printed_to_standard_error(self):
        _, errors = self._main_with_closed_stdout(["kinds"])
        assert errors == ""

    def test_it_is_not_reported_as_an_ordinary_os_error(self):
        _, errors = self._main_with_closed_stdout(["kinds"])
        assert "Broken pipe" not in errors
        assert "error:" not in errors

    def test_both_frontends_agree(self):
        from .._cli import EXIT_BROKEN_PIPE

        for frontend in ("argparse", "click"):
            status, errors = self._main_with_closed_stdout(
                ["--frontend", frontend, "kinds"]
            )
            assert status == EXIT_BROKEN_PIPE
            assert errors == ""

    def test_an_ordinary_os_error_is_still_reported(self):
        """The narrowing must not silence the errors the user can act on."""
        import io

        from .._cli import EXIT_ERROR, main

        errors = io.StringIO()
        status = main(
            ["inspect", "--in", "/nonexistent/cleanprompt/input.txt"],
            stdin=io.StringIO(""),
            stdout=io.StringIO(),
            stderr=errors,
        )
        assert status == EXIT_ERROR
        assert "error:" in errors.getvalue()

    def test_the_constant_is_portable(self):
        """Windows has no SIGPIPE; the constant must still be a valid status."""
        from .._cli import EXIT_BROKEN_PIPE

        assert isinstance(EXIT_BROKEN_PIPE, int)
        assert 0 < EXIT_BROKEN_PIPE < 256

    def test_a_real_pipeline_is_quiet(self):
        """The reported symptom, end to end, through a real process."""
        import subprocess
        import sys

        writer = subprocess.Popen(
            [sys.executable, "-B", "-m", "scikitplot.cleanprompt", "kinds"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert writer.stdout is not None
        writer.stdout.readline()
        writer.stdout.close()
        errors = writer.stderr.read() if writer.stderr else ""
        writer.wait(timeout=60)

        assert "Broken pipe" not in errors
        assert "Exception ignored" not in errors
