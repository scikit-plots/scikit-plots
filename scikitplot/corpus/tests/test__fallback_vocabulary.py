# scikitplot/corpus/tests/test__fallback_vocabulary.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
One fallback vocabulary (R2-5): what ran, what was asked, why they differ.

Every component that can run something other than what was configured reports
``requested_backend``, ``backend``, ``fallback_reason`` and ``allow_fallback``.
Covers the vector backends (F-40: ``impl="auto"`` must pick a binding that
honours the requested options), backend-chain outcomes, and the capability
probes (F-50: an installed backend that fails to load is ``broken``, and the
snapshot loads no backend unless asked).
"""

from __future__ import annotations

import json
import logging
import sys

import numpy as np
import pytest

from .. import _capabilities as caps
from .._backends import BackendCandidate, BackendPolicy, run_backend_chain
from .._capabilities import CapabilityStatus, capability_snapshot, probe_backend
from .._retrieval import LegKind
from .._retrievers import DenseRetriever
from .._similarity import RetrievalConfig, RetrievalIndex
from .._similarity import _backends as backends

logger = logging.getLogger(__name__)

_FOUR = ("requested_backend", "backend", "fallback_reason", "allow_fallback")


class _Doc:
    def __init__(self, doc_id, text, embedding):
        self.doc_id = doc_id
        self.text = text
        self.normalized_text = text
        self.embedding = embedding
        self.tokens = None


def _leg(response, kind):
    (leg,) = [leg for leg in response.legs if leg.leg is kind]
    return leg


def _docs():
    return [
        _Doc("a", "alpha", [1.0, 0.0]),
        _Doc("b", "beta", [0.0, 1.0]),
        _Doc("c", "gamma", [0.7, 0.7]),
    ]


@pytest.fixture
def broken_annoy(monkeypatch):
    """Make Annoy look installed but failing to load (a mis-linked binding)."""
    error = ImportError("undefined symbol: annoy_build")
    monkeypatch.setattr(
        backends.AnnoyBackend, "is_installed", classmethod(lambda cls: True)
    )
    monkeypatch.setattr(
        backends.AnnoyBackend, "load_error", classmethod(lambda cls: error)
    )
    return error


@pytest.fixture
def fake_bindings(monkeypatch):
    """Serve fake classes for both Annoy bindings; ``failing`` ones raise."""
    failing = set()

    class _Index:
        def __init__(self, dim, metric, **kwargs):
            self.kwargs = kwargs

        def add_item(self, i, vector):
            pass

        def build(self, n_trees):
            pass

    def fake_import(impl):
        if impl in failing:
            raise ImportError(f"{impl} binding is broken")
        return _Index

    monkeypatch.setattr(backends, "_import_annoy_impl", fake_import)
    return failing


# ---------------------------------------------------------------------------
# F-40: the Annoy binding is chosen for the options requested
# ---------------------------------------------------------------------------


class TestAnnoyBinding:
    def test_the_option_table_matches_the_bindings(self):
        """The declared table is what each real binding accepts (F-40)."""
        try:
            classes = {
                name: backends._import_annoy_impl(name)
                for name in backends._ANNOY_IMPL_ORDER
            }
        except Exception as exc:  # noqa: BLE001 - optional native binding
            pytest.skip(f"Annoy bindings not loadable here: {exc}")
        for name, index_cls in classes.items():
            for option, value in (("dtype", "float32"), ("index_dtype", "int64")):
                accepted = option in backends._ANNOY_IMPL_OPTIONS[name]
                if accepted:
                    index_cls(4, "angular", **{option: value})
                else:
                    with pytest.raises(TypeError):
                        index_cls(4, "angular", **{option: value})

    def test_auto_with_a_dtype_runs_on_the_binding_that_honours_it(self):
        """The review's probe: ``auto`` + ``float64`` used to discard it."""
        if not backends.AnnoyBackend.is_available():
            pytest.skip("Annoy is not available")
        backend = backends.AnnoyBackend(n_trees=2, impl="auto", dtype="float64")
        backend.build(np.random.default_rng(0).random((20, 4)))
        record = backend.provenance()
        assert record["impl"] == "cython"
        assert record["discarded_options"] == {}
        assert record["fallback_reason"] is None

    def test_auto_without_options_prefers_the_high_level_binding(self, fake_bindings):
        backend = backends.AnnoyBackend(n_trees=1)
        backend.build(np.eye(2, dtype=np.float32))
        assert backend.provenance()["impl"] == "highlevel"
        assert backend.fallback_reason is None

    def test_a_preferred_binding_that_cannot_load_is_a_reported_fallback(
        self, fake_bindings
    ):
        fake_bindings.add("highlevel")
        backend = backends.AnnoyBackend(n_trees=1)
        backend.build(np.eye(2, dtype=np.float32))
        assert backend.provenance()["impl"] == "cython"
        assert backend.fallback_reason == "highlevel_unavailable"
        strict = backends.AnnoyBackend(n_trees=1, allow_fallback=False)
        with pytest.raises(backends.FallbackRefusedError, match="allow_fallback=False"):
            strict.build(np.eye(2, dtype=np.float32))

    def test_options_with_no_loadable_accepting_binding(self, fake_bindings):
        fake_bindings.add("cython")
        backend = backends.AnnoyBackend(n_trees=1, dtype="float64")
        backend.build(np.eye(2, dtype=np.float32))
        assert backend.provenance()["impl"] == "highlevel"
        assert backend.fallback_reason == "option_unsupported"
        assert backend.discarded_options == {"dtype": "float64"}
        with pytest.raises(backends.FallbackRefusedError, match="no binding that does"):
            backends.AnnoyBackend(
                n_trees=1, dtype="float64", allow_fallback=False
            ).build(np.eye(2, dtype=np.float32))

    def test_no_loadable_binding_raises(self, fake_bindings):
        fake_bindings.update({"highlevel", "cython"})
        with pytest.raises(ImportError, match="no Annoy binding"):
            backends.AnnoyBackend(n_trees=1).build(np.eye(2, dtype=np.float32))

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"impl": "fast"}, ValueError),
            ({"impl": 3}, TypeError),
            ({"allow_fallback": 1}, TypeError),
            ({"dtype": np.float64}, TypeError),
            ({"index_dtype": ""}, TypeError),
        ],
    )
    def test_a_bad_option_is_refused_at_construction(self, kwargs, error):
        """A typo is never a fallback: it raises when the backend is made."""
        with pytest.raises(error):
            backends.AnnoyBackend(**kwargs)

    def test_impl_none_still_means_auto(self, fake_bindings):
        """Review F13: ``impl=None`` meant ``"auto"`` before R2-5."""
        backend = backends.AnnoyBackend(impl=None, n_trees=1)
        backend.build(np.eye(2, dtype=np.float32))
        assert backend.provenance()["requested_impl"] == "auto"

    def test_the_built_precision_is_reported(self, fake_bindings):
        """Review F9: ``capabilities()["dtype"]`` names what was built."""
        honoured = backends.AnnoyBackend(n_trees=1, dtype="float64")
        honoured.build(np.eye(2, dtype=np.float32))
        assert honoured.capabilities()["dtype"] == "float64"
        dropped = backends.AnnoyBackend(n_trees=1, impl="highlevel", dtype="float64")
        dropped.build(np.eye(2, dtype=np.float32))
        assert dropped.capabilities()["dtype"] == "float32"
        json.dumps(dropped.provenance())

    def test_a_subclass_receives_the_switch(self, monkeypatch, fake_bindings):
        """Review F2: a subclass of AnnoyBackend gets ``allow_fallback`` too."""

        class MyAnnoy(backends.AnnoyBackend):
            name = "my-annoy"

        monkeypatch.setattr(MyAnnoy, "is_available", classmethod(lambda c: True))
        backend = backends.select_backend(
            MyAnnoy, allow_fallback=False, annoy_impl="highlevel", annoy_dtype="float64"
        )
        assert backend.allow_fallback is False
        with pytest.raises(backends.FallbackRefusedError):
            backend.build(np.eye(2, dtype=np.float32))


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestSelection:
    def test_every_backend_reports_the_four_names(self):
        backend = backends.select_backend("bruteforce")
        record = backend.provenance()
        assert tuple(record) == _FOUR
        assert record == {
            "requested_backend": "bruteforce",
            "backend": "bruteforce",
            "fallback_reason": None,
            "allow_fallback": True,
        }
        assert (
            backends.BruteForceBackend().provenance()["requested_backend"]
            == "bruteforce"
        )

    def test_auto_records_that_it_was_auto(self):
        backend = backends.select_backend("auto")
        assert backend.requested_backend == "auto"
        assert backend.fallback_reason is None

    def test_auto_skips_an_installed_backend_that_cannot_load(self, broken_annoy):
        backend = backends.select_backend("auto")
        assert backend.name != "annoy"
        assert backend.requested_backend == "auto"
        assert backend.fallback_reason == "annoy_broken"

    def test_strict_auto_refuses_a_broken_backend(self, broken_annoy):
        with pytest.raises(
            backends.FallbackRefusedError, match="installed but cannot be loaded"
        ):
            backends.select_backend("auto", allow_fallback=False)

    def test_a_named_broken_backend_says_it_is_broken(self, broken_annoy):
        with pytest.raises(RuntimeError, match="installed but cannot be loaded"):
            backends.select_backend("annoy")

    def test_an_absent_backend_is_not_a_fallback_under_auto(self, monkeypatch):
        monkeypatch.setattr(
            backends.AnnoyBackend, "is_installed", classmethod(lambda c: False)
        )
        monkeypatch.setattr(
            backends.AnnoyBackend, "is_available", classmethod(lambda c: False)
        )
        backend = backends.select_backend("auto", allow_fallback=False)
        assert backend.name != "annoy"
        assert backend.fallback_reason is None

    def test_the_switch_reaches_annoy_and_conflicts_are_refused(self, monkeypatch):
        # Construction imports nothing, so availability can be stubbed.
        monkeypatch.setattr(
            backends.AnnoyBackend, "is_available", classmethod(lambda c: True)
        )
        assert (
            backends.select_backend("annoy", allow_fallback=False).allow_fallback
            is False
        )
        relaxed = backends.select_backend(
            "annoy", index_kwargs={"allow_fallback": False}
        )
        assert relaxed.allow_fallback is False
        with pytest.raises(ValueError, match="conflicting Annoy configuration"):
            backends.select_backend(
                "annoy", allow_fallback=False, index_kwargs={"allow_fallback": True}
            )

    def test_allow_fallback_must_be_a_bool(self):
        with pytest.raises(TypeError, match="allow_fallback"):
            backends.select_backend("bruteforce", allow_fallback="no")


# ---------------------------------------------------------------------------
# RetrievalIndex: config, provenance, dense legs
# ---------------------------------------------------------------------------


class TestRetrievalIndex:
    def test_the_config_carries_the_switch(self):
        assert RetrievalConfig().allow_fallback is True
        with pytest.raises(TypeError, match="allow_fallback"):
            RetrievalConfig(allow_fallback=0)

    def test_provenance_and_the_dense_leg(self):
        index = RetrievalIndex(RetrievalConfig(backend="bruteforce"))
        assert index.backend_provenance is None
        index.build(_docs())
        assert index.backend_provenance == {
            "requested_backend": "bruteforce",
            "backend": "bruteforce",
            "fallback_reason": None,
            "allow_fallback": True,
        }
        response = index.search("alpha", query_embedding=[1.0, 0.0])
        leg = _leg(response, LegKind.DENSE)
        assert (leg.backend, leg.requested_backend, leg.fallback_reason) == (
            "bruteforce",
            "bruteforce",
            None,
        )
        payload = leg.to_dict()
        assert payload["requested_backend"] == "bruteforce"
        assert payload["fallback_reason"] is None

    def test_a_broken_backend_is_reported_on_the_leg(self, broken_annoy):
        index = RetrievalIndex(RetrievalConfig(backend="auto"))
        index.build(_docs())
        response = index.search("alpha", query_embedding=[1.0, 0.0])
        leg = _leg(response, LegKind.DENSE)
        assert leg.requested_backend == "auto"
        assert leg.fallback_reason == "annoy_broken"
        assert index.backend_provenance["fallback_reason"] == "annoy_broken"

    def test_strict_config_refuses_at_build(self, broken_annoy):
        index = RetrievalIndex(RetrievalConfig(backend="auto", allow_fallback=False))
        with pytest.raises(backends.FallbackRefusedError, match="allow_fallback=False"):
            index.build(_docs())

    def test_a_refused_annoy_option_fails_the_build(self, monkeypatch, fake_bindings):
        """Review F1: the refusal is not mistaken for invalid embeddings."""
        monkeypatch.setattr(
            backends.AnnoyBackend, "is_available", classmethod(lambda c: True)
        )
        strict = RetrievalConfig(
            backend="annoy",
            annoy_impl="highlevel",
            annoy_dtype="float64",
            allow_fallback=False,
        )
        with pytest.raises(backends.FallbackRefusedError, match="does not accept"):
            RetrievalIndex(strict).build(_docs())

    def test_a_backend_error_propagates_but_bad_data_disables(self, monkeypatch):
        """Only invalid embeddings disable the dense index (review F1)."""

        def refuse(self, embeddings):
            raise ValueError("Unsupported dtype 'flaot64'")

        monkeypatch.setattr(backends.BruteForceBackend, "build", refuse)
        with pytest.raises(ValueError, match="flaot64"):
            RetrievalIndex(RetrievalConfig(backend="bruteforce")).build(_docs())
        monkeypatch.undo()
        bad = _docs()
        bad[0].embedding = [float("nan"), 0.0]
        index = RetrievalIndex(RetrievalConfig(backend="bruteforce"))
        index.build(bad)
        assert index.backend_provenance is None

    def test_the_dense_retriever_leg_carries_the_reason(self, broken_annoy):
        """Review F4: one source for every dense leg."""
        index = RetrievalIndex(RetrievalConfig(backend="auto"))
        index.build(_docs())
        result = DenseRetriever(index).retrieve(
            "alpha", RetrievalConfig(), query_embedding=[1.0, 0.0]
        )
        outcome = result.outcome
        assert (outcome.requested_backend, outcome.fallback_reason) == (
            "auto",
            "annoy_broken",
        )

    def test_a_backend_without_provenance_requested_itself(self):
        index = RetrievalIndex(RetrievalConfig(backend="bruteforce"))
        index.build(_docs())

        class Duck:
            name = "duck"

        index._backend = Duck()
        assert index.backend_provenance == {
            "requested_backend": "duck",
            "backend": "duck",
            "fallback_reason": None,
            "allow_fallback": True,
        }

    def test_a_lexical_leg_names_no_backend(self):
        index = RetrievalIndex(RetrievalConfig(match_mode="keyword"))
        index.build(_docs())
        leg = _leg(index.search("alpha"), LegKind.LEXICAL)
        assert (leg.backend, leg.requested_backend, leg.fallback_reason) == (
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Backend chains (readers)
# ---------------------------------------------------------------------------


def _run(*candidates, policy=None):
    return run_backend_chain(
        candidates,
        default=None,
        logger=logger,
        component="Reader",
        operation="probe",
        policy=policy,
    )


def _boom():
    raise RuntimeError("down")


class TestBackendChain:
    def test_the_first_candidate_answering_is_no_fallback(self):
        outcome = _run(
            BackendCandidate("a", lambda: 1), BackendCandidate("b", lambda: 2)
        )
        assert (outcome.requested_backend, outcome.backend) == ("a", "a")
        assert outcome.fallback_reason is None
        assert outcome.allow_fallback is True

    @pytest.mark.parametrize(
        ("first", "reason"),
        [
            (BackendCandidate("a", _boom), "failed"),
            (
                BackendCandidate("a", lambda: None, accept=lambda v: v is not None),
                "no_result",
            ),
        ],
        ids=["failed", "no_result"],
    )
    def test_the_reason_names_the_requested_candidates_fate(self, first, reason):
        outcome = _run(first, BackendCandidate("b", lambda: 2))
        assert (outcome.requested_backend, outcome.backend) == ("a", "b")
        assert outcome.fallback_reason == reason
        payload = outcome.to_dict()
        assert {key: payload[key] for key in _FOUR} == {
            "requested_backend": "a",
            "backend": "b",
            "fallback_reason": reason,
            "allow_fallback": True,
        }
        json.dumps(payload)

    def test_a_skipped_candidate_was_never_requested(self):
        """A policy skip is like an absent backend under "auto" (review F5)."""
        outcome = _run(
            BackendCandidate("a", lambda: 1, requires_network=True),
            BackendCandidate("b", lambda: 2),
            policy=BackendPolicy.offline(),
        )
        assert outcome.skipped == ("a",)
        assert (outcome.requested_backend, outcome.backend) == ("b", "b")
        assert outcome.fallback_reason is None

    def test_strict_never_reports_a_fallback(self):
        """``allow_fallback=False`` attempts only the requested candidate."""
        strict = BackendPolicy.first_available()
        ran = _run(
            BackendCandidate("a", lambda: 1),
            BackendCandidate("b", lambda: 2),
            policy=strict,
        )
        assert (ran.backend, ran.fallback_reason, ran.allow_fallback) == (
            "a",
            None,
            False,
        )
        failed = _run(
            BackendCandidate("a", _boom),
            BackendCandidate("b", lambda: 2),
            policy=strict,
        )
        assert (failed.backend, failed.attempted) == (None, ("a",))
        assert failed.fallback_reason == "failed"

    def test_a_chain_that_found_nothing_still_says_why(self):
        outcome = _run(BackendCandidate("a", _boom))
        assert outcome.backend is None
        assert outcome.fallback_reason == "failed"

    def test_the_policy_switch_is_reported(self):
        outcome = _run(
            BackendCandidate("a", lambda: 1), policy=BackendPolicy.first_available()
        )
        assert outcome.allow_fallback is False


# ---------------------------------------------------------------------------
# F-50: probes
# ---------------------------------------------------------------------------


class TestProbes:
    def test_an_installed_backend_that_cannot_load_is_broken(self, broken_annoy):
        status, reason = probe_backend(backends.AnnoyBackend)
        assert status is CapabilityStatus.BROKEN
        assert reason == "load_failed: ImportError"
        snapshot = capability_snapshot(load=True)["ann_backends"]["annoy"]
        assert snapshot["status"] == "broken"
        assert snapshot["available"] is False

    def test_without_load_an_unloaded_backend_is_unknown(self, monkeypatch):
        monkeypatch.setattr(
            backends.AnnoyBackend, "is_installed", classmethod(lambda c: True)
        )
        monkeypatch.setattr(
            backends.AnnoyBackend,
            "probe_modules",
            ("scikitplot._no_such_binding_module",),
        )

        def must_not_load(cls):
            raise AssertionError("probing without load imported the backend")

        monkeypatch.setattr(
            backends.AnnoyBackend, "is_available", classmethod(must_not_load)
        )
        assert probe_backend(backends.AnnoyBackend, load=False) == (
            CapabilityStatus.UNKNOWN,
            "not_loaded",
        )
        entry = capability_snapshot()["ann_backends"]["annoy"]
        assert (entry["status"], entry["reason_code"], entry["available"]) == (
            "unknown",
            "not_loaded",
            False,
        )

    def test_without_load_an_absent_backend_is_absent(self, monkeypatch):
        monkeypatch.setattr(
            backends.AnnoyBackend, "is_installed", classmethod(lambda c: False)
        )
        assert probe_backend(backends.AnnoyBackend, load=False) == (
            CapabilityStatus.ABSENT,
            "not_installed",
        )

    def test_a_backend_without_the_new_probes_keeps_its_answer(self):
        class Plain:
            @classmethod
            def is_available(cls):
                return False

        assert probe_backend(Plain) == (CapabilityStatus.ABSENT, "not_installed")
        assert probe_backend(Plain, load=False) == (
            CapabilityStatus.ABSENT,
            "not_installed",
        )

    def test_an_absent_custom_backend_is_not_broken(self):
        """Review F6: no evidence of a failed load means ABSENT, not BROKEN."""

        class Custom(backends.VectorIndexBackend):
            name = "custom"

            @classmethod
            def is_available(cls):
                return False

        assert probe_backend(Custom) == (CapabilityStatus.ABSENT, "not_installed")
        with pytest.raises(RuntimeError, match="is not available"):
            backends.select_backend(Custom)

    def test_a_native_library_that_fails_to_load_is_broken(self, monkeypatch, tmp_path):
        """Review F7: FAISS (and Voyager) are probed by loading them."""
        package = tmp_path / "faiss"
        package.mkdir()
        (package / "__init__.py").write_text(
            "raise ImportError('libfaiss.so: cannot open shared object file')\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "faiss", raising=False)
        assert backends.FaissBackend.is_installed()
        assert probe_backend(backends.FaissBackend, load=False) == (
            CapabilityStatus.UNKNOWN,
            "not_loaded",
        )
        assert probe_backend(backends.FaissBackend) == (
            CapabilityStatus.BROKEN,
            "load_failed: ImportError",
        )
        with pytest.raises(RuntimeError, match="installed but cannot be loaded"):
            backends.select_backend("faiss")

    def test_a_raising_load_error_probe_never_propagates(self, monkeypatch):
        monkeypatch.setattr(
            backends.AnnoyBackend, "is_installed", classmethod(lambda c: True)
        )
        monkeypatch.setattr(
            backends.AnnoyBackend, "is_available", classmethod(lambda c: False)
        )

        def explode(cls):
            raise OSError("probe itself failed")

        monkeypatch.setattr(backends.AnnoyBackend, "load_error", classmethod(explode))
        assert probe_backend(backends.AnnoyBackend) == (
            CapabilityStatus.BROKEN,
            "load_failed: OSError",
        )
        assert caps._installed(object) is None
