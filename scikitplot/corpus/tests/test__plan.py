# scikitplot/corpus/tests/test__plan.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`scikitplot.corpus._plan` — the CF verification matrix.

These correspond to the CF-01…CF-25 checks specified by ADR-C21 and ADR-C22.
The invariant worth stating once: the fluent form is a *view* over canonical
configuration, so it must not create a second pipeline, a hidden ordering, a
silent override, or any runtime effect at configuration time.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from .._plan import (
    CONFIG_DOMAINS,
    DEFAULT_STAGES,
    ConfigConflictError,
    CorpusPlan,
    FluentCorpus,
)

__all__: "list[str]" = [
    "TestOrderIndependence",
    "TestConflictPolicy",
    "TestImmutability",
    "TestStageOrder",
    "TestValidation",
    "TestPlanIdentity",
]


class TestOrderIndependence:
    """CF-01, CF-19 — independent fragments commute."""

    def test_two_domains_commute(self) -> None:
        a = FluentCorpus().embedder("E").storage("S")
        b = FluentCorpus().storage("S").embedder("E")
        assert a.plan() == b.plan()
        assert a.plan().fingerprint == b.plan().fingerprint

    def test_all_domains_commute(self) -> None:
        values = {d: d.upper() for d in CONFIG_DOMAINS}
        forward = FluentCorpus()
        for domain in CONFIG_DOMAINS:
            forward = forward.config(domain, values[domain])
        backward = FluentCorpus()
        for domain in reversed(CONFIG_DOMAINS):
            backward = backward.config(domain, values[domain])
        assert forward.plan() == backward.plan()

    def test_fluent_and_nested_forms_agree(self) -> None:
        """CF-02 — both construction styles compile to the same plan."""
        fluent = FluentCorpus().reader("R").embedder("E").plan()
        nested = CorpusPlan.of(reader="R", embedder="E")
        assert fluent == nested

    def test_partial_configuration_is_valid(self) -> None:
        """CF-03 — configuring one domain is legitimate, not an error."""
        assert FluentCorpus().embedder("E").plan().configured == ["embedder"]


class TestConflictPolicy:
    """CF-04, CF-22 — same-domain conflict is an error by default."""

    def test_duplicate_domain_raises(self) -> None:
        with pytest.raises(ConfigConflictError, match="already configured"):
            FluentCorpus().embedder("A").embedder("B")

    def test_error_names_both_values(self) -> None:
        """The message must say what would have been lost."""
        with pytest.raises(ConfigConflictError) as excinfo:
            FluentCorpus().embedder("A").embedder("B")
        message = str(excinfo.value)
        assert "A" in message and "B" in message

    def test_explicit_replace_method(self) -> None:
        plan = FluentCorpus().embedder("A").replace_embedder("B").plan()
        assert plan.get("embedder") == "B"

    def test_explicit_replace_keyword(self) -> None:
        plan = FluentCorpus().embedder("A").embedder("B", conflict="replace").plan()
        assert plan.get("embedder") == "B"

    def test_replace_works_on_an_unset_domain(self) -> None:
        assert FluentCorpus().replace_embedder("B").plan().get("embedder") == "B"

    def test_unknown_conflict_policy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="conflict must be"):
            FluentCorpus().embedder("A", conflict="merge-somehow")

    def test_unknown_domain_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown configuration domain"):
            FluentCorpus().config("teleporter", "X")


class TestImmutability:
    """CF-23 — every call returns a new builder."""

    def test_original_is_unchanged(self) -> None:
        base = FluentCorpus().reader("R")
        base.storage("A")
        assert base.plan().configured == ["reader"]

    def test_branches_do_not_interfere(self) -> None:
        base = FluentCorpus().reader("R")
        left, right = base.storage("A"), base.storage("B")
        assert left.plan().get("storage") == "A"
        assert right.plan().get("storage") == "B"

    def test_plan_is_hashable_and_comparable(self) -> None:
        a = FluentCorpus().reader("R").plan()
        b = FluentCorpus().reader("R").plan()
        assert a == b
        assert len({a, b}) == 1


class TestStageOrder:
    """CF-20, CF-21 — call order never defines execution order."""

    def test_configuration_order_does_not_change_stages(self) -> None:
        forward = FluentCorpus().chunker("C").normalizer("N").plan()
        backward = FluentCorpus().normalizer("N").chunker("C").plan()
        assert forward.effective_stages == backward.effective_stages == DEFAULT_STAGES
        assert forward == backward

    def test_stages_is_the_only_way_to_set_order(self) -> None:
        plan = FluentCorpus().stages("read", "chunk", "embed").plan()
        assert plan.effective_stages == ("read", "chunk", "embed")

    def test_stage_order_participates_in_the_fingerprint(self) -> None:
        """Two plans differing only in stage order are different plans."""
        a = FluentCorpus().reader("R").stages("read", "chunk").plan()
        b = FluentCorpus().reader("R").stages("chunk", "read").plan()
        assert a != b


class TestValidation:
    """CF-08, CF-09 — phase-2 validation, with no side effects."""

    def test_coherent_plan_has_no_problems(self) -> None:
        assert FluentCorpus().embedder("E").index("I").validate() == []

    def test_index_without_embedder_is_reported(self) -> None:
        codes = [e.code for e in FluentCorpus().index("I").validate()]
        assert codes == ["PLAN_INDEX_WITHOUT_EMBEDDER"]

    def test_unknown_stage_is_reported(self) -> None:
        codes = [e.code for e in FluentCorpus().stages("read", "bogus").validate()]
        assert "PLAN_UNKNOWN_STAGE" in codes

    def test_duplicate_stage_is_reported(self) -> None:
        codes = [e.code for e in FluentCorpus().stages("read", "read").validate()]
        assert "PLAN_DUPLICATE_STAGE" in codes

    def test_all_problems_reported_not_just_the_first(self) -> None:
        problems = FluentCorpus().index("I").stages("read", "bogus", "bogus").validate()
        assert len(problems) >= 2

    def test_build_raises_on_an_invalid_plan(self) -> None:
        with pytest.raises(ValueError, match="invalid corpus plan"):
            FluentCorpus().index("I").build()

    def test_build_returns_the_plan_when_valid(self) -> None:
        builder = FluentCorpus().embedder("E").index("I")
        assert builder.build() == builder.plan()

    def test_configuration_performs_no_io(self) -> None:
        """CF-06, CF-07 — no network, no model loading at configuration time.

        Checked in a fresh interpreter: chaining every domain must not import a
        single heavyweight backend.
        """
        source = (
            "import sys\n"
            "from scikitplot.corpus import FluentCorpus\n"
            "b = FluentCorpus()\n"
            "for d in ('source','reader','embedder','storage','index'):\n"
            "    b = b.config(d, 'X')\n"
            "b.plan().fingerprint\n"
            "watched = ('torch','sentence_transformers','transformers','annoy',"
            "'faiss','requests')\n"
            "print(','.join(m for m in watched if m in sys.modules))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", source], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stderr
        loaded = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        assert loaded == "", f"configuration imported {loaded}"


class TestPlanIdentity:
    """CF-10, CF-11 — a stable, content-derived plan fingerprint."""

    def test_is_deterministic(self) -> None:
        assert (
            FluentCorpus().reader("R").plan().fingerprint
            == FluentCorpus().reader("R").plan().fingerprint
        )

    def test_changes_with_any_fragment(self) -> None:
        a = FluentCorpus().reader("R").plan().fingerprint
        b = FluentCorpus().reader("OTHER").plan().fingerprint
        assert a != b

    def test_fields_are_length_prefixed(self) -> None:
        """Adjacent fragments must not be able to collide."""
        a = FluentCorpus().reader("ab").storage("c").plan()
        b = FluentCorpus().reader("a").storage("bc").plan()
        assert a.fingerprint != b.fingerprint

    def test_plan_is_serialisable(self) -> None:
        payload = FluentCorpus().reader("R").embedder("E").plan().to_dict()
        json.dumps(payload)
        assert payload["configured"] == ["reader", "embedder"]

    def test_repr_is_informative(self) -> None:
        text = repr(FluentCorpus().reader("R"))
        assert "reader" in text


class TestRuntimeMaterialization:
    """Maintenance RUNTIME-01..06 — plan -> runtime without a second pipeline."""

    def test_materialize_does_not_read_source(self, tmp_path) -> None:
        from .._runtime import RuntimeCorpus, materialize_plan

        missing = tmp_path / "not-created.txt"
        plan = FluentCorpus().source(missing).reader("auto").plan()
        runtime = materialize_plan(plan)
        assert isinstance(runtime, RuntimeCorpus)
        assert not missing.exists()
        assert runtime.documents == ()

    def test_fluent_materialize_is_public_and_keeps_build_contract(self, tmp_path) -> None:
        import scikitplot.corpus as corpus

        builder = FluentCorpus().source(tmp_path / "x.txt").reader("auto")
        assert isinstance(builder.build(), CorpusPlan)
        runtime = builder.materialize()
        assert isinstance(runtime, corpus.RuntimeCorpus)
        assert runtime.plan == builder.plan()

    def test_network_is_rejected_by_default_before_reader_io(self) -> None:
        from .._runtime import materialize_plan

        runtime = materialize_plan(
            FluentCorpus().source("https://example.invalid/data.txt").reader("auto").plan()
        )
        with pytest.raises(PermissionError, match="allow_network=False"):
            runtime.run()

    def test_noncanonical_stage_reordering_is_rejected_at_materialization(self) -> None:
        from .._runtime import materialize_plan

        plan = FluentCorpus().stages("chunk", "read").plan()
        with pytest.raises(ValueError, match="stage reordering"):
            materialize_plan(plan)

    def test_index_kwargs_order_does_not_change_plan_fingerprint(self) -> None:
        from scikitplot.corpus._similarity import RetrievalConfig

        left = FluentCorpus().index(
            RetrievalConfig(backend="bruteforce", index_kwargs={"b": 2, "a": 1})
        ).plan()
        right = FluentCorpus().index(
            RetrievalConfig(backend="bruteforce", index_kwargs={"a": 1, "b": 2})
        ).plan()

        assert left == right
        assert left.fingerprint == right.fingerprint

    def test_real_local_run_store_index_search_export_and_add(self, tmp_path) -> None:
        np = pytest.importorskip("numpy")
        from .._chunkers import ParagraphChunkerConfig
        from .._embeddings import EmbeddingEngine
        from .._normalizers import TextNormalizerConfig
        from .._runtime import materialize_plan
        from .._schema import ExportFormat
        from .._similarity import RetrievalConfig
        from .._storage import InMemoryStorage

        first = tmp_path / "first.txt"
        first.write_text(
            "Ghosts remember the old king.\n\nSleep and dreams trouble Hamlet.",
            encoding="utf-8",
        )
        second = tmp_path / "second.txt"
        second.write_text("Words and actors perform upon the stage.", encoding="utf-8")

        def embed(texts):
            return np.asarray(
                [
                    [
                        float(text.lower().count("ghost") + text.lower().count("king")),
                        float(text.lower().count("sleep") + text.lower().count("dream") + 1),
                    ]
                    for text in texts
                ],
                dtype=np.float32,
            )

        engine = EmbeddingEngine(
            backend="custom",
            custom_fn=embed,
            enable_cache=False,
        )
        index_cfg = RetrievalConfig(
            match_mode="semantic",
            backend="bruteforce",
            top_k=3,
        )
        plan = (
            FluentCorpus()
            .source(first)
            .reader("auto")
            .normalizer(TextNormalizerConfig(steps=["unicode", "whitespace"]))
            .chunker(ParagraphChunkerConfig(min_length=1))
            .embedder(engine)
            .storage(InMemoryStorage)
            .index(index_cfg)
            .retrieval(index_cfg)
            .export(ExportFormat.JSONL)
            .plan()
        )

        runtime = materialize_plan(plan)
        result = runtime.run()
        assert result.n_documents >= 2
        assert len(runtime.documents) == result.n_documents
        assert runtime.storage.count() == result.n_documents
        assert runtime.index is not None
        assert runtime.index.backend_name == "bruteforce"
        assert runtime.index_generation is not None

        response = runtime.search("ghost king")
        assert response.query == "ghost king"
        assert len(response) >= 1

        exported = runtime.export(tmp_path / "runtime.jsonl", include_embedding=False)
        assert exported.exists()

        added = runtime.add(second)
        assert added.n_documents >= 1
        assert runtime.storage.count() == len(runtime.documents)
        assert len(runtime.documents) == result.n_documents + added.n_documents

        with pytest.raises(RuntimeError, match="already completed"):
            runtime.run()

        runtime.close()
        runtime.close()
        assert runtime.closed is True
