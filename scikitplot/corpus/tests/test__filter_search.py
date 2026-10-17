# corpus/tests/test__filter_search.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
One filter (R2-2): ``search(..., filter=)``, the security filter, the JSON form.

A filtered search must return what the unfiltered ranking returns once the
excluded documents are removed: never fewer because excluded documents took the
places, never an excluded document, and never a filter silently dropped.
"""

from __future__ import annotations

import datetime as dt
import math
import random

import pytest

from scikitplot.corpus import (
    And,
    CorpusDocument,
    Eq,
    Exists,
    Filter,
    FusionPolicy,
    In,
    Not,
    NotEq,
    NotIn,
    Or,
    Range,
    RetrievalConfig,
    RetrievalIndex,
    SecurityFilter,
    SourceType,
    filter_from_dict,
    filter_to_dict,
)
from scikitplot.corpus._custom_hooks import CustomRetrievalIndex
from scikitplot.corpus._filters import MAX_FILTER_DEPTH, MAX_FILTER_NODES
from scikitplot.corpus._retrieval import LegKind

np = pytest.importorskip("numpy")

_WORDS = ["alpha", "beta", "gamma", "delta", "omega", "sigma"]


def _corpus(rng, n=40):
    docs = []
    for i in range(n):
        words = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(2, 7)))
        doc = CorpusDocument.create(
            "c.txt",
            i,
            f"{words} {i}",
            language=rng.choice(["en", "de", "fr"]),
            source_type=rng.choice([SourceType.BOOK, SourceType.WEB]),
        ).replace(
            embedding=np.asarray([rng.random() for _ in range(4)], dtype=np.float32)
        )
        docs.append(doc)
    return docs


def _random_filter(rng, depth=0):
    leaf = rng.random() < 0.6 or depth > 2
    if leaf:
        kind = rng.choice(["eq", "not_eq", "in", "not_in", "range", "exists"])
        if kind == "eq":
            return Eq("language", rng.choice(["en", "de", "fr"]))
        if kind == "not_eq":
            return NotEq("source_type", rng.choice(["book", "web"]))
        if kind == "in":
            return In("language", rng.sample(["en", "de", "fr"], 2))
        if kind == "not_in":
            return NotIn("language", [rng.choice(["en", "de", "fr"])])
        if kind == "range":
            lo = rng.randint(0, 30)
            return Range("chunk_index", lo=lo, hi=lo + rng.randint(0, 20))
        return Exists(rng.choice(["language", "parent_doc_id"]))
    op = rng.choice(["and", "or", "not"])
    if op == "not":
        return Not(_random_filter(rng, depth + 1))
    clauses = [_random_filter(rng, depth + 1) for _ in range(rng.randint(0, 3))]
    return And(*clauses) if op == "and" else Or(*clauses)


def _index(docs, security=None):
    index = RetrievalIndex(
        RetrievalConfig(backend="bruteforce"), security_filter=security
    )
    index.build(docs)
    return index


def _config(mode, top_k):
    return RetrievalConfig(match_mode=mode, top_k=top_k, backend="bruteforce")


# ---------------------------------------------------------------------------
# A filtered search is the filtered prefix of the full ranking
# ---------------------------------------------------------------------------


class TestFilteredRanking:
    @pytest.mark.parametrize("mode", ["strict", "keyword", "semantic"])
    def test_single_leg_modes_equal_the_filtered_full_ranking(self, mode):
        rng = random.Random(3865)  # noqa: S311 - a seeded test sample
        for _ in range(40):
            docs = _corpus(rng)
            index = _index(docs)
            query = rng.choice(_WORDS)
            qe = np.asarray([rng.random() for _ in range(4)], dtype=np.float32)
            expression = _random_filter(rng)
            top_k = rng.randint(1, 8)
            full = index.search(
                query, config=_config(mode, len(docs)), query_embedding=qe
            )
            expected = [h.doc.doc_id for h in full if expression.matches(h.doc)][:top_k]
            got = index.search(
                query,
                config=_config(mode, top_k),
                query_embedding=qe,
                filter=expression,
            )
            assert [h.doc.doc_id for h in got] == expected
            assert all(expression.matches(h.doc) for h in got)

    def test_hybrid_equals_fusing_the_filtered_legs(self):
        rng = random.Random(4866)  # noqa: S311 - a seeded test sample
        for _ in range(40):
            docs = _corpus(rng)
            index = _index(docs)
            query = rng.choice(_WORDS)
            qe = np.asarray([rng.random() for _ in range(4)], dtype=np.float32)
            expression = _random_filter(rng)
            top_k = rng.randint(1, 6)
            alpha = rng.choice([0.25, 0.5, 0.75])
            fetch_k = min(top_k * 3, len(docs))
            legs = []
            for name, mode in (("lexical", "keyword"), ("dense", "semantic")):
                full = index.search(
                    query, config=_config(mode, len(docs)), query_embedding=qe
                )
                ids = [h.doc.doc_id for h in full if expression.matches(h.doc)]
                legs.append((name, ids[:fetch_k]))
            expected = [
                item.key
                for item in FusionPolicy.from_alpha(alpha, top_k=top_k).fuse(legs)
            ]
            config = RetrievalConfig(
                match_mode="hybrid",
                top_k=top_k,
                hybrid_alpha=alpha,
                backend="bruteforce",
            )
            got = index.search(
                query, config=config, query_embedding=qe, filter=expression
            )
            assert [h.doc.doc_id for h in got] == expected

    def test_a_filter_that_admits_nothing_gives_an_empty_complete_answer(self):
        docs = _corpus(random.Random(1))  # noqa: S311 - a seeded test sample
        response = _index(docs).search(
            "alpha", config=_config("keyword", 5), filter=Eq("language", "xx")
        )
        assert list(response) == []
        assert response.status.value == "empty"

    def test_every_leg_that_ran_reports_the_filter(self):
        docs = _corpus(random.Random(2))  # noqa: S311 - a seeded test sample
        qe = np.ones(4, dtype=np.float32)
        expression = And(Eq("language", "en"), Not(Exists("parent_doc_id")))
        response = _index(docs).search(
            "alpha", config=_config("hybrid", 5), query_embedding=qe, filter=expression
        )
        for leg in response.legs:
            assert leg.filter_capability is not None
            assert leg.filter_capability.ok
            assert leg.filter_capability.emulated == {"And", "Eq", "Not", "Exists"}
        payload = response.to_dict()
        assert payload["filter"] == filter_to_dict(expression)
        assert payload["legs"][0]["filter"]["emulated"] == [
            "And",
            "Eq",
            "Exists",
            "Not",
        ]

    def test_an_unfiltered_search_reports_no_filter(self):
        docs = _corpus(random.Random(3))  # noqa: S311 - a seeded test sample
        response = _index(docs).search("alpha", config=_config("keyword", 5))
        assert response.filter is None
        assert all(leg.filter_capability is None for leg in response.legs)

    def test_the_json_form_is_accepted_and_anything_else_refused(self):
        docs = _corpus(random.Random(4))  # noqa: S311 - a seeded test sample
        index = _index(docs)
        as_json = {"op": "eq", "field": "language", "value": "en"}
        a = index.search("alpha", config=_config("keyword", 5), filter=as_json)
        b = index.search(
            "alpha", config=_config("keyword", 5), filter=Eq("language", "en")
        )
        assert [h.doc.doc_id for h in a] == [h.doc.doc_id for h in b]
        with pytest.raises(TypeError, match="filter must be"):
            index.search("alpha", config=_config("keyword", 5), filter="language=en")


# ---------------------------------------------------------------------------
# The security filter narrows every query and cannot be widened or removed
# ---------------------------------------------------------------------------


class TestSecurityFilter:
    def test_a_query_cannot_widen_the_scope(self):
        docs = _corpus(random.Random(5))  # noqa: S311 - a seeded test sample
        scope = SecurityFilter(Eq("language", "de"))
        index = _index(docs, scope)
        everything = Or(Exists("language"), Not(Exists("language")))
        qe = np.ones(4, dtype=np.float32)
        for mode in ("strict", "keyword", "semantic", "hybrid"):
            for expression in (None, everything, Eq("language", "en")):
                hits = index.search(
                    "alpha",
                    config=_config(mode, 40),
                    query_embedding=qe,
                    filter=expression,
                )
                assert all(h.doc.language == "de" for h in hits), (mode, expression)
        assert (
            list(
                index.search(
                    "alpha", config=_config("keyword", 40), filter=Eq("language", "en")
                )
            )
            == []
        )

    def test_the_scope_is_read_only_and_typed(self):
        index = _index(
            _corpus(random.Random(6)), SecurityFilter(Eq("language", "de"))
        )  # noqa: S311
        with pytest.raises(AttributeError):
            index.security_filter = None  # type: ignore[misc]
        with pytest.raises(TypeError, match="SecurityFilter"):
            RetrievalIndex(security_filter=Eq("language", "de"))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="must be a Filter"):
            SecurityFilter({"op": "eq"})  # type: ignore[arg-type]

    def test_a_custom_scorer_only_sees_admitted_documents(self):
        seen = []

        def scorer(query, documents, config):
            seen.append({d.language for d in documents})
            return []

        index = CustomRetrievalIndex(
            custom_scorer_fn=scorer,
            security_filter=SecurityFilter(Eq("language", "de")),
        )
        index.build(_corpus(random.Random(7)))  # noqa: S311 - a seeded test sample
        index.search("alpha")
        index.search("alpha", filter=Eq("source_type", "book"))
        assert seen == [{"de"}, {"de"}]


# ---------------------------------------------------------------------------
# JSON form: strict, round-trip exact
# ---------------------------------------------------------------------------


class TestJsonForm:
    def test_every_operator_round_trips(self):
        expression = And(
            Eq("source_type", SourceType.BOOK),
            NotEq("language", "de"),
            In("chunk_index", [1, 2, 3]),
            NotIn("language", ["fr"]),
            Range("chunk_index", lo=0.5, hi=None),
            Or(Exists("parent_doc_id"), Not(Exists("language"))),
            Or(),
            And(),
        )
        data = filter_to_dict(expression)
        assert data["args"][0] == {"op": "eq", "field": "source_type", "value": "book"}
        rebuilt = filter_from_dict(data)
        assert filter_to_dict(rebuilt) == data
        assert expression.to_dict() == data

    def test_random_trees_round_trip_and_agree(self):
        rng = random.Random(5866)  # noqa: S311 - a seeded test sample
        docs = _corpus(rng)
        for _ in range(500):
            expression = _random_filter(rng)
            data = filter_to_dict(expression)
            rebuilt = filter_from_dict(data)
            assert filter_to_dict(rebuilt) == data
            assert [expression.matches(d) for d in docs] == [
                rebuilt.matches(d) for d in docs
            ]

    @pytest.mark.parametrize(
        ("data", "match"),
        [
            ([], "must be an object"),
            ({"op": "like", "field": "language", "value": 1}, "unknown op"),
            ({"op": "eq", "field": "language"}, "takes exactly the keys"),
            (
                {"op": "eq", "field": "language", "value": 1, "x": 2},
                "takes exactly the keys",
            ),
            ({"op": "eq", "field": "", "value": 1}, "non-empty str"),
            ({"op": "eq", "field": 3, "value": 1}, "non-empty str"),
            ({"op": "eq", "field": "language", "value": [1]}, "JSON scalar"),
            ({"op": "eq", "field": "language", "value": {"b": 1}}, "JSON scalar"),
            ({"op": "eq", "field": "language", "value": float("nan")}, "finite"),
            ({"op": "in", "field": "language", "values": "abc"}, "must be a list"),
            ({"op": "in", "field": "language", "values": [[1]]}, "JSON scalar"),
            (
                {"op": "and", "args": {"op": "exists", "field": "language"}},
                "must be a list",
            ),
            ({"op": "not", "args": []}, "takes exactly the keys"),
        ],
    )
    def test_off_contract_json_is_refused(self, data, match):
        with pytest.raises(ValueError, match=match):
            filter_from_dict(data)

    def test_depth_nodes_and_values_are_bounded(self):
        node = {"op": "exists", "field": "language"}
        for _ in range(MAX_FILTER_DEPTH + 1):
            node = {"op": "not", "arg": node}
        with pytest.raises(ValueError, match="deeper than"):
            filter_from_dict(node)
        wide = {
            "op": "or",
            "args": [{"op": "exists", "field": "language"}] * MAX_FILTER_NODES,
        }
        with pytest.raises(ValueError, match="nodes"):
            filter_from_dict(wide)
        many = {"op": "in", "field": "language", "values": list(range(5000))}
        with pytest.raises(ValueError, match="values"):
            filter_from_dict(many)

    @pytest.mark.parametrize("value", [object(), float("inf"), [1], (1,)])
    def test_a_value_json_cannot_hold_is_refused_on_export(self, value):
        with pytest.raises(ValueError, match=r"filter\.args\[1\]"):
            filter_to_dict(And(Exists("language"), Eq("text", value)))

    def test_a_foreign_filter_subclass_is_refused_on_export(self):
        class Mine(Filter):
            pass

        with pytest.raises(TypeError, match="not a filter expression"):
            filter_to_dict(Mine())

    def test_large_ints_and_bools_keep_their_type(self):
        data = filter_to_dict(In("chunk_index", [True, 2**62, 1.5, None]))
        assert data["values"] == [True, 2**62, 1.5, None]
        assert [type(v) for v in filter_from_dict(data).values] == [
            bool,
            int,
            float,
            type(None),
        ]
        assert math.isfinite(filter_from_dict(data).values[2])


def test_builder_search_forwards_the_filter(tmp_path):
    from scikitplot.corpus import BuilderConfig, CorpusBuilder  # noqa: PLC0415

    path = tmp_path / "a.txt"
    path.write_bytes(b"Alpha one here. Alpha two here. Alpha three here.")
    config = BuilderConfig(
        filter_kwargs={"min_words": 1, "min_chars": 1}, build_index=True
    )
    with CorpusBuilder(config) as builder:
        builder.build(path)
        hits = builder.search(
            "alpha", match_mode="keyword", filter=Range("chunk_index", lo=1)
        )
    assert hits and all(h.doc.chunk_index >= 1 for h in hits)
    assert hits.legs[0].filter_capability is not None
    assert LegKind.LEXICAL == hits.legs[0].leg


# ---------------------------------------------------------------------------
# Review findings (2f)
# ---------------------------------------------------------------------------


def _scoped_corpus():
    docs = [
        CorpusDocument.create("p.txt", 0, "merger plan merger", collection_id="public"),
        CorpusDocument.create(
            "p.txt", 1, "acquisition target plan", collection_id="public"
        ),
    ]
    secret = [
        CorpusDocument.create("s.txt", i, text, collection_id="secret")
        for i, text in enumerate(["merger merger merger", "merger talks", "merger day"])
    ]
    return docs, secret


class TestReviewFindings:
    def test_the_vector_seam_honours_the_scope(self):
        docs = [
            CorpusDocument.create("a.txt", i, f"text {i}", collection_id=c).replace(
                embedding=np.asarray([1.0, i / 10], dtype=np.float32)
            )
            for i, c in enumerate(["public", "secret"] * 5)
        ]
        index = _index(docs, SecurityFilter(Eq("collection_id", "public")))
        ids = {doc_id for doc_id, _ in index.query(np.asarray([1.0, 0.5]), 10)}
        allowed = {d.doc_id for d in docs if d.collection_id == "public"}
        assert ids == allowed

    def test_out_of_scope_documents_do_not_change_in_scope_scores(self):
        public, secret = _scoped_corpus()
        scope = SecurityFilter(Eq("collection_id", "public"))
        alone = _index(public, scope).search(
            "merger plan", config=_config("keyword", 5)
        )
        mixed = _index(public + secret, scope).search(
            "merger plan", config=_config("keyword", 5)
        )
        assert [(h.doc.doc_id, h.score) for h in alone] == [
            (h.doc.doc_id, h.score) for h in mixed
        ]

    def test_a_short_approximate_backend_falls_back_to_exact_ranking(self):
        docs = _corpus(random.Random(8), n=60)  # noqa: S311 - a seeded test sample
        index = _index(docs)
        real = index._backend

        class _Short:
            name = "short"
            score_semantics = "cosine_similarity"

            def query(self, vector, k):
                return real.query(vector, min(k, 3))

        index._backend = _Short()
        qe = np.ones(4, dtype=np.float32)
        expression = Eq("language", "de")
        got = index.search(
            "x", config=_config("semantic", 5), query_embedding=qe, filter=expression
        )
        index._backend = real
        full = index.search("x", config=_config("semantic", 60), query_embedding=qe)
        expected = [h.doc.doc_id for h in full if expression.matches(h.doc)][:5]
        assert [h.doc.doc_id for h in got] == expected

    def test_in_keeps_its_lookup(self):
        node = In("language", ["en", "de"])
        assert node._lookup == frozenset({("value", "en"), ("value", "de")})

    @pytest.mark.parametrize(
        "field",
        ["colection_id", "replace", "__class__", "_documents", "embedding", "metadata"],
    )
    def test_a_field_a_filter_may_not_name_is_refused(self, field):
        with pytest.raises(ValueError, match="unknown field"):
            SecurityFilter(Not(Eq(field, "b")))
        with pytest.raises(ValueError, match="unknown field"):
            filter_from_dict({"op": "exists", "field": field})

    def test_a_metadata_key_is_reachable(self):
        docs = [
            CorpusDocument.create("a.txt", i, "alpha", metadata={"tenant": t})
            for i, t in enumerate(["a", "b", "a"])
        ]
        index = _index(docs, SecurityFilter(Not(Eq("metadata.tenant", "b"))))
        hits = index.search("alpha", config=_config("keyword", 5))
        assert [h.doc.metadata["tenant"] for h in hits] == ["a", "a"]
        with pytest.raises(ValueError, match="needs a key"):
            filter_from_dict({"op": "exists", "field": "metadata."})

    def test_the_scope_is_frozen_when_bound(self):
        class Box:
            value = "public"

        box = Box()
        scope = SecurityFilter(Eq("collection_id", box))
        box.value = "secret"
        assert scope.expression == Eq("collection_id", "public")
        with pytest.raises(ValueError, match="JSON scalar"):
            SecurityFilter(Eq("collection_id", object()))

    def test_the_response_reports_the_scope_label_not_its_expression(self):
        docs = _corpus(random.Random(9))  # noqa: S311 - a seeded test sample
        index = _index(docs, SecurityFilter(Eq("language", "de"), label="tenant-a"))
        response = index.search(
            "alpha", config=_config("keyword", 5), filter=Exists("text")
        )
        payload = response.to_dict()
        assert payload["scope"] == "tenant-a"
        assert payload["filter"] == {"op": "exists", "field": "text"}
        assert "language" not in str(payload)

    def test_a_query_filter_that_cannot_be_reported_is_refused_up_front(self):
        docs = _corpus(random.Random(10))  # noqa: S311 - a seeded test sample
        index = _index(docs)
        bad = Range("chunk_index", lo=dt.date(2026, 1, 1))
        with pytest.raises(ValueError, match="JSON scalar"):
            index.search("alpha", config=_config("keyword", 5), filter=bad)

    def test_uncomparable_fields_do_not_match_and_bool_is_not_int(self):
        doc = CorpusDocument.create(
            "a.txt", 1, "alpha", metadata={"tags": ["x"], "n": 1, "vec": np.zeros(2)}
        )
        # An array's == is elementwise: neither equal nor different.
        assert not Eq("metadata.vec", 0).matches(doc)
        assert not NotEq("metadata.vec", 0).matches(doc)
        # A list is unhashable: no membership either way.
        assert not In("metadata.tags", ["x"]).matches(doc)
        assert not NotIn("metadata.tags", ["y"]).matches(doc)
        assert not Eq("chunk_index", True).matches(doc)
        assert Eq("chunk_index", 1).matches(doc)
        assert not In("metadata.n", [True]).matches(doc)
        assert Eq("metadata.n", 1.0).matches(doc)

    def test_depth_32_is_accepted_and_33_refused(self):
        def nested(levels):
            node = {"op": "exists", "field": "language"}
            for _ in range(levels - 1):
                node = {"op": "not", "arg": node}
            return node

        filter_from_dict(nested(MAX_FILTER_DEPTH))
        with pytest.raises(ValueError, match="deeper than"):
            filter_from_dict(nested(MAX_FILTER_DEPTH + 1))

    def test_strings_and_ints_are_bounded(self):
        with pytest.raises(ValueError, match="longer than"):
            filter_from_dict({"op": "eq", "field": "text", "value": "x" * 5000})
        with pytest.raises(ValueError, match="longer than"):
            filter_from_dict({"op": "exists", "field": "metadata." + "k" * 5000})
        with pytest.raises(ValueError, match="2\\*\\*63"):
            filter_from_dict({"op": "eq", "field": "chunk_index", "value": 10**30})

    def test_the_builder_binds_its_scope_to_every_index(self, tmp_path):
        from scikitplot.corpus import BuilderConfig, CorpusBuilder  # noqa: PLC0415

        path = tmp_path / "a.txt"
        path.write_bytes(b"Alpha one here. Alpha two here. Alpha three here.")
        config = BuilderConfig(
            filter_kwargs={"min_words": 1, "min_chars": 1},
            build_index=True,
            security_filter=SecurityFilter(Range("chunk_index", hi=0)),
        )
        with CorpusBuilder(config) as builder:
            builder.build(path)
            first = builder.search("alpha", match_mode="keyword")
            builder.add(path)
            second = builder.search("alpha", match_mode="keyword", filter=None)
        assert [h.doc.chunk_index for h in first] == [0]
        assert {h.doc.chunk_index for h in second} == {0}
