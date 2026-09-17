"""
Analyzer and recipe identity regressions (slices S-15, S-16).

A lexical index held frequencies and lengths and nothing that said how they were
produced. The tokenizer was an opaque callable, so two indexes built with
different tokenisation were indistinguishable afterwards, and the scoring
formula was identified only by a class name that implies a standard it does not
match. Both now have declared identities that a build can record.

See Also
--------
scikitplot.rank_bm25._identity.Analyzer
scikitplot.rank_bm25._identity.Recipe
"""

import pytest

from .._identity import RECIPES, Analyzer, Recipe, lexical_identity
from .._rank_bm25 import BM25L, BM25Okapi, BM25Plus

CORPUS = [["a", "b"], ["b"], ["c"]]


# -- S-15: the analyzer is declared ---------------------------------------


def test_an_analyzer_declares_what_it_does():
    """Name, version and language are part of the record, not folklore."""
    analyzer = Analyzer(name="whitespace", version="1", language="en")
    assert analyzer.fingerprint
    assert analyzer.tokenize("a b  c") == ["a", "b", "c"]


def test_equal_analyzers_share_an_identity():
    """Identity is the declaration, not the object."""
    assert (Analyzer(name="whitespace", version="1").fingerprint
            == Analyzer(name="whitespace", version="1").fingerprint)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (dict(name="whitespace"), dict(name="unicode")),
        (dict(version="1"), dict(version="2")),
        (dict(language="en"), dict(language="de")),
        (dict(stopwords=frozenset({"the"})), dict(stopwords=frozenset())),
        (dict(stemmer="porter"), dict(stemmer=None)),
        (dict(lowercase=True), dict(lowercase=False)),
    ],
)
def test_every_declared_field_moves_the_identity(left, right):
    """A field that changes the tokens must change the identity."""
    base = dict(name="whitespace", version="1")
    assert (Analyzer(**{**base, **left}).fingerprint
            != Analyzer(**{**base, **right}).fingerprint)


def test_stopwords_order_does_not_matter():
    """A set is a set; identity must not depend on how it was built."""
    assert (Analyzer(name="a", stopwords=frozenset({"x", "y"})).fingerprint
            == Analyzer(name="a", stopwords=frozenset({"y", "x"})).fingerprint)


def test_the_analyzer_applies_what_it_declares():
    """The declaration is not decoration: it drives tokenisation."""
    analyzer = Analyzer(name="a", lowercase=True, stopwords=frozenset({"the"}))
    assert analyzer.tokenize("The Cat") == ["cat"]


def test_an_analyzer_is_stable_across_processes():
    """An identity that moves between interpreters cannot name a build."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[3])
    script = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "from scikitplot.rank_bm25._identity import Analyzer\n"
        "print(Analyzer(name='w', stopwords=frozenset({'a','b','c'})).fingerprint)\n"
    )
    seen = set()
    for seed in ("1", "2", "3"):
        proc = subprocess.run([sys.executable, "-c", script, root],
                              capture_output=True, text=True, check=True,
                              env=dict(os.environ, PYTHONHASHSEED=seed))
        seen.add(proc.stdout.strip().splitlines()[-1])
    assert len(seen) == 1


# -- S-16: the recipe is named --------------------------------------------


@pytest.mark.parametrize("scorer", [BM25Okapi, BM25L, BM25Plus])
def test_every_scorer_names_its_recipe(scorer):
    """A class name is not a recipe; the formula is identified explicitly."""
    recipe = scorer(CORPUS).recipe
    assert isinstance(recipe, Recipe)
    assert recipe.name and recipe.version
    assert recipe.identifier in RECIPES


def test_recipes_are_distinct():
    """Three formulas, three identities."""
    identifiers = {scorer(CORPUS).recipe.identifier
                   for scorer in (BM25Okapi, BM25L, BM25Plus)}
    assert len(identifiers) == 3


def test_the_okapi_recipe_records_its_specific_choices():
    """The epsilon floor and accumulated query terms are choices, not universals."""
    recipe = BM25Okapi(CORPUS).recipe
    assert "epsilon" in recipe.notes.lower() or "floor" in recipe.notes.lower()


# -- build identity --------------------------------------------------------


def test_a_build_identity_covers_recipe_analyzer_and_parameters():
    """What produced the numbers is what names the build."""
    analyzer = Analyzer(name="whitespace", version="1")
    base = lexical_identity(BM25Okapi(CORPUS).recipe, analyzer, {"k1": 1.5, "b": 0.75})
    assert base == lexical_identity(
        BM25Okapi(CORPUS).recipe, analyzer, {"b": 0.75, "k1": 1.5}
    )
    assert base != lexical_identity(
        BM25Okapi(CORPUS).recipe, analyzer, {"k1": 1.2, "b": 0.75}
    )
    assert base != lexical_identity(
        BM25L(CORPUS).recipe, analyzer, {"k1": 1.5, "b": 0.75}
    )
    assert base != lexical_identity(
        BM25Okapi(CORPUS).recipe, Analyzer(name="other"), {"k1": 1.5, "b": 0.75}
    )
