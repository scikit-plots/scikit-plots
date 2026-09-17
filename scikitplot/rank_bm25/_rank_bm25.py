# scikitplot/rank_bm25/_rank_bm25.py
#
# Authors: D. Brown
# SPDX-License-Identifier: Apache License 2.0

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined.

Here we implement all the BM25 variations mentioned.
"""  # ruff: ignore[missing-blank-line-after-summary]

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import pathlib
import shutil
import uuid
from multiprocessing import Pool

import numpy as np

from ._identity import RECIPES, Analyzer, Recipe, lexical_identity
from ._validation import require_count, require_index

__all__ = [  # ruff: ignore[unsorted-dunder-all]
    "BM25",
    "BM25Okapi",
    "BM25L",
    "BM25Plus",
    # "BM25Adpt",
    # "BM25T",
]


def _check_parameter(
    name, value, *, minimum=None, maximum=None, exclusive_minimum=False
):
    """
    Validate one scoring parameter against its documented domain.

    Parameters
    ----------
    name : str
        Parameter name, used in the message.
    value : object
        Supplied value.
    minimum, maximum : float or None, optional
        Inclusive bounds, unless ``exclusive_minimum`` is set.
    exclusive_minimum : bool, optional
        Whether ``minimum`` itself is outside the domain.

    Returns
    -------
    float
        The value, as a float.

    Raises
    ------
    TypeError
        If the value is not a real number. ``bool`` is rejected explicitly
        because it is an ``int`` subclass and ``True`` is not a weight.
    ValueError
        If the value is outside the documented domain, or is not finite.

    Notes
    -----
    **Developer.** Validating here rather than at score time is what makes the
    mistake attributable: an out-of-domain ``k1`` previously reached the
    arithmetic and returned ``NaN`` for every document lacking the query term,
    which looks like a data problem rather than a call-site one.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number!r}.")
    if minimum is not None:
        too_small = number <= minimum if exclusive_minimum else number < minimum
        if too_small:
            bound = "greater than" if exclusive_minimum else "at least"
            raise ValueError(f"{name} must be {bound} {minimum}, got {number!r}.")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {number!r}.")
    return number


def _require_token_sequences(corpus):
    """
    Return ``corpus`` as a list, refusing anything that is not tokenised.

    Parameters
    ----------
    corpus : iterable
        Documents, each a sequence of token strings.

    Returns
    -------
    list
        The documents, materialised so they can be counted and iterated twice.

    Raises
    ------
    TypeError
        If a document is a ``str`` or ``bytes``. Both are iterable, so the
        element-wise loop below would index them one character at a time and
        build a plausible but meaningless character index.
    ValueError
        If the corpus holds no documents, or holds documents but no tokens.
        Each is a distinct mistake with a distinct fix, so each is named
        separately rather than surfacing as a division by zero.

    Notes
    -----
    **User.** Pass ``[doc.split() for doc in texts]``, or pass the raw texts
    together with ``tokenizer=``. An empty document is fine as long as some
    document in the corpus has tokens.
    """
    documents = list(corpus)
    if not documents:
        raise ValueError(
            "corpus must contain at least one document; an empty corpus has no "
            "collection statistics to score against."
        )
    for position, document in enumerate(documents):
        if isinstance(document, (str, bytes, bytearray)):
            raise TypeError(
                f"document at position {position} is a "
                f"{type(document).__name__}, not a sequence of tokens; "
                "iterating it would index single characters. Pass "
                "[doc.split() for doc in texts], or pass the raw texts with "
                "tokenizer=."
            )
    for position, document in enumerate(documents):
        for token in document:
            if not isinstance(token, str):
                raise TypeError(
                    f"document at position {position} contains a "
                    f"{type(token).__name__} token ({token!r}); tokens must be "
                    "strings. A non-string token survives in memory but not "
                    "across a save: JSON records term keys as strings, so the "
                    "same query stops matching after a reload."
                )
    if not any(len(document) for document in documents):
        raise ValueError(
            f"none of the {len(documents)} documents contains a token; BM25 "
            "needs at least one token to derive an average document length."
        )
    return documents


def _require_matching_documents(corpus_size, documents):
    """
    Check that ``documents`` describes the same corpus the index was built on.

    Parameters
    ----------
    corpus_size : int
        Number of documents the index was built over.
    documents : sized
        Caller-supplied documents, in index row order.

    Raises
    ------
    ValueError
        If the two sizes differ, which means row offsets from this index would
        not name the documents the caller intends to return.

    Notes
    -----
    **Developer.** Expressed as a raised exception rather than ``assert`` so the
    guard survives ``python -O``, which removes assertions. This is the only
    check standing between a mismatched sequence and silently mislabelled
    results.
    """
    supplied = len(documents)
    if supplied != corpus_size:
        raise ValueError(
            f"documents has {supplied} entries but the index was built over "
            f"{corpus_size}; row offsets from this index would not name the "
            "documents you intend to return."
        )


def _require_valid_positions(doc_ids, corpus_size):
    """
    Check that every requested position exists in the index.

    Parameters
    ----------
    doc_ids : iterable
        Requested row positions.
    corpus_size : int
        Number of rows the index holds.

    Raises
    ------
    TypeError
        If a position is not an integer, or is a ``bool``.
    IndexError
        If a position is negative or not below ``corpus_size``.

    Notes
    -----
    **Developer.** The domain comes from
    :func:`scikitplot._utils._indexing.require_index`, so this boundary and the
    corpus artifact answer the same question the same way. Expressed as a
    raised exception rather than ``assert`` so the guard survives ``python -O``.
    """
    for position in doc_ids:
        require_index(position, corpus_size, name="document position")


_INDEX_NAME = "index.json"
_INDEX_SCHEMA = "bm25-index/1"
_POINTER_NAME = "current.json"
_RETAINED_GENERATIONS = 2


class BM25:
    #: The scoring formula this class implements. Subclasses override it; the
    #: class name is not the recipe, because "BM25Okapi" implies a standard the
    #: implementation makes specific choices within.
    RECIPE_ID: str = "okapi-epsilon-floor/1"

    def __init__(self, corpus, tokenizer=None, doc_ids=None, workers=None):
        self._workers = (
            None if workers is None else require_count(workers, name="workers")
        )
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = self._record_analyzer(tokenizer)

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._record_doc_ids(doc_ids)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in _require_token_sequences(corpus):
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for (
                word,
                _freq,
            ) in frequencies.items():  # ruff: ignore[incorrect-dict-iterator]
                try:
                    nd[word] += 1
                except KeyError:  # ruff: ignore[try-except-in-loop]
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _record_analyzer(self, tokenizer):
        """
        Record a declared analyzer, and return the callable to tokenise with.

        Parameters
        ----------
        tokenizer : Analyzer, callable or None
            A declared :class:`~scikitplot.rank_bm25._identity.Analyzer`, a bare
            callable, or ``None`` when the caller supplies tokens directly.

        Returns
        -------
        callable or None
            What to call on each document.

        Notes
        -----
        **Developer.** A bare callable is still accepted, and still leaves the
        build unidentifiable -- that is the point of preferring an Analyzer, not
        a reason to refuse working code.
        """
        if isinstance(tokenizer, Analyzer):
            self._analyzer = tokenizer
            return tokenizer.tokenize
        self._analyzer = None
        return tokenizer

    def _tokenize_corpus(self, corpus):
        """
        Tokenise ``corpus``, in process unless the caller asked for workers.

        Notes
        -----
        **User.** Tokenising runs in this process by default. Pass ``workers=N``
        to spread it across processes; that also requires the tokenizer to be
        picklable, which a lambda or a closure is not.

        **Developer.** This previously created ``Pool(cpu_count())``
        unconditionally, so a two-document build started one process per core
        and a lambda tokenizer failed on pickling with no way to opt out. The
        cost and the picklability requirement now both follow from a choice the
        caller made.
        """
        workers = getattr(self, "_workers", None)
        if not workers or workers <= 1:
            return [self.tokenizer(document) for document in corpus]
        try:
            with Pool(workers) as pool:
                return pool.map(self.tokenizer, corpus)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            raise type(exc)(
                f"{exc}. Parallel tokenisation requires a picklable tokenizer; "
                "a lambda or a closure is not. Drop workers= to tokenise in "
                "this process."
            ) from exc

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def _require_query_tokens(self, query):
        """
        Refuse a query that is not a sequence of string tokens.

        Notes
        -----
        **User.** Pass ``query.split()``, not ``query``. A bare string is
        iterable, so it was scored as one token per character.
        """
        if isinstance(query, (str, bytes, bytearray)):
            raise TypeError(
                f"query is a {type(query).__name__}, not a sequence of tokens; "
                "it would be scored one character at a time. Pass "
                "query.split(), or tokenise with the same analyzer used to "
                "build the index."
            )
        tokens = list(query)
        for token in tokens:
            if not isinstance(token, str):
                raise TypeError(
                    f"query contains a {type(token).__name__} token "
                    f"({token!r}); tokens must be strings."
                )
        return tokens

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    @property
    def recipe(self) -> Recipe:
        """The named, versioned scoring formula this index used."""
        return RECIPES[self.RECIPE_ID]

    @property
    def analyzer(self) -> Analyzer | None:
        """The declared analyzer, or ``None`` when tokens were supplied directly."""
        return getattr(self, "_analyzer", None)

    @property
    def parameters(self) -> dict:
        """The scoring parameters that belong to this index's identity."""
        return {
            name: getattr(self, name)
            for name in ("k1", "b", "epsilon", "delta")
            if hasattr(self, name)
        }

    @property
    def identity(self) -> str:
        """
        Digest of what produced this index: recipe, analyzer and parameters.

        Notes
        -----
        **User.** Two indexes sharing this value were built the same way. It does
        not cover the documents: that is what the corpus records, and folding it
        in here would make a re-tokenisation and a new document look like the
        same kind of change.
        """
        return lexical_identity(self.recipe, self.analyzer, self.parameters)

    def _record_doc_ids(self, doc_ids):
        """
        Record the stable identity of each row, if the caller supplied one.

        Parameters
        ----------
        doc_ids : sequence of str or None
            One identity per document, in row order.

        Raises
        ------
        ValueError
            If the count does not match the corpus, or an identity repeats. One
            identity naming two rows is the same defect the artifact sidecar
            refuses: a hit could then name either row and the caller could not
            tell which.

        Notes
        -----
        **Developer.** Refused here, at the point the mistake is made, rather
        than at query time when the offending call is long gone.
        """
        if doc_ids is None:
            self._doc_ids = None
            return
        ids = tuple(str(value) for value in doc_ids)
        if len(ids) != self.corpus_size:
            raise ValueError(
                f"doc_ids has {len(ids)} entries but the corpus has "
                f"{self.corpus_size}; every row needs exactly one identity."
            )
        seen = set()
        repeated = sorted({i for i in ids if i in seen or seen.add(i)})
        if repeated:
            raise ValueError(
                f"doc_ids repeats {repeated}; one identity cannot name two rows, "
                "because a hit could then name either and the caller could not "
                "tell which."
            )
        self._doc_ids = ids

    @property
    def doc_ids(self):
        """Tuple of str or None: Row identities, in row order."""
        return self._doc_ids

    def _ranked_positions(self, query, n):
        """
        Return the ``n`` best row positions, ties broken by position.

        Notes
        -----
        **Developer.** ``argsort`` is ascending and stable, so reversing it also
        reverses ties: three identical documents came back as rows 2, 1, 0. The
        negated scores sort descending while leaving equal scores in ascending
        row order, which is a tie-break someone chose rather than a side effect
        of how the sort was flipped.
        """
        scores = self.get_scores(query)
        order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
        return [int(position) for position in order[:n]]

    def get_top_ids(self, query, n=5):
        """
        Return the ``n`` best ``(doc_id, score)`` pairs.

        Parameters
        ----------
        query : sequence of str
            Query tokens.
        n : int, optional
            Maximum number of hits.

        Returns
        -------
        list of tuple
            ``(doc_id, score)`` in descending score, ties in row order.

        Raises
        ------
        ValueError
            If the index holds no identities, or ``n`` is not a valid count.

        Notes
        -----
        **User.** This is the binding to prefer. :meth:`get_top_n` maps row
        offsets into a sequence you pass at query time, which is only correct
        while that sequence is in the same order as the corpus the index was
        built from -- a precondition nothing can check for you.
        """
        n = require_count(n, name="n")
        if self._doc_ids is None:
            raise ValueError(
                "this index holds no doc_ids, so a hit cannot name a document; "
                "pass doc_ids= when building it, or use get_top_n with a "
                "sequence in corpus order."
            )
        sparse = self._sparse_scores(query)
        ranked = sorted(sparse.items(), key=lambda item: (-item[1], item[0]))
        return [(self._doc_ids[row], float(score)) for row, score in ranked[:n]]

    # -- sparse evaluation -------------------------------------------------

    @property
    def _postings(self):
        """
        dict: term -> list of ``(row, frequency)``, built once on demand.

        Notes
        -----
        **Developer.** Derived from ``doc_freqs``, so it stores no information
        the index did not already hold; it inverts it. Built lazily because an
        index that is only ever asked for ``get_scores`` should not pay for it.
        """
        cached = getattr(self, "_postings_cache", None)
        if cached is None:
            cached = {}
            for row, frequencies in enumerate(self.doc_freqs):
                for term, count in frequencies.items():
                    cached.setdefault(term, []).append((row, count))
            self._postings_cache = cached
        return cached

    def candidate_count(self, query):
        """
        Number of documents that contain at least one query term.

        Parameters
        ----------
        query : sequence of str
            Query tokens.

        Returns
        -------
        int
            Documents the sparse path will score.

        Notes
        -----
        **User.** This is what a query actually costs. ``get_scores`` returns one
        value per document in the corpus whatever the query, which is a
        diagnostic rather than a retrieval path.
        """  # ruff: ignore[non-imperative-mood]
        rows = set()
        for term in query:
            rows.update(row for row, _ in self._postings.get(term, ()))
        return len(rows)

    def _sparse_scores(self, query):
        """
        Return ``{row: score}`` for documents containing a query term.

        Notes
        -----
        **Developer.** Identical arithmetic to :meth:`get_scores`, evaluated only
        where a term occurs. A document with no query term scores zero under
        every recipe here, and the two paths are held to agreement by test.
        """
        doc_len = self.doc_len
        scores: dict[int, float] = {}
        for term in query:
            idf = self.idf.get(term) or 0
            for row, frequency in self._postings.get(term, ()):
                contribution = self._term_contribution(idf, frequency, doc_len[row])
                scores[row] = scores.get(row, 0.0) + contribution
        return scores

    def _term_contribution(self, idf, frequency, length):
        """One term's contribution to one document, per this recipe."""
        raise NotImplementedError

    # -- persistence -------------------------------------------------------

    def _state(self):
        """Return the JSON-serialisable state of this index."""
        analyzer = self.analyzer
        return {
            "schema": _INDEX_SCHEMA,
            "recipe": self.RECIPE_ID,
            "identity": self.identity,
            "parameters": self.parameters,
            "analyzer": (
                None
                if analyzer is None
                else dataclasses.asdict(analyzer)
                | {"stopwords": sorted(analyzer.stopwords)}
            ),
            "corpus_size": self.corpus_size,
            "avgdl": self.avgdl,
            "doc_len": list(self.doc_len),
            "doc_freqs": [dict(freqs) for freqs in self.doc_freqs],
            "idf": dict(self.idf),
            "average_idf": getattr(self, "average_idf", None),
            "doc_ids": None if self.doc_ids is None else list(self.doc_ids),
        }

    def save(self, directory):
        """
        Publish this index as a generation behind a pointer.

        Parameters
        ----------
        directory : str or path-like
            Index root. Created if absent.

        Returns
        -------
        pathlib.Path
            The generation directory written.

        Raises
        ------
        OSError
            If the generation could not be written. Nothing is published in that
            case, and any previously published generation is untouched.

        Notes
        -----
        **User.** A saved index reloads without the corpus: the statistics are
        what get published, not the documents.

        **Developer.** Same shape as the corpus artifact and the Annoy bundle:
        candidate, verify by reloading, then move the pointer. The three carry
        their own implementation rather than a shared one, which is the cost of
        each submodule standing alone; the shape is deliberately identical so a
        reader of one recognises the others.
        """
        if self.tokenizer is not None and self.analyzer is None:
            raise ValueError(
                "this index was built with a bare callable tokenizer, which has "
                "no identity that survives the process: two different "
                "tokenizers would produce one durable identity for two "
                "different indexes. Rebuild with an Analyzer before saving, or "
                "keep the index in memory."
            )
        root = pathlib.Path(os.fspath(directory))
        root.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._state(), indent=2, sort_keys=True)
        # The generation is named for the state it holds, not for `identity`.
        # identity answers "how was this built" and deliberately excludes the
        # documents, so naming a generation after it made two different corpora
        # built the same way share one directory: the second save found the
        # directory present, wrote nothing, and pointed at the first build.
        build = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        generation = root / f"generation-{build}"
        if not generation.is_dir():
            candidate = root / f".candidate-{uuid.uuid4().hex}"
            try:
                candidate.mkdir(parents=True)
                (candidate / _INDEX_NAME).write_text(payload, encoding="utf-8")
                type(self).load(candidate)
                os.replace(candidate, generation)
            except BaseException:
                shutil.rmtree(candidate, ignore_errors=True)
                raise
        pointer = json.dumps({"generation": generation.name}, indent=2)
        (root / _POINTER_NAME).write_text(pointer, encoding="utf-8")
        # Keep the current generation and the most recent other one, so a reader
        # that opened the previous build keeps it. Counting inside the loop while
        # removing directories, as an earlier version of this did, prunes a
        # different set depending on iteration order.
        others = sorted(
            (
                child
                for child in root.iterdir()
                if child.is_dir()
                and child.name.startswith("generation-")
                and child.name != generation.name
            ),
            key=lambda child: child.stat().st_mtime,
            reverse=True,
        )
        for stale in others[_RETAINED_GENERATIONS - 1 :]:
            shutil.rmtree(stale, ignore_errors=True)
        return generation

    @classmethod
    def load(cls, directory):
        """
        Load a published index, or a generation directory directly.

        Parameters
        ----------
        directory : str or path-like
            Index root, or a generation directory.

        Returns
        -------
        BM25
            The reloaded index, ready to query.

        Raises
        ------
        ValueError
            If the root has no readable pointer, or the state is not from this
            recipe. A root with no pointer is refused rather than guessed at.
        """
        root = pathlib.Path(os.fspath(directory))
        if not (root / _INDEX_NAME).is_file():
            pointer = root / _POINTER_NAME
            if not pointer.is_file():
                raise ValueError(
                    f"{root} has no {_POINTER_NAME}; there is no way to know "
                    "which generation is current."
                )
            name = json.loads(pointer.read_text(encoding="utf-8"))["generation"]
            root = root / name
        state = json.loads((root / _INDEX_NAME).read_text(encoding="utf-8"))
        if state.get("schema") != _INDEX_SCHEMA:
            raise ValueError(
                f"index at {root} was written under schema "
                f"{state.get('schema')!r}, not {_INDEX_SCHEMA!r}."
            )
        obj = cls.__new__(cls)
        obj.corpus_size = int(state["corpus_size"])
        obj.avgdl = state["avgdl"]
        obj.doc_len = list(state["doc_len"])
        obj.doc_freqs = [dict(f) for f in state["doc_freqs"]]
        for frequencies in obj.doc_freqs:
            for term in frequencies:
                if not isinstance(term, str):
                    raise ValueError(  # ruff: ignore[type-check-without-type-error]
                        f"persisted term {term!r} is not a string; the index "
                        "was written by an incompatible writer."
                    )
        obj.idf = dict(state["idf"])
        obj.tokenizer = None
        obj._postings_cache = None
        if state.get("average_idf") is not None:
            obj.average_idf = state["average_idf"]
        for name, value in state["parameters"].items():
            setattr(obj, name, value)
        declared = state.get("analyzer")
        obj._analyzer = (
            None
            if declared is None
            else Analyzer(**{**declared, "stopwords": frozenset(declared["stopwords"])})
        )
        ids = state.get("doc_ids")
        obj._doc_ids = None if ids is None else tuple(ids)
        if obj.tokenizer is None and obj._analyzer is not None:
            obj.tokenizer = obj._analyzer.tokenize
        return obj

    def get_top_n(self, query, documents, n=5):

        _require_matching_documents(self.corpus_size, documents)
        n = require_count(n, name="n")

        return [documents[p] for p in self._ranked_positions(query, n)]


class BM25Okapi(BM25):
    def _term_contribution(self, idf, frequency, length):
        """Okapi contribution for one term in one document."""
        return (idf * frequency * (self.k1 + 1)) / (
            frequency + self.k1 * (1 - self.b + self.b * length / self.avgdl)
        )

    def __init__(  # ruff: ignore[too-many-positional-arguments]
        self,
        corpus,
        tokenizer=None,
        k1=1.5,
        b=0.75,
        epsilon=0.25,
        doc_ids=None,
        workers=None,
    ):
        self.k1 = _check_parameter("k1", k1, minimum=0, exclusive_minimum=True)
        self.b = _check_parameter("b", b, minimum=0, maximum=1)
        self.epsilon = _check_parameter("epsilon", epsilon, minimum=0)
        super().__init__(corpus, tokenizer, doc_ids=doc_ids, workers=workers)

    def _calc_idf(self, nd):
        """
        Calculate frequencies of terms in documents and in corpus.

        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        ATIRE BM25 variant uses an idf function which uses a log(idf) score.

        To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        query = self._require_query_tokens(query)
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                q_freq
                * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs.
        """
        _require_valid_positions(doc_ids, len(self.doc_freqs))
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (
                q_freq
                * (self.k1 + 1)
                / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            )
        return score.tolist()


class BM25L(BM25):
    RECIPE_ID: str = "bm25l-delta/1"

    def _term_contribution(self, idf, frequency, length):
        """BM25L contribution for one term in one document."""
        ctd = frequency / (1 - self.b + self.b * length / self.avgdl)
        return (idf * (self.k1 + 1) * (ctd + self.delta)) / (self.k1 + ctd + self.delta)

    def __init__(  # ruff: ignore[too-many-positional-arguments]
        self,
        corpus,
        tokenizer=None,
        k1=1.5,
        b=0.75,
        delta=0.5,
        doc_ids=None,
        workers=None,
    ):
        # Algorithm specific parameters
        self.k1 = _check_parameter("k1", k1, minimum=0, exclusive_minimum=True)
        self.b = _check_parameter("b", b, minimum=0, maximum=1)
        self.delta = _check_parameter("delta", delta, minimum=0)
        super().__init__(corpus, tokenizer, doc_ids=doc_ids, workers=workers)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf

    def get_scores(self, query):
        query = self._require_query_tokens(query)
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (
                (self.idf.get(q) or 0)
                * (self.k1 + 1)
                * (ctd + self.delta)
                / (self.k1 + ctd + self.delta)
            )
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs.
        """
        _require_valid_positions(doc_ids, len(self.doc_freqs))
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (
                (self.idf.get(q) or 0)
                * (self.k1 + 1)
                * (ctd + self.delta)
                / (self.k1 + ctd + self.delta)
            )
        return score.tolist()


class BM25Plus(BM25):
    RECIPE_ID: str = "bm25plus-delta/1"

    def _term_contribution(self, idf, frequency, length):
        """BM25+ contribution for one term in one document."""
        return idf * (
            self.delta
            + (frequency * (self.k1 + 1))
            / (self.k1 * (1 - self.b + self.b * length / self.avgdl) + frequency)
        )

    def __init__(  # ruff: ignore[too-many-positional-arguments]
        self,
        corpus,
        tokenizer=None,
        k1=1.5,
        b=0.75,
        delta=1,
        doc_ids=None,
        workers=None,
    ):
        # Algorithm specific parameters
        self.k1 = _check_parameter("k1", k1, minimum=0, exclusive_minimum=True)
        self.b = _check_parameter("b", b, minimum=0, maximum=1)
        self.delta = _check_parameter("delta", delta, minimum=0)
        super().__init__(corpus, tokenizer, doc_ids=doc_ids, workers=workers)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq)
            self.idf[word] = idf

    def get_scores(self, query):
        query = self._require_query_tokens(query)
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (
                self.delta
                + (q_freq * (self.k1 + 1))
                / (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq)
            )
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs.
        """
        _require_valid_positions(doc_ids, len(self.doc_freqs))
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (
                self.delta
                + (q_freq * (self.k1 + 1))
                / (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq)
            )
        return score.tolist()


# BM25Adpt and BM25T are a bit more complicated than the previous algorithms here. Here a term-specific k1
# parameter is calculated before scoring is done

# class BM25Adpt(BM25):
#     def __init__(self, corpus, k1=1.5, b=0.75, delta=1):
#         # Algorithm specific parameters
#         self.k1 = k1
#         self.b = b
#         self.delta = delta
#         super().__init__(corpus)
#
#     def _calc_idf(self, nd):
#         for word, freq in nd.items():
#             idf = math.log((self.corpus_size + 1) / freq)
#             self.idf[word] = idf
#
#     def get_scores(self, query):
#         score = np.zeros(self.corpus_size)
#         doc_len = np.array(self.doc_len)
#         for q in query:
#             q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
#             score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
#                                                (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
#         return score
#
#
# class BM25T(BM25):
#     def __init__(self, corpus, k1=1.5, b=0.75, delta=1):
#         # Algorithm specific parameters
#         self.k1 = k1
#         self.b = b
#         self.delta = delta
#         super().__init__(corpus)
#
#     def _calc_idf(self, nd):
#         for word, freq in nd.items():
#             idf = math.log((self.corpus_size + 1) / freq)
#             self.idf[word] = idf
#
#     def get_scores(self, query):
#         score = np.zeros(self.corpus_size)
#         doc_len = np.array(self.doc_len)
#         for q in query:
#             q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
#             score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
#                                                (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
#         return score
