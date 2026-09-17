"""
Tokenisation execution regressions (slice S-19).

Building with a tokenizer always created ``Pool(cpu_count())``, whatever the
corpus size, so a two-document build paid for a process pool and a lambda
tokenizer failed on pickling. Parallelism is now something a caller asks for.

See Also
--------
scikitplot.rank_bm25._rank_bm25.BM25
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from .._identity import Analyzer
from .._rank_bm25 import BM25Okapi

ROOT = str(Path(__file__).resolve().parents[3])


def _in_child(body):
    """Run ``body`` in a fresh interpreter and return its JSON result."""
    script = textwrap.dedent(
        """
        import json, sys
        sys.path.insert(0, sys.argv[1])
        from scikitplot.rank_bm25._rank_bm25 import BM25Okapi
        """
    ) + textwrap.dedent(body)
    proc = subprocess.run([sys.executable, "-c", script, ROOT],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_no_process_pool_is_created_by_default():
    """A small build must not pay for a pool it never asked for."""
    result = _in_child(
        """
        import multiprocessing
        before = len(multiprocessing.active_children())
        index = BM25Okapi(["a b", "b c"], tokenizer=str.split)
        print(json.dumps({"built": index.corpus_size,
                          "children": len(multiprocessing.active_children()) - before}))
        """
    )
    assert result["built"] == 2
    assert result["children"] == 0


def test_a_lambda_tokenizer_works_in_process():
    """Picklability is a requirement of parallelism, not of tokenising."""
    index = BM25Okapi(["a b", "b c"], tokenizer=lambda text: text.split())
    assert index.doc_len == [2, 2]


def test_an_analyzer_works_in_process():
    """A declared analyzer is a bound method, which a pool could not pickle either."""
    index = BM25Okapi(["a b"], tokenizer=Analyzer(name="whitespace"))
    assert index.doc_len == [2]


def test_parallel_tokenisation_is_opt_in():
    """The caller asks for workers, and gets the same tokens either way."""
    serial = BM25Okapi(["a b", "b c", "c d"], tokenizer=str.split)
    parallel = BM25Okapi(["a b", "b c", "c d"], tokenizer=str.split, workers=2)
    assert parallel.doc_len == serial.doc_len
    assert parallel.idf.keys() == serial.idf.keys()


def test_workers_is_validated():
    """A worker count is a count, held to the shared contract."""
    with pytest.raises(ValueError):
        BM25Okapi(["a b"], tokenizer=str.split, workers=-1)
    with pytest.raises(TypeError):
        BM25Okapi(["a b"], tokenizer=str.split, workers=True)


def test_workers_one_stays_in_process():
    """Asking for one worker is asking for no pool."""
    result = _in_child(
        """
        import multiprocessing
        before = len(multiprocessing.active_children())
        BM25Okapi(["a b"], tokenizer=str.split, workers=1)
        print(json.dumps({"children": len(multiprocessing.active_children()) - before}))
        """
    )
    assert result["children"] == 0


def test_an_unpicklable_tokenizer_under_workers_says_why():
    """Opting into parallelism is opting into its requirement, and the error says so."""
    with pytest.raises(Exception) as excinfo:
        BM25Okapi(["a b"], tokenizer=lambda t: t.split(), workers=2)
    assert "pickl" in str(excinfo.value).lower() or "workers" in str(excinfo.value).lower()
