<!-- Archived verbatim from maintenances/_repo/todo/todo.md on 2026-10-26 (CI-2). -->

## Task: CI-1 - the segfault of CI run 40244733381, offline models (2026-10-26)

- Context: run 40244733381 (PR 869) stopped at 62 percent, exit 139, no
  Python frame, semantic registry test; passed locally.

### Root causes (each reproduced here, offline, CI's versions)
- `import tensorflow; import triton._C.libtriton` segfaults (tensorflow
  2.21.0, triton 3.8.0; 4 of 4), the reverse order does not. The
  `visualkeras` conftest imported TensorFlow at collection; the semantic
  test's `import sentence_transformers` reached triton. Locally TensorFlow
  is not installed. The model download is not involved (Rule 158).
- No traceback: `annoy/tests/angular_index_test.py:282` called
  `faulthandler.enable()` inside a test (Rule 159).
- `pytest-exit-code.txt` = 0 for a failing local run: `python -m cProfile`
  exits 0 whatever pytest returns (Rule 161).
- Same crash, order-dependent: two `TestTorchAdapter` tests called an unused
  `importorskip("tensorflow")`; torch tests then semantic test exit 139.
- Hidden next: `rank_bm25/_validation.py` drifted from corpus/mcp copies.

### Changes
- [x] `visualkeras` tests skip one by one unless `SKPLT_TEST_TENSORFLOW=1`
  where triton is installed; both workflows run `TENSORFLOW_SUITES` in a
  second session. `test_tensorflow` and the corpus TensorFlow adapter tests
  run in a child; `set_seed` no longer imports TensorFlow.
- [x] `scikitplot/_testing/_runtime_report.py`: `ML runtime:` and `model
  cache:` header lines; a warning names a test that loads TensorFlow into
  the main session. `test_session_diagnostics.py`: faulthandler untouched;
  no in-process TensorFlow outside `TENSORFLOW_SUITES` (AST scan).
- [x] `docker/scripts/install_models.py`, four wrappers; workflows cache the
  models, test with `HF_HUB_OFFLINE=1`. Makefile keeps pytest's status.

### Results Review - CI-1 - 2026-10-26
- Before: `pytest scikitplot/visualkeras <semantic test>` exits 139 (no
  traceback with the annoy test in the run, as in CI). After: passes; a
  stray TensorFlow test is named by the warning, the crash keeps its stack.
- CI step, full shard, CI versions, offline: no 139; 25106 passed, 20 failed
  (16 network/NLTK; colortable logger, public_api, TestLogReport: untouched
  files); main session loads no TensorFlow; TF session 47 passed.
  Changed tests 608 passed on pytest 9.0.3 and 9.1.1, serial and `-n auto`.
- Offline recorder: one model (`paraphrase-multilingual-mpnet-base-v2`).
- Not verified: a GitHub run; real downloads (no Hugging Face here).
