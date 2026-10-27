# Concurrency, caches and stores

Categories: Concurrency, Caches, Stores that gain deletion, SQLite

Rules keep the global number they were given; a new rule takes the next
free number (see [`../SKILL.md`](../SKILL.md)).

### Rule 65: SQLite - `with sqlite3.connect(...)` does not close the connection
- **When:** opening a `sqlite3` connection, in product code or tests.
- **Then:** `with closing(sqlite3.connect(...)) as conn, conn:`: `closing` releases the handle, the inner `with conn` keeps commit/rollback.
- **Verified by:** `python -m pytest scikitplot/_externals/_sphinx_ext/_sphinx_feedback --confcutdir scikitplot/_externals/_sphinx_ext/_sphinx_feedback -W error` on Python 3.13+, where an unclosed connection raises `ResourceWarning`.
- **Added:** 2026-10-10. Root cause: `SQLiteFeedbackStore` used the connection's context manager, which only ends the transaction; every `put`/`events` leaked a handle in the long-running service, and the suite failed at random tests under `-W error`. Rule 62 applied to product code.

### Rule 81: Caches - the key names everything that changes the value, the provider included
- **When:** a persistent cache stores the output of a pluggable function (a callable, a model, a remote API).
- **Then:**
  - The key includes a stable provider id the caller declares (`custom_fn_id`).
  - A provider without an id bypasses the persistent cache, and this is logged once.
  - A hit returns the stored value at its stored dtype, bit-identical to a miss.
  - The cache does not depend on unrelated arguments such as `input_path`.
- **Verified by:** `TestCustomProviderIdentity`, `TestCachePrecision` and `test_without_input_path_still_caches` in `corpus/_embeddings/tests/test__embedding.py`.
- **Added:** 2026-10-12. Root cause: the key was `texts + model_name + backend`. Provider B was served provider A's vectors and never called. A float64 engine got float32 values back from a hit.

### Rule 88: Concurrency - a check, the action it permits and the record of the action happen together
- **When:** a method checks state, acts (a system call, a load) and then records the result, and another thread can call into the same object.
- **Then:**
  - Keep the GIL (or one lock) from the check to the record. Release it only around work whose effect no other method reads.
  - Lazy initialisation publishes its "done" flag last, under a lock, after the outcome (success or a remembered failure) is stored.
  - `KeyboardInterrupt` and `SystemExit` are not outcomes: they leave the flag unset, so the next call tries again.
- **Verified by:**
  - `test_mprotect_and_an_export_on_another_thread_do_not_race`, a gated child process. Against the first fix it failed at attempt 0.
  - `test_concurrent_first_calls_share_one_load`, `test_an_interrupted_first_load_is_tried_again` and `test_a_broken_package_import_falls_back_and_says_so`.
- **Added:** 2026-10-12.
- **Root cause (found by an independent review of Run 1):**
  - `mprotect` checked the views, released the GIL for the system call, and only then recorded `_prot`. Another thread exported writable views over pages that were already read-only.
  - `SemanticChunker` set "loaded" before loading. A non-`ImportError` import failure, an interrupt or a second thread left a chunker whose every later call raised `TypeError`.

### Rule 109: Concurrency - per-process state is fork-safe per instance, not by a module hook
- **When:** an object keeps a sequence, a lock or a seed (an id generator, a counter, a pool).
- **Then:** record `os.getpid()` in the instance and reset state and lock when it differs, before taking the lock; `os.register_at_fork` on one module instance leaves every other instance continuing the parent's sequence.
- **Verified by:** `test_a_forked_child_starts_a_new_sequence[private]` and `test_a_lock_held_at_the_fork_does_not_hang_the_child` (fork in a fresh single-threaded child interpreter).
- **Added:** 2026-10-13. Root cause: the UUIDv7 generator reset only the module instance after fork; a caller's instance repeated the parent's 96-bit prefix, and a lock held at the fork hung the child.

### Rule 118: Stores that gain deletion - every reader treats a vanished entry as a miss, and a bound never evicts for an entry it then refuses
- **When:** a cache or store that only ever grew gains eviction, expiry or cleanup (quota, LRU, TTL), from this process or another.
- **Then:** (1) read each entry with one operation inside a `try` (no `exists()` check first); `FileNotFoundError` anywhere on the read path is a miss, other `OSError` a logged miss; (2) decide whether the new entry fits before evicting anything, and never evict the entry just stored to make room; (3) count only what the scan saw; (4) a bound is best effort and never fails the call that stored the entry; test each case with a racing or refusing filesystem stub.
- **Verified by:** `TestConcurrency` and `TestBounds` in `corpus/_embeddings/tests/test__cache_policy.py`.
- **Added:** 2026-10-16. Root cause: `_load_from_cache` checked `exists()` and then called `stat()` outside its `try`, harmless while nothing deleted entries; 3c added eviction, and the independent review measured 79 `FileNotFoundError`s in 18k threaded calls. The first eviction loop also sorted the new entry last and evicted down to it, so one oversized result emptied the cache.
