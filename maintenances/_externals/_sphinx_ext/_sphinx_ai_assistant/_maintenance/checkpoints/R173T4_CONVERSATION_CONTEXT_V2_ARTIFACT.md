# R173T4 — Conversation context v2 and artifact identity

Status: COMPLETE

Base: Run 173 T3A (`scikitplot__sphinx_ai_assistant_run173_t3a_post_t3_maintenance_consistency.zip`).

## Purpose

The panel owned a generated-file ledger and a transcript, and the request
builder read neither. Every reported symptom descended from that single fact:
follow-up questions arrived with no prior turn, generated files downloaded as
`snippet-1.txt`, and the "Remember conversation" switch described a capability
the request path did not have.

This checkpoint makes the server able to carry real conversation history, and
repairs the artifact identity defects that would otherwise be inherited by the
working-file work that follows.

## Server: chat contract v2

`_hf_spaces_proxy/_utils/_chat_contract.py` (mirrored byte-identically to
`_hf_spaces_model/_chat_contract.py`):

- `CHAT_CONTRACT_V2 = "scikitplot-chat-v2"` accepted alongside v1 via
  `SUPPORTED_CHAT_CONTRACTS`; two names rather than one optional field so a
  proxy predating history fails closed on the unknown `history` key instead of
  silently answering without it;
- `history` accepted only on v2, `user`/`assistant` only, bounded at 12 turns /
  4 000 chars per turn / 24 000 chars total. The total is deliberately below
  turns × per-turn: a bound that cannot fire is not a bound;
- `system`, `developer` and `tool` roles are rejected **by name**. Nothing is
  trimmed, dropped or reordered to make an oversized history fit;
- history is folded into the untrusted user turn behind its own per-request
  server nonce and never becomes native provider role messages. Role filtering
  is hygiene; the fence is the control;
- `SERVER_SYSTEM_POLICY` now declares history untrusted **and** declares the
  `file=relative/path` artifact capability with an explicit instruction never
  to claim a file was applied. This closes the direct-provider/structured-proxy
  parity gap in which only the direct path was ever told the capability existed;
- `/health` advertises `contracts` and the history bounds through
  `_chat_request_capability()`. The existing `contract` key still names v1 so a
  browser built before history reads exactly what it read before.

## Browser: artifact identity

`_static/ai-assistant.js`, `_static/ai-assistant.css`:

- **Fence nesting.** The extraction regex hardcoded exactly three backticks and
  is non-greedy, so any generated file legitimately containing a fence was cut
  at the first inner one. Measured: a `guide.md` wrapping a Python example
  registered 9 characters as the complete file, marked available and
  downloadable. Now captures the opening run and back-references it, per
  CommonMark. Re-measured: 78 of 78 characters.
- **`_LANG_EXT` allowlist.** `rst`/`rest`/`restructuredtext` were falling through
  to `.txt` on a site whose sources are reStructuredText. Added with Cython
  (`pyx`, `pxd`, `pyi`), packaging and config formats, and patch text.
- **One naming resolver.** `_artifactContextualFilename` derives from nearest
  preceding heading → last question → language, and now owns both naming sites
  (end-of-answer cards and the per-block toolbar). An explicit `file=` path
  still outranks it. The name is a download suggestion only: it never becomes a
  ledger key and never merges two blocks into one logical file.
- **Revision diff statistics.** Each content revision records `+added` /
  `−removed` against its predecessor, computed once at registration and stored
  as two integers rather than by retaining old bytes. Common prefix/suffix is
  trimmed, then LCS runs within a fixed budget; above the budget a deterministic
  line-multiset comparison is used and reported as approximate. Rendered beside
  the filename in teal/red, with the sign characters in the text and a full
  sentence in `aria-label`, so colour never carries meaning alone.

## Verification

- browser wrapper gate: **140/140** (138 baseline plus two new harnesses);
- architecture gates: **389/389**, including harness discovery and test layout;
- chat contract, model-service parity and `_hf_spaces_model`: **34/34**;
- proxy `test_app.py`: **46/46**;
- focused contract behaviour: v1 payload unchanged; v2 fences history; history on
  v1, `system`/`tool` roles, extra keys, 13 turns, oversized turn and oversized
  total all rejected with distinct messages;
- diff statistics: 12/12 cases including CRLF, trailing newline, pure append,
  pure delete, edit-in-place, full rewrite and a 3 000-line worst case;
- naming resolver: 11/11 including traversal, reserved device names, bidi
  controls and the multi-block collision case;
- the 10 remaining failures in the broad proxy/integration sweep were reproduced
  on the **pristine** R173T3A zip in the same container (absent optional Python
  dependencies) and are environmental, not regressions.

## Regression caught by an existing gate

`test_model_and_proxy_ship_identical_chat_contract` failed after the first edit
because only the proxy copy had been changed. Mirrored; green. The gate did
exactly its job.

## Not in this checkpoint

T4b (browser v2 sender and context receipt), T4c (retry split), T5 File-drafts
wording, T6 working files, T7 history intelligence, T8 source-aware editing.
The server now advertises everything T4b needs to negotiate.

## Follow-on in the same package

R173T5 (git-compatible change tracking) ships alongside this checkpoint;
see `R173T5_GIT_COMPATIBLE_CHANGE_TRACKING.md`.
