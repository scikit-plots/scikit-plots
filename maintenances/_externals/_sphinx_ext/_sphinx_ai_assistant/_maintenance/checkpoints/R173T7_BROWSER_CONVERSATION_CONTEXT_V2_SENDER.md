# R173T7 — Browser conversation-context v2 sender

Status: COMPLETE

Base: R173T6 snippet-to-working-file transform, same package.

## What this closes

The original Annoy workflow failed like this:

    turn 1  "generate an Annoy Python file" + page attached  -> relevant code
    turn 2  "add sys argv capability regenerate"             -> a different example

T4 made the server able to accept history. Nothing sent any. This checkpoint
makes the browser negotiate and send it, and — equally important — say what it
sent.

## Negotiation, not assumption

`_chatRequestCapsParse` reads `capabilities.chat_request` from `/health` and
returns the newest contract both sides know, plus the published history bounds.
A server advertising only the legacy `contract` string still resolves to v1, so
an older proxy keeps working with no special case at the call site.

A v2 endpoint that publishes **no usable bounds degrades to v1** rather than
having limits invented for it. Guessing high would convert a recoverable local
decision into a rejected request; guessing low would silently discard context
the server would have accepted.

`_chatContractDiscover` now returns `{contract, history}` instead of a bare
string. Cached records written before this change hold a bare string and are
still read as v1, so a warm session survives the upgrade without a re-probe.

## History construction

`_chatHistoryForRequest` selects **newest-first** and emits **oldest-first**:
when the budget cannot hold everything, the turns nearest the question are the
ones worth keeping, but the model must still read them in the order they
happened.

Nothing is summarised or elided mid-turn. A turn that does not fit whole is
dropped whole, and the number dropped is returned so the activity surface can
report it. Truncating a turn into the request would recreate the exact failure
this contract exists to remove: an answer that looks informed and is not.

Only `user` and `assistant` turns are eligible. The current question is excluded
because it travels as `user_message`.

## Honesty surfaces

The activity step was previously written before negotiation finished and said
only "Prepared request context". It is now rewritten with what is actually
being sent — resource count, page context, turn count and approximate size,
and any turns omitted to stay inside the endpoint's limits. That is the
difference between a claim and a receipt.

The first-message privacy notice now states that recent turns may be sent when
the endpoint supports it, and points at the per-turn activity panel for the
instance. A privacy notice that understates what leaves the browser is worse
than none.

## Verification

- browser wrapper gate: **141/141**;
- architecture gates: **396/396** (three new mutants, each caught);
- **end-to-end**: a `/health` document generated from the real server constants
  was fed to the browser negotiator, which selected v2 and built a two-turn
  history; the resulting body was parsed by the real server parser, which
  accepted it, fenced the history behind a nonce, emitted `['system','user']`
  roles only, and carried the prior question into the prompt. The Annoy
  follow-up now has its predecessor.

## Stale expectation corrected, not deleted

`test_index__chat_authority.mjs` pinned the exact source line
`useStructuredProxy = (proxyContract === _CHAT_CONTRACT_V1);`. The invariant it
guards — the advertised contract controls the structured path — is unchanged
and still guarded; only the literal moved. The assertion was rewritten to pin
the rule and six further assertions were added so it cannot pass vacuously.

## New mutants

- `chat-history-sent-under-legacy-contract` — history riding on v1 would never
  be fenced as untrusted evidence;
- `chat-history-accepts-any-role` — admitting system/developer/tool turns hands
  a browser transcript the ability to assert server authority;
- `chat-history-bounds-guessed-when-unpublished` — inventing limits for an
  endpoint that published none.

## Still open

- Working files travel as attachments, not yet as revision-bound
  `working_files` entries with `baseRevision`/`baseSha256`; stale-response
  protection needs that binding.
- Retry still resends text without a retained context snapshot, so
  "resend this question as-is" can still mean a different effective request.
- `revision` remains an event counter rather than a content version.
