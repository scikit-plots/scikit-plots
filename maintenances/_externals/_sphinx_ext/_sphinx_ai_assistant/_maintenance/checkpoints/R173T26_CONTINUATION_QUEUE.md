# R173T26 — A visible continuation queue, and half the tokens

Status: COMPLETE

Base: R173T25 solo patch footer, same package.

## A real bug found on the way in

`Continue editing` staged a composer attachment **and** registered the file in
`_workingFileContinuations`, which R173T8's request builder then sent as a bound
`working_files` entry. On any endpoint that accepts working files, every
continued file therefore travelled **twice** — once bound to its revision and
digest, once as an unbound attachment.

That doubled the token cost of each continued file and handed the model two
copies to reconcile, which is worse than either copy alone. Exactly the waste
raised in this request, sitting one layer below where it was being looked for.

Continuation is now **intent, not transport**. Queuing registers the key; the
request builder carries the bytes, bound. The attachment path survives only as
the fallback for endpoints that cannot take working files, where it is the only
way the bytes travel at all. Mutant `continuation-sends-the-bytes-twice` forces
the old behaviour and is caught.

## Continue several at once

`Continue editing all N files` sits in the footer. It applies **every** bound
the endpoint published — file count, per-file characters, total characters —
and reports what fitted and what did not, before the request is built rather
than after it is rejected.

Files left out are named as such. A reader who cannot see what was dropped
cannot tell a short answer from an incomplete request.

## Drop what you did not mean to send

- The ⋮ item **toggles**: `Continue editing` ↔ `Stop continuing`, so the same
  place both starts and ends it.
- A **tray** under the section heading states `N files travel with your next
  message` with one `Clear`. A queue the reader cannot see is a queue they
  cannot manage, and dropping everything is the action that is most awkward
  through per-file menus.

## The composer is never overwritten

Priming an instruction only happens into an empty composer. Overwriting a
question the reader already typed would destroy their work to save them a
sentence. The wording follows the count — one file or N. Mutant
`continuation-overwrites-the-composer` removes the guard and is caught.

## Verification

- browser wrapper gate: **147/147** (42 assertions in the owning harness);
- architecture gates: **462/462**, three new mutants, all caught:
  `continuation-sends-the-bytes-twice`, `continue-all-ignores-endpoint-limits`,
  `continuation-overwrites-the-composer`.

## A harness escape bug worth noting

An assertion compared a JS string literal containing `\\u2019` against source
text holding the literal escape sequence. The harness string became the
character, the source kept the escape, and the two could never match — the
assertion failed against correct code. Rewritten to match on the ASCII portion
either side of it. Matching source text on characters that have an escaped form
is a comparison between two different encodings of the same thing.
