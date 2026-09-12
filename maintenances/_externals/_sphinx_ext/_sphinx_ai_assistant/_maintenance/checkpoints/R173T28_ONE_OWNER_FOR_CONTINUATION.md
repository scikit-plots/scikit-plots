# R173T28 — One owner for "this file travels with my next message"

Status: COMPLETE

Base: R173T27 continuation queue fixes, same package.

## The design error underneath the reported bug

"Removing the chip does not stop the continuation" was a symptom. The cause was
that one intent had **two authorities**, each true on some endpoints:

| representation | authoritative when |
|---|---|
| `_workingFileContinuations` | the endpoint accepts `working_files` |
| composer attachment chip | it does not |

Two consequences followed, neither fixable by adding a handler:

- on a working-file endpoint **no chip existed at all**, so files travelled with
  nothing in the composer to show for it — invisible attached context, which is
  the failure this run has corrected in four other places;
- removing the chip left the registry set, so on such an endpoint a file the
  reader had explicitly removed still travelled, bound to a revision, as though
  they had asked for it.

## The fix, and a correction to my own recommendation

I first proposed patching the reverse direction and staging a *virtual* chip on
working-file endpoints, and ruled out always staging on the grounds that "what
the preflight shows is not what is sent."

That objection was wrong. The same files are sent either way; only the encoding
differs, and the preflight never promised a transport. Always staging is the
better design: the chip already carries preview, remove, the attachment manager
and the privacy preflight, so making it the visible truth on every endpoint
costs no new concepts and invents no virtual items.

- **Continue always stages** and always registers. The chip is what the reader
  sees and removes, everywhere.
- **`_syncContinuationForRemovedItem`** runs inside `_removeComposerResourceItem`,
  so every path a reader can take to the same intent — the chip's ×, the
  attachment manager, `Stop continuing` — clears the registry and refreshes the
  tray from one place.
- **The duplicate transmission is suppressed at request build**, where the
  transport is finally known, and the activity receipt names what was
  suppressed and why.

## The suppression had to remove bytes, not just a descriptor

Filtering `bodyObj.resources` alone would have left the file **uploading**: the
multipart body is built from `requestResources`, a different array. That is a
worse state than the duplicate it was meant to remove — bytes on the wire with
nothing in the request describing them. Both are filtered, and the mutant
removes the byte filter specifically.

## Verification

- browser wrapper gate: **147/147** (61 assertions in the owning harness);
- architecture gates: **470/470**, one new mutant
  (`chip-removal-leaves-the-continuation-registered`) and one rewritten
  (`continuation-sends-the-bytes-twice`, now guarding the multipart filter
  rather than the removed conditional).

## A mutant that wrapped instead of removing

The first rewrite of `continuation-sends-the-bytes-twice` replaced the filter
with a version that still contained the asserted substring, so the assertion
passed and the mutant survived. Rewritten to delete the filter outright.

A mutation must remove the thing the assertion matches, not decorate it —
otherwise it tests the assertion's spelling rather than the behaviour.
