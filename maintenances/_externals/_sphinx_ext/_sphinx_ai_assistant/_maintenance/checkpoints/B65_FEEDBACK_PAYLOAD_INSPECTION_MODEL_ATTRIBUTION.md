# B65 — Feedback Payload Inspection & Originating Model Attribution

Status: **GREEN**

## Scope

Run 46 aligns the Feedback workspace with the Dataset contribution inspection UX and
strengthens one-Q&A feedback evidence so reviewed/training-eligible feedback cannot be
created without usable originating-model attribution.

## Feedback payload inspection

The Feedback tab now exposes a dedicated **Inspect feedback payload** section using the
same Endpoint Configuration / contribution I/O primitives:

- **Inspect JSON**;
- **Copy JSON to clipboard**;
- **Download JSON file**;
- exact local payload byte size;
- adaptive JSON preview density.

These actions are browser-local. They never enable review permission, create a provider
review, or send a request. The preview is the same client object consumed by
`/v1/feedback/review`.

## Model attribution invariant

For a rendered answer index, the assistant transcript turn is the authority for model
identity. The feedback-review payload uses that turn's persisted `model` object rather
than whichever model happens to be selected later in the UI.

Reviewable feedback fails closed when the originating model cannot be established. The
server independently requires non-empty `model.provider` and `model.model`, preventing
ambiguous Q&A rows from entering native review or training eligibility.

Model identity is also included in the feedback-review semantic fingerprint, so a model
evidence change cannot be incorrectly treated as an unchanged no-op.

## Popup icons

The feedback popup no longer renders the words `Review` and `Feedback` inside the icon
slot. It now uses shared inline SVG constants:

- GitHub Octicon `comment-discussion` for review/model-improvement sharing;
- GitHub Octicon `pulse` for the Feedback workspace entry.

The constants remain local/static and use harmless visual fallbacks.

## Validator correction

While tightening feedback payload validation, Run 46 fixed an adjacent bug where the
`ratingTitle` maximum-length branch accidentally measured `ratingLabel`. `ratingTitle`
is now validated independently.

## Verification

- Run 46 Python contract: **4 passed**;
- Run 46 browser/source contract: **15/15**;
- feedback/control-plane/documentation + every registered Node harness: **83 passed**;
- full runnable non-Sphinx tree: **869 passed, 3 skipped**;
- all-inclusive container boundary: Sphinx-only failures/errors remain environmental
  (`ModuleNotFoundError: sphinx`); no non-Sphinx failure remains;
- Python/JavaScript syntax: GREEN.
