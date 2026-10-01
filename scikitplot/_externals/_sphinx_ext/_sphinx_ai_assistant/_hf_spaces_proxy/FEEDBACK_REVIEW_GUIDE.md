# Feedback review guide

The current system has three deliberately separate feedback surfaces. They do not
share consent, payload contracts, browser credentials, or durable identity.

| Surface | Route | Content | Authority |
| --- | --- | --- | --- |
| AI Assistant local rating | none | rating + optional local note | browser tab only |
| Generic documentation page feedback | `POST /v1/feedback` | `page.feedback-request.v1` | `_sphinx_feedback` service |
| AI Assistant Q&A maintainer review | `POST /v1/feedback/review` | one Q&A + rating + optional note | provider-native review |

There is no anonymous Assistant rating-telemetry transport. A local Assistant rating
never falls through to `/v1/feedback` and never needs a telemetry-consent switch.

## Assistant local ratings

Quick and detailed rating controls update synchronized local UI state. They may also
emit the bounded same-origin page-integration projection when the reader has explicitly
enabled that separate permission. Page integration is not a network submission API.

## Explicit Q&A maintainer review

`/v1/feedback/review` is content-bearing and therefore has its own versioned reader
permission. The browser shows the exact review payload before submission. The service
requires the current review contract and current consent version, stores a bounded
private recovery receipt, and opens or updates one provider-native review for the same
logical feedback lineage.

A provider review is not training eligibility. Only an authorized merge to the fixed
canonical branch makes the reviewed row eligible. Closing/rejecting a review does not.
Withdrawal removes the active reviewed view according to provider/lifecycle semantics;
it does not claim forensic deletion of immutable provider history or backups.

## Generic page feedback

`POST /v1/feedback` belongs to `_sphinx_feedback` and accepts only
`page.feedback-request.v1`. The route is contract-strict: Assistant-local payloads,
telemetry-shaped payloads, and unknown contracts receive a 422 response instead of
being interpreted under another feedback model.

Generic feedback is anonymous by construction. A fresh random event nonce represents
one reaction, not a person. Durable event JSON contains no browser, device, referrer,
account, session, cookie, or rate-limit identity fields. The service may derive a
transient keyed pseudonym from network information for abuse control; that pseudonym
is not part of the feedback event or provider review.

## Storage and provider review

Generic page feedback uses `FEEDBACK_STORAGE_TARGETS` and `FEEDBACK_REVIEW_MODE`.
Assistant Q&A review and dataset contribution use the record-storage/review authorities
owned by the HF Space proxy. Provider credentials are server-only; static Sphinx pages
never receive GitHub/Hugging Face/GitLab/Bitbucket write tokens.

Exactly one Primary determines accepted state. Optional Mirrors are durability copies;
a degraded mirror cannot turn a committed Primary write into an uncommitted event.

## Troubleshooting

### Generic page feedback returns 422

Inspect the request contract. `/v1/feedback` accepts only `page.feedback-request.v1`.
If the caller is trying to share an Assistant Q&A with maintainers, it is using the
wrong route; use `/v1/feedback/review` through the Assistant review workflow.

### A local Assistant rating does not create a repository review

That is expected. Local rating state is intentionally local. Enable **Share with
maintainers** and use the explicit review action if the Q&A should leave the browser.

### A review exists but is not training eligible

An open review is pending. Eligibility changes only after an authorized provider merge
and successful lifecycle reconciliation.

### Generic counts stay at zero

The widget displays only reviewed/build-time aggregate data. A newly submitted event is
not optimistically added to the public count. Regenerate the complete V3 aggregate from
reviewed events and rebuild the docs.

## Security invariants

- no browser-side provider storage credentials;
- no Assistant anonymous rating telemetry route;
- no implicit conversion between local ratings, generic page feedback, Q&A review, and dataset contribution;
- generic `/v1/feedback` accepts one current contract only;
- current Q&A review consent is explicit and versioned;
- retries are idempotent and do not introduce participant identifiers;
- page-view load performs no feedback network request by default.
