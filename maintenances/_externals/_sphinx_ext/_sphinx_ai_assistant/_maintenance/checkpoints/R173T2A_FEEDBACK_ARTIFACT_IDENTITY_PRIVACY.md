# R173T2A — Feedback artifact identity + privacy boundary

Status: COMPLETE

Base: Run 173 T1 (`73772505ff09f2e6f8434c72d11e3140bac6ac241f6343f0c053b4874fa3f599`).

## Purpose

Make local feedback downloads self-identifying by lifecycle role and tab format, and ensure the exact feedback-review request does not leak model transport configuration or unsafe source-page URLs.

## Artifact naming

- JSON request tab: `ai-feedback-review-request-json-<timestamp>.json`
- JSONL cloud projection tab: `ai-feedback-review-cloud-projection-jsonl-<timestamp>.jsonl`
- reserved human-export roles for the next slice: `cloud-record-jsonl` and `cloud-merged-jsonl`

Opaque provider storage names remain internal lifecycle keys and are not repurposed as human export names.

## Security boundary

Feedback-review network attribution is reduced to `id`, `provider`, and concrete `model`. Endpoint URLs, info URLs, labels, descriptions and other transport/configuration metadata do not cross the review boundary. The browser sanitizes page evidence to HTTP(S) origin + path and the Python server repeats both normalizations for legacy/hostile clients.

The JSON tab remains the exact request-envelope view. The JSONL tab is explicitly labeled as a pre-save cloud projection; `<server-assigned>` and `<receipt-id>` remain the only cloud-owned placeholders.

## Verification

- proxy app owner: 46/46
- feedback/privacy + feedback-review documentation: 17/17
- dataset schema owner: 26/26
- registered Node architecture: 140/140
- layout architecture: 9/9
- collection: 2332 / 0 errors
- maintenance checker: GREEN
