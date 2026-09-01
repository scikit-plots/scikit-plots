# B64 — Feedback Review Training Eligibility & Quality Signal

Status: **GREEN**

## Scope

Complete Run 45 by turning explicitly shared, content-bearing one-Q&A feedback into a
merge-gated training source while preserving strict separation from anonymous rating
telemetry and whole-conversation dataset contribution.

## Authority model

Local rating remains zero-network by default. Anonymous `/v1/feedback` telemetry keeps
its own versioned permission and remains `trainingStatus=telemetry`.

The content-bearing feedback-review permission is now version **2.0.0** and visibly
means **review + model improvement**. It authorizes exactly one Q&A, rating, optional
note, and training use only if a maintainer merges the provider-native review. The
request also carries an independently versioned `trainingConsentFlag` /
`trainingConsentVersion` marker. Historical v1 review-only browser consent fails closed
and must be granted again.

## Eligibility lifecycle

```text
local rating
    -> explicit review/model-improvement consent
    -> provider PR/MR
    -> IN REVIEW / trainingEligible=false
    -> merge
    -> ELIGIBLE / trainingEligible=true
```

Close/decline never grants eligibility. Pending or post-merge participant withdrawal
removes the active review/canonical view according to the existing lifecycle.

The review ref carries the future canonical `trainingStatus=eligible` row, matching the
existing contribution review architecture, but the canonical branch remains the actual
eligibility boundary.

## Quality signal

Reviewed feedback retains the raw signed rating and its scale bounds, and the server
computes—not trusts from the browser—the normalized quality signal:

```text
qualityScore = (ratingValue - ratingScaleMin) / (ratingScaleMax - ratingScaleMin)
qualityPercent = qualityScore * 100
```

Both are clamped/bounded to `0..1` / `0..100`. Examples:

- quick `-1 / +1`: `0% / 100%`;
- five-level `[-2,-1,0,+1,+2]`: `0%, 25%, 50%, 75%, 100%`.

The original `ratingValue`, slug/title, mode, Q&A, optional note, model evidence,
review/training consent versions, and normalized quality fields remain in the canonical
row. Low-rated answers remain useful as negative/preference examples; the quality score
is a weight/signal, not an automatic deletion rule.

## Training builder

The default training gate now admits only:

- `_source=contribution` + `trainingStatus=eligible`; or
- `_source=feedback` + `trainingStatus=eligible`.

Privacy-minimal feedback telemetry remains excluded. Contribution keeps higher source
priority if a future canonical dedup key collides with feedback.

## UI

The Feedback tab and quick/detailed surfaces now say **Share feedback for review & model
improvement**. The current-Q&A summary displays the raw signed rating plus its normalized
quality percentage. Review status explicitly says that merge is required for training
eligibility.

## Version

The bundled proxy public API is ratcheted from **7.3.0** to **7.4.0** because
`/v1/feedback/review` now has a new consent contract, quality fields, and merge-gated
training semantics.

## Verification

- feedback review control-plane tests: GREEN;
- consent migration (v1 -> fail closed, v2 -> explicit): GREEN;
- quick + multi-level quality normalization: GREEN;
- training builder admits eligible feedback but excludes telemetry: GREEN;
- all registered browser/Node harnesses: GREEN;
- full runnable non-Sphinx suite: **864 passed, 3 skipped**;
- Python and JavaScript syntax: GREEN.
