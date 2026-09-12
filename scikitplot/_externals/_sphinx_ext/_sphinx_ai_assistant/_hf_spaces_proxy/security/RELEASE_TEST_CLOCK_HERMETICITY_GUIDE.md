# Run 169 — hermetic release-security test clock

Run 169 removes the last wall-clock dependency from the synthetic cryptographic release
history used by Runs 149–168. It changes **test authority only**; production verifiers keep
their live-time freshness, expiry, rollback, and freeze-risk checks unchanged.

## Why this gate exists

Run 158 intentionally gives its synthetic attestation-status snapshot a bounded validity
window. Run 159 previously chose `NOW` from the machine wall clock. Once real time moved
inside Run 158's minimum-remaining-time safety margin, all downstream Runs 160–168 could
fail while constructing their predecessor fixture even though the repository bytes had
not changed.

That is not acceptable release evidence. A frozen test vector must remain replayable after
its calendar date passes.

## Authority boundary

Run 159 now derives its synthetic time from the fixed predecessor timeline:

```text
Run 158 fixed NOW
        ↓ + 1 hour
Run 159 synthetic NOW
        ↓
Runs 160–168 predecessor-derived clocks
```

This does **not** freeze production time. Runtime tools still use their explicit `now=`
argument when supplied and the real UTC clock when callers omit it. Expired production
status therefore continues to fail closed.

## Release gate

`tests/_architecture/test_release_gate_clock_hermeticity.py` statically rejects direct wall-clock
reads (`datetime.now`, `datetime.utcnow`, `date.today`, or `time.time`) in Runs 149–168
release-security tests. New release fixtures should derive time from a fixed predecessor
or an explicit deterministic constant.

A release MUST fail if a synthetic cryptographic fixture silently reacquires wall-clock
authority. Tests that intentionally exercise real-time behavior belong in a separately
identified live/integration gate, not in deterministic provenance replay.
