# Deep semantic review — methodology (`scikitplot.corpus`)

The process that produced the resolved findings. It is deliberately mechanical
so it survives a context reset: any session can pick up a finding and apply the
same steps to the same standard. This is the "logic" half of the durable memory;
`MAINTAINING.md` is the "state" half.

## Engineering charter (applies to every change)

- **Zero-hallucination grounding.** Every claim, fix, and test is traceable to
  the *actual source in the uploaded tree*. Read the real code before asserting
  what it does. If the source doesn't support a statement, don't make it.
- **Minimal-impact, root-cause fixes only.** Fix the defect at its source; touch
  the fewest files/lines. No band-aids, no symptom-masking, no speculative
  refactors bundled in.
- **Always-green gate.** Nothing is "done" without a passing test. Rewrite a test
  to a corrected contract; never delete a test to make it pass.
- **Evidence per claim.** "Resolved" requires a named guard test and a paste-able
  run result, not "it works".
- **No `==` dependency pins.** Ranges only (`>=`, `~=`, documented upper bounds).
- **Python 3.8 → 3.15+.** See the compatibility rule in `MAINTAINING.md`.
- **Register discipline.** Findings are marked resolved with evidence, never
  deleted. Code + review guide + `MAINTAINING.md` change together.

## Per-finding workflow

1. **Ground.** Open the finding in the review guide. Read the *real* source at
   the cited files/symbols. Reproduce the defect's premise (e.g. show the stdlib
   parser expands a billion-laughs payload) so the fix has a baseline.
2. **Design the minimal fix.** Prefer a small shared primitive when the same
   defect recurs across sites (that is how `_atomic`, `_xml_safety`,
   `stream_copy_bounded` were born). Decide the invariant the fix must hold.
3. **Fix at root cause.** Edit the source. Keep unrelated code untouched. Remove
   now-dead imports. Keep it 3.8→3.15+ safe.
4. **Test twice.**
   - a **standalone harness** under `harness/` for the fast loop (load the module
     in isolation — see *Environment & harness patterns* in `MAINTAINING.md`);
   - a **permanent in-repo `pytest`** guard next to the code
     (`.../tests/test__*.py`), which is the durable artifact. Include a *baseline*
     assertion where possible (prove the unpatched behaviour was wrong) so the
     guard catches regressions.
5. **Reconcile the register.** In the review guide: flip the summary row status
   (`… → RESOLVED/PARTIAL`), add a `Status` row to the finding detail describing
   what was done + what (if anything) remains, and add a §29 positive-control row
   pointing at the new guard.
6. **Stage & present.** Copy the changed source + tests + guide to the outputs
   area; present the files with a concise, evidence-backed summary.
7. **Log it.** Append one line to `_maintenance/SESSION_LOG.md`.

## When to STOP rather than push

- The fix needs a **live environment** you don't have (network/TLS for the SSRF
  pinning transport, a real build for reproduction) → record what's done, scope
  the remainder honestly, mark PARTIAL. Do **not** ship untested security code.
- The finding requires a **design decision that is the maintainer's** (a new
  artifact envelope, a concurrency contract, a governance model) → do not invent
  the contract unilaterally. Record it as OPEN with a concrete recommendation and
  ask for direction.
- A fix would exceed minimal impact (wide refactor for an enhancement) → split it
  or defer; note the trade-off.

Honesty about the boundary is part of the standard: a withheld speculative change
is a small loss; a wrong "fix" to security code, or a design commitment made on
the maintainer's behalf, is a large one.

## Definition of done (per finding)

- [ ] Root cause fixed at source; unrelated code untouched.
- [ ] Standalone harness + permanent in-repo pytest both green.
- [ ] Baseline/regression assertion where feasible.
- [ ] 3.8→3.15+ clean; no `==` pins.
- [ ] Review guide summary row + detail `Status` + §29 control updated.
- [ ] `SESSION_LOG.md` appended.
