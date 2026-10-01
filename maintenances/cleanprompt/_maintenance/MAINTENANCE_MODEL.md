# Maintenance model

Three statuses, never conflated:

- **maintenance** — this plane is internally consistent (checker green);
- **runtime** — the structural contract holds (checker's runtime findings empty);
- **release** — every verification lane in `VERIFICATION.md` is green.

A lane that was not run is `UNAVAILABLE`, never `PASS`. `--update` refuses to
refresh the evidence fingerprint while the runtime contract is failing.

Findings are closed only by a named regression test plus an executable probe.
Source inspection does not close a finding.
