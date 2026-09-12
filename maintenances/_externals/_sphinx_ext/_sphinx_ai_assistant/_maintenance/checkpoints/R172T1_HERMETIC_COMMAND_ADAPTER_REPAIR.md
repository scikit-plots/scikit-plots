# R172T1 — hermetic command-adapter local repair

## Trigger

User local Python 3.11.14 / pytest 9.0.2 run stopped after 376 passes at
`test_run170_command_adapter_cannot_read_parent_secret`. The child returned 127
and pytest also emitted an unclosed stdout `ResourceWarning`.

## Classification

1. **Test environment/isolation mismatch:** the probe used
   `#!/usr/bin/env python3` while production deliberately passes only
   `PATH=os.defpath`. A micromamba-only Python is therefore not discoverable.
2. **Production resource-hygiene defect:** `publish_release.command_publisher()`
   did not close stdout on every error exit.

The security invariant itself held: the parent canary secret was not exposed.

## Fix

- Keep the production environment exactly secret-free and allowlisted:
  `PATH=os.defpath`, `LC_ALL=C`, `LANG=C`.
- Invoke Python test probes with `sys.executable`; do not depend on a PATH shebang.
- Reap killed publisher children and close stdout in a final ownership block.
- Add a canonical `test_publish_release.py` regression asserting failed children
  leave both stdin and stdout closed.
- Convert the other publisher command-protocol tests in the same module to
  explicit interpreter invocation so the same false failure does not recur.

## Verification

```text
exact failing node                          1/1 passed
Run170 process-hermeticity owner          10/10 passed
publish_release canonical owner           17/17 passed
combined, ResourceWarning promoted error  27/27 passed
```

## Next authority

The user's next `pytest -x -vv` run against the packaged R172T1 workspace.
