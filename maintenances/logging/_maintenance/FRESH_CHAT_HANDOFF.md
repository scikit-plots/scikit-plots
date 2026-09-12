# Fresh-chat handoff — `scikitplot.logging`

Current layout: public facade `scikitplot/logging/__init__.py`, implementation `scikitplot/logging/_logging.py`, focused tests under `scikitplot/logging/tests/`.

Status: maintenance PASS; runtime FAIL; release BLOCKED.

The move itself introduced two first-priority review items: public stdlib compatibility forwarding is no longer present on the package facade (LOG-PKG-001), and the focused test imports the pre-move shape and fails collection (LOG-TEST-001). A harness-only import correction produces 171 passing tests, but those tests also preserve known defective behavior (LOG-TEST-002).

Then revisit the pre-move runtime findings: error_log no-op, environment policy unwired, broken stream switching, CLI duplicate handlers, caller metadata, side-effectful private __getattr__, and formatter fallthrough.

Do not edit runtime just to make maintenance green. Do not classify harness-only test edits as native evidence.
