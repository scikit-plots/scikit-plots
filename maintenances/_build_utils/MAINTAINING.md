# Maintaining `scikitplot._build_utils`

`scikitplot._build_utils` is the repository's build-time tooling plane. It generates source, rewrites vendored imports, extracts version metadata, detects compiler properties, and installs a custom Meson feature module. It is **not** a runtime service and must not become a dependency of installed `scikitplot` functionality.

Start with `_maintenance/FRESH_CHAT_HANDOFF.md`, then `STATE.json`, `FAMILY.md`, and `VERIFICATION.md`. Run the maintenance checker/tests before changing runtime build tools. Keep build-tool correctness, live toolchain integration, and release evidence separate: a Python unit test cannot prove a Meson/site-packages mutation is safe, and a successful template render cannot prove every validator or Cython compile lane works.

Current owned findings are recorded in `REVIEW.json`. Runtime code was not changed during onboarding.
