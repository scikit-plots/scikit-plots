# Verification ladder

Use evidence in increasing strength; do not collapse these levels.

1. `compileall` / AST / inventory / plane checks.
2. Maintenance mutation suite.
3. Shipped focused tests.
4. Negative probes for error paths and filesystem state drift.
5. Repository-input generator probes (`annoylib.*.in`, `.src` fixtures).
6. Real Cython/compiler command execution in package-shaped directories.
7. Live Meson import of the copied `features` module on every supported Meson version.
8. Complete-root Git/version generation, including no-Git/tarball and dubious-ownership simulations without persistent user-config leakage.
9. Cross-platform filesystem/compiler lanes (Linux/macOS/Windows).

Current release remains BLOCKED. Meson integration is unavailable here, and six owned findings are open.
