# CI-stable CLI output routing

Version 0.2.4 separates machine-readable output from diagnostics.

- `--print-effective-config` and `--self-test` write one JSON document to stdout.
- Warnings and operational logs remain on stderr.
- `main(..., stdout=stream)` allows tests and embedding callers to inject a stream
  directly instead of relying on process-global `sys.stdout` redirection.
- JSON is no longer duplicated through warning logs.

This avoids capture-order differences between local pytest runs and CI runners that
install additional stdout/logging capture plugins.
