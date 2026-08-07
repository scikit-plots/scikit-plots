# Verification — 0.2.4

The CI failure was caused by a test depending on `redirect_stdout` while the
GitHub Actions environment installed a different capture/logging stack. The CLI
JSON reached pytest capture but not the test-owned `StringIO`.

The fix provides an explicit output stream boundary:

```python
output = io.StringIO()
exit_code = main(["--docker", "--print-effective-config"], stdout=output)
payload = json.loads(output.getvalue())
```

Machine-readable JSON is written once to stdout. Operational warnings and logs
remain on stderr.

Verified in the sandbox:

- CLI tests: 22 passed.
- Complete dependency-independent suite, run 1: 94 passed, 2 skipped.
- Complete dependency-independent suite, run 2: 94 passed, 2 skipped.
- Source manifest check: passed.
- `compileall`: passed.
- Shell syntax checks: passed.
- `python -m scikitplot.mcp --docker --print-effective-config`: valid JSON.
- `python -m scikitplot.mcp --self-test`: valid JSON with clean stderr.

The skipped tests require the optional MCP SDK or an explicitly enabled live
HTTP server. Run the full SDK/live suites in the project Docker and CI images.
