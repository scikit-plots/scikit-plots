# The `scikitplot-mcp` plugin

Opt-in documentation search for scikit-plots in Claude Code and local Cowork:
read-only, and every passage carries a validated source citation. It
registers one MCP server, `scikitplot-docs`, that runs
`python -m scikitplot.mcp` on your machine.

## What it runs and sends

It starts a local program: `python -m scikitplot.mcp`, in the first `python`
on your `PATH`, speaking MCP over stdio. The server only reads the
scikit-plots documentation it ships with and sends nothing to other
services. That is why it is a plugin of its own: the `scikitplot` plugin
stays Markdown only and runs nothing.

## Install

The `python` on your `PATH` needs scikit-plots with the MCP SDK:

```sh
pip install "scikit-plots[mcp]"
python -m scikitplot.mcp --help      # must print the usage, not ModuleNotFoundError
claude plugin marketplace add scikit-plots/scikit-plots --sparse .claude-plugin plugins
claude plugin install scikitplot-mcp@scikit-plots
```

Without the plugin, the same server can be registered directly:
`claude mcp add scikitplot-docs -- python -m scikitplot.mcp`. The
`scikitplot mcp` console command is equivalent to `python -m scikitplot.mcp`.

claude.ai chat ignores local servers; bundles for other clients (Cursor,
Windsurf, VS Code, Continue, Cline, Codex, OpenClaw) are in
`scikitplot/mcp/_plugins/` of the package. The full guide is
[`plugins/README.md`](../README.md).
