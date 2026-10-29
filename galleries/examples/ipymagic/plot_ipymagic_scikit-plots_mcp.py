"""
Inspect and Self-test the MCP Server from a Notebook
====================================================

.. currentmodule:: scikitplot._ipymagic

``scikitplot mcp`` serves scikit-plots documentation to AI clients over the
Model Context Protocol. From a notebook, ``%scikitplot mcp ...`` is the
quickest way to look at what a server *would* do before you start one:

1. its effective configuration, as JSON;
2. the tools and resources it advertises;
3. a bounded self-test that searches the built-in corpus and exits;
4. a real server in a cell, stopped by the ``timeout`` setting (opt in).

``mcp`` is a *delegated* command: the magic runs it in a subprocess of the
kernel's Python, so a server that never exits cannot hang the kernel. The
interrupt button or ``timeout`` stops it and every process it started.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Where it runs
# -------------

import io
import json
import os
import socket

from scikitplot._ipymagic import MagicSettings, route, run_command

# In a notebook (this page's download), load the magics so the ``%%`` cells
# below run; a plain script has no IPython and skips this.
_shell = globals().get("get_ipython", lambda: None)()
if _shell is not None:
    _shell.run_line_magic("load_ext", "scikitplot")

print(route(["mcp", "--help"]))

# %%
# The effective configuration
# ---------------------------
#
# .. code-block:: ipython
#
#     r = %scikitplot mcp --print-effective-config
#     json.loads(r.stdout)["host"]
#
# The server prints JSON itself (it has no ``--format`` option), so the
# notebook parses ``result.stdout``. Nothing is started and no port is opened.

result = run_command("mcp --print-effective-config", out=io.StringIO())
config = json.loads(result.stdout)
print(result.summary())
for key in ("transport", "host", "port", "path", "max_concurrency"):
    print(f"{key:<16} {config[key]}")

# %%
# The default binds to ``127.0.0.1``: reachable from this machine only.

assert config["host"] == "127.0.0.1"

# %%
# What it advertises
# ------------------
#
# .. code-block:: ipython
#
#     %scikitplot mcp --list-capabilities

capabilities = json.loads(
    run_command("mcp --list-capabilities", out=io.StringIO()).stdout
)
print("server:", capabilities["server"]["name"], capabilities["server"]["version"])
print("tools:", [tool["name"] for tool in capabilities["tools"]])
print("resources:", [res["uri_template"] for res in capabilities["resources"]])

# %%
# A bounded self-test
# -------------------
#
# .. code-block:: ipython
#
#     %scikitplot mcp --self-test --self-test-query transport --self-test-require-match
#
# The self-test runs one search through the same code a client would reach,
# prints the result and exits: a CI-friendly check that needs no network, no
# port and no MCP client. ``--self-test-require-match`` makes an empty answer
# a failure (non-zero exit status).

test = run_command(
    "mcp --self-test --self-test-query transport --self-test-require-match",
    out=io.StringIO(),
)
answer = json.loads(test.stdout)
print(test.summary())
print("status:", answer["retrieval_status"], "| passages:", answer["count"])
print("notice:", answer["security"]["notice"][:72], "...")
assert test.ok and answer["count"] >= 1

# %%
# A real server, stopped by the timeout (opt in)
# ----------------------------------------------
#
# .. code-block:: ipython
#
#     %scikitplot_config timeout=5
#     %scikitplot mcp --transport streamable-http --port 8765
#
# A server runs until it is stopped. With ``timeout`` the cell ends after the
# given seconds with exit status 124 and ``result.timed_out``; the server and
# its children are stopped. It needs the MCP server dependencies
# (``pip install scikit-plots[mcp]``) and opens a port, so a documentation
# build runs it only when ``SCIKITPLOT_GALLERY_RUN_MCP_DOCKER=1``.

if os.environ.get("SCIKITPLOT_GALLERY_RUN_MCP_DOCKER", "0").strip() == "1":
    with socket.socket() as probe:  # a free loopback port
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = run_command(
        [
            "mcp",
            "--transport",
            "streamable-http",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        settings=MagicSettings(timeout=5),
        err=io.StringIO(),
    )
    print(server.summary())
    if not server.timed_out:  # it ended on its own: say why, do not pretend
        print("the server stopped before the timeout:", server.stderr.strip()[-300:])
else:
    print("SKIP: set SCIKITPLOT_GALLERY_RUN_MCP_DOCKER=1 to start a real server")

# %%
# .. tip::
#
#    In a notebook, the interrupt button stops a server started with
#    ``%scikitplot mcp`` the same way, and "Run all" stops there too.
#
# .. tags::
#
#    model-workflow: ipymagic
#    model-workflow: mcp
#    plot-type: text
#    level: intermediate
