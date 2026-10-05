#!/usr/bin/env python
"""
Execute every cleanprompt gallery example and check what a gallery must hold.

Notes
-----
**User notes.** Run it from the repository root::

    python -B maintenances/cleanprompt/_maintenance/evidence/probe_gallery.py

It exits non-zero when an example fails, when one would write outside its own
temporary workspace, or when a reserved value would reach a published page.

**Developer notes — what this lane is for.**

Sphinx-Gallery executes these scripts during a documentation build, so they are
not prose: they are code that runs on somebody else's machine, in a process
that also builds the docs. Three things can go wrong that no unit test would
catch, and each has a check here.

*The example stops working.* A renamed option or a changed default turns a
tutorial into a failing build, and the failure surfaces at release time. Every
script is executed end to end and its status is asserted.

*The example writes somewhere real.* A vault holds removed values in clear
text, and the CLI's default location is the platform state directory. An
example that forgot to set ``CLEANPROMPT_VAULT`` would leave those values in
the documentation builder's home directory. Each script runs with ``HOME`` and
``XDG_STATE_HOME`` pointed at a sandbox, and the sandbox is checked afterwards:
anything written there is a script that did not confine itself.

*The example publishes a value that can reach somebody.* A gallery page is
published. Every address, telephone number and network address printed by these
examples has to come from a reserved range, and this probe greps the captured
output for the shapes that are not.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

#: Where the examples live, relative to the repository root.
GALLERY = Path("galleries/examples/cleanprompt")

#: Values that may appear in a published page, because none can reach anybody.
#: RFC 2606 reserves the domains, RFC 5737 and RFC 3849 the address ranges, and
#: +1 555 0100-0199 is the North American fiction block.
RESERVED_HOSTS = ("example.com", "example.org", "example.net", "example.invalid")
RESERVED_V4 = ("192.0.2.", "198.51.100.", "203.0.113.", "127.0.0.1", "0.0.0.0")
RESERVED_V6 = ("2001:db8:",)

#: Shapes that must not appear in captured output unless they are reserved.
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
IPV4 = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
PHONE = re.compile(r"\+1 555 \d{4}")

#: A telephone number is reserved only inside the fiction block.
FICTION_PHONE = re.compile(r"\+1 555 01[0-9]{2}\b")


def _scripts() -> "list[Path]":
    """Return the gallery scripts, in a stable order."""
    return sorted(GALLERY.glob("plot_*.py"))


def _sandboxed_environment(home: Path) -> "dict[str, str]":
    """Return an environment whose every state path is inside ``home``."""
    environment = dict(os.environ)
    environment["HOME"] = str(home)
    environment["USERPROFILE"] = str(home)
    environment["XDG_STATE_HOME"] = str(home / "state")
    environment["LOCALAPPDATA"] = str(home / "local")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path.cwd()), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    # A script that forgot CLEANPROMPT_VAULT must fall through to the state
    # directory, which is inside the sandbox, so the omission is visible as a
    # file rather than as a silent write to the real home.
    environment.pop("CLEANPROMPT_VAULT", None)
    environment.pop("CLEANPROMPT_VAULT_KEY", None)
    return environment


def _unreserved_values(text: str) -> "list[str]":
    """Return every value in ``text`` that is not from a reserved range."""
    offenders = []
    for address in EMAIL.findall(text):
        if not any(address.lower().endswith(host) for host in RESERVED_HOSTS):
            offenders.append("email: {0}".format(address))
    for address in IPV4.findall(text):
        if not any(address.startswith(prefix) for prefix in RESERVED_V4):
            offenders.append("ipv4: {0}".format(address))
    for number in PHONE.findall(text):
        if not FICTION_PHONE.fullmatch(number):
            offenders.append("phone: {0}".format(number))
    return offenders


def _leftovers(home: Path) -> "list[str]":
    """Return every path a script left inside the sandboxed home."""
    if not home.exists():
        return []
    return sorted(
        str(path.relative_to(home))
        for path in home.rglob("*")
        if path.is_file()
    )


def main() -> int:
    """Run every gallery script and report. Returns a process exit status."""
    if not GALLERY.is_dir():
        sys.stderr.write("no gallery at {0}; run from the repository root\n".format(GALLERY))
        return 2

    scripts = _scripts()
    if not scripts:
        sys.stderr.write("no plot_*.py scripts in {0}\n".format(GALLERY))
        return 2

    results = []
    failed = 0

    for script in scripts:
        with tempfile.TemporaryDirectory(prefix="cleanprompt-gallery-home-") as home_name:
            home = Path(home_name)
            completed = subprocess.run(
                [sys.executable, "-B", str(script)],
                capture_output=True,
                text=True,
                env=_sandboxed_environment(home),
                cwd=str(Path.cwd()),
            )
            captured = completed.stdout + completed.stderr
            leftovers = _leftovers(home)
            offenders = _unreserved_values(captured)

        problems = []
        if completed.returncode != 0:
            problems.append("exit {0}".format(completed.returncode))
        if leftovers:
            problems.append("wrote outside its workspace: {0}".format(leftovers[:5]))
        if offenders:
            problems.append("unreserved values: {0}".format(offenders[:5]))

        record = {
            "script": script.name,
            "status": completed.returncode,
            "stdout_lines": len(completed.stdout.splitlines()),
            "skips": captured.count("[SKIP]"),
            "leftover_files": len(leftovers),
            "unreserved_values": len(offenders),
            "ok": not problems,
        }
        results.append(record)

        if problems:
            failed += 1
            sys.stderr.write("FAIL {0}: {1}\n".format(script.name, "; ".join(problems)))
            tail = "\n".join(captured.splitlines()[-25:])
            sys.stderr.write("{0}\n".format(tail))

        print(
            "{0:<6} {1:<46} exit={2} lines={3:<4} skips={4} leftovers={5}".format(
                "ok" if record["ok"] else "FAIL",
                script.name,
                record["status"],
                record["stdout_lines"],
                record["skips"],
                record["leftover_files"],
            )
        )

    print()
    print(
        json.dumps(
            {
                "lane": "gallery_examples",
                "scripts": len(scripts),
                "failed": failed,
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
