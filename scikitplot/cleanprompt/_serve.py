"""
Starting the web interface, and the container files to run it elsewhere.

Notes
-----
**User notes.**

.. code-block:: sh

    python -m scikitplot.cleanprompt flask                 # http://127.0.0.1:5000
    python -m scikitplot.cleanprompt flask --docker        # inside a container
    python -m scikitplot.cleanprompt docker --write ./ops  # write container files

**Developer notes — what ``--docker`` actually changes.**

One thing, and it is a security decision rather than a convenience: the bind
address. A process inside a container that binds ``127.0.0.1`` is reachable only
from inside that container, so the published port appears dead — which is the
single most common way a containerised web app looks broken. ``--docker``
therefore binds ``0.0.0.0``.

Binding all interfaces on a tool that holds personal data is a real exposure,
and the app has no authentication. So the widening is never silent: the address
is printed, the exposure is named, and a non-loopback bind outside container
mode additionally requires ``--allow-remote``. A container is the case where
``0.0.0.0`` is both necessary and bounded by the container's own network, which
is why ``--docker`` carries its own acknowledgement and ``--host 0.0.0.0`` does
not.

A secret key is required before the server starts. In container mode it must
come from the environment, because a per-process key means a restarted or
replicated container silently invalidates its own sessions.
"""

from __future__ import annotations

import argparse
import os
from typing import IO

from ._capabilities import require
from ._exceptions import CleanPromptError

__all__ = [
    "container_files",
    "resolve_bind",
    "serve",
]

#: Addresses that are reachable only from the local machine.
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


def resolve_bind(
    host: str | None,
    docker: bool,
    allow_remote: bool,
    debug: bool = False,
) -> tuple[str, str]:
    """
    Decide the bind address and describe its exposure.

    Parameters
    ----------
    host : str or None
        Explicit ``--host``, or ``None`` to choose one.
    docker : bool
        Whether container mode was requested.
    allow_remote : bool
        Whether the caller acknowledged a non-loopback bind.
    debug : bool, default=False
        Whether Flask's debug mode was requested. Allowed on loopback only.

    Returns
    -------
    address : str
        The address to bind.
    exposure : str
        A sentence describing who can reach it.

    Raises
    ------
    CleanPromptError
        If a non-loopback bind was requested without acknowledgement, or
        debug mode was requested on any address other than loopback.

    Notes
    -----
    **Developer notes — debug mode is refused off loopback (``CP-097``).**
    Flask's debug mode serves Werkzeug's interactive debugger, which runs
    Python typed into the browser. This app has no authentication and handles
    pasted personal data, so on a reachable address debug mode hands every
    visitor a shell. ``--allow-remote`` and ``--docker`` acknowledge that the
    *page* is reachable; neither is an acknowledgement that *code execution*
    is, and no flag here offers one. Container mode is included: ``0.0.0.0``
    inside a container is reachable from every network it is attached to.

    Examples
    --------
    >>> resolve_bind(None, False, False)[0]
    '127.0.0.1'
    >>> resolve_bind(None, True, False)[0]
    '0.0.0.0'
    >>> resolve_bind(None, False, False, debug=True)[0]
    '127.0.0.1'
    """
    if host is None:
        host = "0.0.0.0" if docker else "127.0.0.1"  # noqa: S104 - see module notes

    if host in LOOPBACK:
        return host, "Reachable only from this machine."

    if debug:
        raise CleanPromptError(
            f"refusing --debug on {host!r}: Flask's debug mode serves an "
            "interactive debugger that runs code typed into the browser, and "
            "this interface has no authentication. Use --debug only on "
            "127.0.0.1 (the default without --docker), or drop --debug."
        )

    if not (docker or allow_remote):
        raise CleanPromptError(
            f"refusing to bind {host!r}: that is reachable from other machines and "
            "this interface has no authentication. Pass --allow-remote to "
            "accept that, or --docker if you are running inside a "
            "container."
        )
    if docker:
        return host, (
            "Reachable from every network this container is attached to. "
            "Publish the port only to where you need it."
        )
    return host, (
        "Reachable from other machines on this network, without authentication."
    )


def serve(args: argparse.Namespace, stderr: IO[str]) -> int:
    """
    Start the Flask development server.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from the ``flask`` subcommand.
    stderr : file-like
        Where the banner and diagnostics go.

    Returns
    -------
    int
        Process exit status.

    Raises
    ------
    CapabilityError
        If the ``web`` tier is unavailable.
    CleanPromptError
        If the bind address is refused, or no secret key is configured.

    Notes
    -----
    **Developer notes.** The capability check runs before anything else, so an
    installation without Flask gets the install command rather than a traceback
    from an import three frames down.

    This is Flask's development server. It is right for a local single-user
    tool and wrong for a shared deployment, and the banner says so rather than
    leaving the reader to find out.
    """
    from ._cli import (  # ruff: ignore[import-outside-top-level]
        EXIT_OK,
        _policy_from,
        _registry_for,
    )

    require("web")  # actionable failure before flask is imported

    from ._app import create_app  # noqa: PLC0415 - deferred past the tier check
    from ._diagnostics import diagnose  # noqa: PLC0415

    host, exposure = resolve_bind(
        args.host, args.docker, args.allow_remote, debug=args.debug
    )
    policy, settings = _policy_from(args)

    if args.docker and not os.environ.get("CLEANPROMPT_SECRET_KEY"):
        raise CleanPromptError(
            "container mode needs CLEANPROMPT_SECRET_KEY in the environment. A "
            "per-process key would be regenerated on every restart and on "
            "every replica, silently invalidating sessions. Generate one with:"
            '\n  python -c "import secrets;print(secrets.token_hex(32))"'
        )

    ephemeral = not os.environ.get("CLEANPROMPT_SECRET_KEY")
    app = create_app(
        policy=policy,
        ephemeral_secret_key=ephemeral,
        enable_ner=settings["ner"],
        ner_model=settings["ner_model"],
        ner_engine=settings["ner_engine"],
        language=settings["language"],
        model_size=settings["model_size"],
        hide_terms=settings["hide"],
        word_boundary=settings["word_boundary"],
    )

    report = diagnose(policy, _registry_for(policy, settings))
    stderr.write("\nCleanPrompt web interface\n")
    stderr.write(
        "  http://{}:{}\n".format(
            (
                "127.0.0.1"  # ruff: ignore[hardcoded-bind-all-interfaces]
                if host == "0.0.0.0"  # ruff: ignore[hardcoded-bind-all-interfaces]
                else host
            ),
            args.port,  # noqa: S104
        )
    )
    stderr.write(f"  {exposure}\n")
    stderr.write(f"  {report.headline()}\n")
    for spot in report.blind_spots:
        if spot.severity == "high":
            stderr.write(f"  ! {spot.category}\n     fix: {spot.remedy}\n")
    if ephemeral:
        stderr.write(
            "  note: using a per-process session key; sessions end when this "
            "process does.\n"
        )
    stderr.write("  This is Flask's development server: local use only.\n\n")

    if args.open:
        _open_browser(host, args.port, stderr)

    app.run(host=host, port=args.port, debug=args.debug, use_reloader=False)
    return EXIT_OK


def _open_browser(host: str, port: int, stderr: IO[str]) -> None:
    """Open a browser, reporting rather than raising if it cannot."""
    import webbrowser  # noqa: PLC0415 - only needed on this path

    target = "http://{}:{}".format(
        (
            "127.0.0.1"  # ruff: ignore[hardcoded-bind-all-interfaces]
            if host == "0.0.0.0"  # ruff: ignore[hardcoded-bind-all-interfaces]
            else host
        ),
        port,  # noqa: S104
    )
    try:
        webbrowser.open(target)
    except Exception as exc:  # noqa: BLE001 - a browser is a convenience
        stderr.write(f"could not open a browser ({exc}); visit {target}\n")


def container_files(  # ruff: ignore[too-many-locals]
    port: int = 5000,
    with_ner: bool = False,
    language: str = "en",
    model_size: str = "sm",
) -> dict[str, str]:
    """
    Return the container files for the web interface.

    Parameters
    ----------
    port : int, default=5000
        Port to expose.
    with_ner : bool, default=False
        Install spaCy and a model in the image, and run with that model.
    language : str, default='en'
        Language whose model the image installs, when ``with_ner`` is set.
    model_size : str, default='sm'
        Preferred model size, resolved exactly as the runtime resolves it.

    Returns
    -------
    dict of str to str
        Filename to contents: a ``Dockerfile``, a ``docker-compose.yml``, a
        ``.dockerignore`` and a ``README.md``.

    Notes
    -----
    **Developer notes.** Three choices in the Dockerfile are deliberate.

    The image runs as a **non-root** user. A container that processes pasted
    personal data should not also hand an attacker root in its own filesystem.

    ``CLEANPROMPT_SECRET_KEY`` is **required**, not defaulted. A default would
    be a published secret, and every deployment would share it.

    The ``ner`` layer is **optional and separate**, because a model is tens to
    hundreds of megabytes and most users do not need it. Putting it behind a
    flag keeps the default image small rather than making everyone pay for a
    capability they may not use.

    **Developer notes — everything generated derives from the runtime.**
    Three defects came from writing a deployment file by hand next to the code
    it deploys (``CP-095``, ``CP-096``):

    - the image installed ``en_core_web_lg`` while the runtime default had
      moved to ``en_core_web_sm``, so an image built for entity detection
      paid for a large model and then failed looking for the small one. The
      model now comes from :func:`~scikitplot.cleanprompt._languages.resolve_model`,
      and the image passes it back explicitly (``--ner-engine spacy
      --ner-model <model>``), so a later change of default cannot separate
      what is installed from what is asked for;
    - the Dockerfile's ``docker run`` line published ``-p PORT:PORT``, which
      Docker binds on every host interface, while the compose file published
      on ``127.0.0.1`` only. Every launch line now uses ``127.0.0.1``;
    - the compose file set ``CLEANPROMPT_NER: "1"``, which nothing reads. A
      setting that looks load-bearing and is not is removed, not documented.
    """
    from ._languages import resolve_model  # ruff: ignore[import-outside-top-level]

    model = resolve_model(language, model_size)[0] if with_ner else None
    ner_layer = (
        (
            "\n# Optional named-entity tier: spaCy and the model the CMD below runs.\n"
            'RUN pip install --no-cache-dir "spacy>=3.4,<5" \\\n'
            f" && python -m spacy download {model}\n"
        )
        if with_ner
        else "\n"
    )
    cmd = (
        f'"--ner", "--ner-engine", "spacy", "--ner-model", "{model}"'
        if with_ner
        else ""
    )

    dockerfile = f"""\
# CleanPrompt web interface.
#
# Build:  docker build -t cleanprompt .
# Run:    docker run --rm -p 127.0.0.1:{port}:{port} \\
#           -e CLEANPROMPT_SECRET_KEY="$(python -c 'import secrets;print(secrets.token_hex(32))')" \\
#           cleanprompt
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# The base tier needs no third-party package; only the web tier does.
RUN pip install --no-cache-dir "flask>=2.2,<4" scikit-plots
{ner_layer}
# Run as a non-root user: this process handles pasted personal data.
RUN useradd --create-home --uid 10001 cleanprompt
USER cleanprompt

EXPOSE {port}

# CLEANPROMPT_SECRET_KEY is required and deliberately has no default: a default
# would be a published secret shared by every deployment.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \\
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:{port}/healthz')"

ENTRYPOINT ["python", "-m", "scikitplot.cleanprompt", "flask", "--docker", "--port", "{port}"]
CMD [{cmd}]
"""

    compose = f"""\
services:
  cleanprompt:
    build:
      context: .
      dockerfile: Dockerfile
    image: cleanprompt:local
    ports:
      # Published on loopback only. Change to "{port}:{port}" to expose it to
      # the network, and read the note in README.md before you do.
      - "127.0.0.1:{port}:{port}"
    environment:
      # Required. Generate with:
      #   python -c "import secrets;print(secrets.token_hex(32))"
      CLEANPROMPT_SECRET_KEY: "${{CLEANPROMPT_SECRET_KEY:?set CLEANPROMPT_SECRET_KEY}}"
    read_only: true
    tmpfs:
      - /tmp
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    restart: unless-stopped
"""

    dockerignore = """\
__pycache__/
*.py[cod]
.git/
.venv/
*.egg-info/
# Never ship a vault into an image: it holds the values that were removed.
*.vault.json
vault*.json
.cleanprompt.toml
"""

    readme = f"""\
# CleanPrompt in a container

```sh
export CLEANPROMPT_SECRET_KEY="$(python -c 'import secrets;print(secrets.token_hex(32))')"
docker compose up --build
```

Then open <http://127.0.0.1:{port}>.

Without compose, publish on loopback the same way:

```sh
docker build -t cleanprompt .
docker run --rm -p 127.0.0.1:{port}:{port} -e CLEANPROMPT_SECRET_KEY cleanprompt
```

`-p {port}:{port}` without the `127.0.0.1:` prefix publishes on **every**
interface of the host.

`--debug` is refused in a container: Flask's debugger runs code typed into the
browser, and it is allowed only on a loopback address.

## What the flags mean

`--docker` makes the app bind `0.0.0.0`. Inside a container that is necessary:
a process bound to `127.0.0.1` is reachable only from inside the container, so
the published port looks dead. It also means the app is reachable from every
network the container is attached to.

**This interface has no authentication.** The compose file therefore publishes
the port on `127.0.0.1` only. If you change that, put an authenticating proxy
in front of it first — anything pasted into this page is, by definition,
personal data.

`CLEANPROMPT_SECRET_KEY` is required and has no default. A default would be a
published secret shared by every deployment, and it signs the session cookie
that identifies which stored vault belongs to which browser.

## Name detection

The default image does not include spaCy, so names, organisations and places
are **not** detected. Rebuild with the `ner` layer to add it:

```sh
python -m scikitplot.cleanprompt docker --write . --with-ner
docker compose up --build
```

Without it, use the suggestion panel in the interface, which lists capitalised
candidates for you to hide explicitly.

## Checking what is active

```sh
docker compose exec cleanprompt \\
  python -m scikitplot.cleanprompt doctor --format json
```
"""

    return {
        "Dockerfile": dockerfile,
        "docker-compose.yml": compose,
        ".dockerignore": dockerignore,
        "README.md": readme,
    }
