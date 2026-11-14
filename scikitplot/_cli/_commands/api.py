# scikitplot/_cli/_commands/api.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""`scikitplot api [MODULE] [--name NAME]` - the installed public API.

Lists the public names of a scikit-plots module (its ``__all__``): kind, call
signature and the first sentence of the docstring, read from the installed
package. With ``--name``, one name in full: its whole signature and docstring.
An assistant runs it to see the API of the version the project has, instead of
a list written down for another version.

``--format`` is ``text`` (default), ``markdown``, or a structured format
(``json``, ``yaml``, ``toml``) of the same data. Read-only; no network. Only
``scikitplot`` and modules below it are imported.
"""

from __future__ import annotations

from .. import exit_codes
from ..context import Context
from ..errors import CliError, UsageError
from ..output import emit


def run(
    ctx: Context,
    *,
    module: str = "scikitplot",
    name: str | None = None,
    fmt: str = "text",
) -> int:
    """List *module*'s public API, or one *name* of it in full."""
    # lazy: the library renderer is the source of truth (shared with the skills)
    from ..._api_listing import (  # ruff: ignore[import-outside-top-level]
        ApiListingError,
        module_listing,
        name_detail,
        render_markdown,
        render_text,
    )

    if isinstance(module, (list, tuple)):  # argparse nargs / click tuple safety
        module = module[0] if module else "scikitplot"
    try:
        data = name_detail(module, name) if name else module_listing(module)
    except ApiListingError as exc:
        if exc.reason == "name":
            raise UsageError(str(exc)) from exc
        cause = exc.__cause__
        missing = (
            getattr(cause, "name", None)
            if isinstance(cause, ModuleNotFoundError)
            else None
        )
        if missing and missing.split(".")[0] != "scikitplot":
            # An optional dependency of the module is not installed.
            raise CliError(
                str(exc),
                exit_code=exit_codes.UNAVAILABLE,
                hint=f"Install the missing dependency ({missing}), then run it again.",
            ) from exc
        hint = (
            "The installed scikit-plots may predate this module or name; check"
            " `python -m scikitplot show-versions` and the release notes."
        )
        raise CliError(str(exc), hint=hint) from exc
    if fmt in ("json", "yaml", "toml"):
        emit(ctx, data)
        return 0
    if name:
        lines = [
            f"{data['module']}.{data['name']} ({data['kind']}, scikit-plots {data['version']})",
            "",
            data["signature"],
        ]
        if data["doc"]:
            lines += ["", data["doc"]]
        text = "\n".join(lines) + "\n"
        if fmt == "markdown":
            text = (
                f"# `{data['module']}.{data['name']}` ({data['kind']}, scikit-plots"
                f" {data['version']})\n\n```python\n{data['signature']}\n```\n\n{data['doc']}\n"
            )
        ctx.stdout.write(text)
        return 0
    ctx.stdout.write(render_markdown(data) if fmt == "markdown" else render_text(data))
    return 0
