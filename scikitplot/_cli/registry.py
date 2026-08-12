# scikitplot/_cli/registry.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit built-in command registry (metadata only).

This module holds metadata and MUST NOT import any handler module. Both
frontends render from this registry, so top-level help never imports handlers
(FINDINGS CLI-FE-002).
"""

from __future__ import annotations

from ._spec import CommandSpec, Param

# Shared, reusable neutral option: the one output-format contract
# (FINDINGS CLI-FE-003).
FORMAT = Param(
    dest="fmt",
    flags=("--format",),
    kind="option",
    choices=("text", "json", "yaml", "toml"),
    default="text",
    help="Output format (text, json, yaml, toml).",
)

# Native library "mode" passthrough, per command. `--mode` drives the library
# call; an explicit structured `--format` takes precedence and emits the data.
# Defaults (mode="stdout", format="text") preserve human output.
MODE_VERSIONS = Param(
    dest="mode",
    flags=("--mode",),
    kind="option",
    choices=("stdout", "dict", "yaml", "rich"),
    default="stdout",
    help="Library render mode: stdout, dict, yaml, rich (default: stdout).",
)
MODE_CONFIG = Param(
    dest="mode",
    flags=("--mode",),
    kind="option",
    choices=("stdout", "dicts"),
    default="stdout",
    help="Library render mode: stdout, dicts (default: stdout).",
)

BUILTIN_COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        name="info",
        summary="Show runtime and installation information.",
        handler="scikitplot._cli._commands.info:run",
        params=(FORMAT,),
    ),
    CommandSpec(
        name="doctor",
        summary="Diagnose the current scikit-plots environment.",
        handler="scikitplot._cli._commands.doctor:run",
        params=(
            Param(
                dest="mask_envs",
                flags=("--mask-envs",),
                kind="flag",
                default=False,
                help="Mask sensitive environment variable values.",
            ),
            FORMAT,
        ),
    ),
    CommandSpec(
        name="show-versions",
        summary="Show scikit-plots version and dependency information.",
        handler="scikitplot._cli._commands.show_versions:run",
        aliases=("show_versions",),  # back-compat with the old underscore name
        params=(MODE_VERSIONS, FORMAT),
    ),
    CommandSpec(
        name="show-config",
        summary="Show scikit-plots build and runtime configuration.",
        handler="scikitplot._cli._commands.show_config:run",
        aliases=("show_config",),
        params=(MODE_CONFIG, FORMAT),
    ),
    CommandSpec(
        name="sysinfo",
        summary="Show operating-system and interpreter information.",
        handler="scikitplot._cli._commands.sysinfo:run",
        params=(FORMAT,),
    ),
    CommandSpec(
        name="greet",
        summary="Greet someone by name (example command).",
        handler="scikitplot._cli._commands.greet:run",
        params=(
            Param(dest="name", kind="argument", default="World", help="Name to greet."),
            Param(
                dest="emoji",
                flags=("--emoji",),
                kind="flag",
                negatable=True,
                default=True,
                help="Add an emoji to the greeting.",
            ),
        ),
    ),
    # Delegated (pass-through) command: forwards all trailing arguments to the
    # mcp submodule's own entry point. Imported lazily; see EXTENDING.md.
    CommandSpec(
        name="mcp",
        summary="Run or probe the scikit-plots documentation MCP server.",
        delegate="scikitplot.mcp.__main__:main",
        capabilities=("mcp",),
        install_hint="Install the MCP extra: pip install scikit-plots[mcp]",
    ),
)


def _build_indexes() -> tuple[dict[str, CommandSpec], dict[str, str]]:
    by_name: dict[str, CommandSpec] = {}
    by_alias: dict[str, str] = {}
    for spec in BUILTIN_COMMANDS:
        if spec.name in by_name:
            raise ValueError(f"duplicate command name {spec.name!r}")
        by_name[spec.name] = spec
    for spec in BUILTIN_COMMANDS:
        for alias in spec.aliases:
            if alias in by_name:
                raise ValueError(f"alias {alias!r} collides with a command name")
            if alias in by_alias:
                raise ValueError(f"duplicate alias {alias!r}")
            by_alias[alias] = spec.name
    return by_name, by_alias


BY_NAME, BY_ALIAS = _build_indexes()


def resolve(name: str) -> CommandSpec | None:
    """Return the :class:`CommandSpec` for a command name or alias, else None."""
    if name in BY_NAME:
        return BY_NAME[name]
    if name in BY_ALIAS:
        return BY_NAME[BY_ALIAS[name]]
    return None


__all__ = ["BUILTIN_COMMANDS", "BY_ALIAS", "BY_NAME", "FORMAT", "resolve"]
