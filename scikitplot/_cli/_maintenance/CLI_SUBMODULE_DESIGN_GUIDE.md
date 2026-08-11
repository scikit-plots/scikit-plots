# CLI Submodule Design Guide

> **Project:** scikit-plots
> **Target submodule:** `scikitplot._cli`
> **Status:** Architecture and rebuild guideline
> **Approach:** Restructure from scratch while preserving useful public behavior
> **Primary design rule:** **Discover early, import late, initialize only when required.**

---

## 1. Purpose

This document defines the target architecture, behavior, public interface, migration strategy, reliability rules, and implementation standards for rebuilding the `scikitplot._cli` submodule.

The goal is not to make a thin wrapper around Python functions.

The goal is to build a small, predictable CLI runtime that can safely expose library workflows while remaining:

- fast to start;
- lazy by default;
- dependency-light at bootstrap;
- easy to extend;
- safe for shell and CI use;
- stable for scripts and automation;
- usable by humans;
- machine-readable when requested;
- robust when optional capabilities are unavailable;
- testable without importing the whole project;
- compatible with native and restricted runtimes where practical;
- ready for future first-party commands and third-party plugins.

The CLI should evolve independently from the internal package layout.

A Python module existing in the library does **not** automatically mean it should be a CLI command.

---

## 2. Scope

This guide covers:

1. public CLI structure;
2. command naming and taxonomy;
3. bootstrap architecture;
4. command registry design;
5. strict lazy command loading;
6. optional dependency loading;
7. runtime resource activation;
8. runtime context;
9. capability detection;
10. output and rendering;
11. logging;
12. errors and exit codes;
13. configuration;
14. aliases and deprecation;
15. plugin discovery and loading;
16. security boundaries;
17. signals and cleanup;
18. shell completion;
19. native/WASM/runtime differences;
20. testing;
21. performance regression protection;
22. migration from the current implementation;
23. phased implementation plan;
24. acceptance criteria.

This document does **not** require every future command to be implemented during the initial rebuild.

---

# Part I — Architectural Direction

## 3. Current baseline

The current CLI slice is organized approximately as:

```text
scikitplot/_cli/
├── __init__.py
├── cli.py
├── _misc.py
│
├── _commands/
│   ├── __init__.py
│   ├── greet.py
│   ├── show_config.py
│   ├── show_versions.py
│   └── sysinfo.py
│
└── _cmd_options/
    ├── __init__.py
    ├── _cmd_options_click.py
    └── _cmd_options_optparse.py
```

The installed console entry point currently targets:

```toml
[project.scripts]
scikitplot = "scikitplot._cli.cli:cli"
```

The current implementation contains several ideas worth preserving:

- Click-based command handling;
- modular command files;
- reusable options;
- diagnostic commands;
- version/configuration/system information;
- JSON output experiments;
- Rich-based human output;
- aliases;
- dynamic command discovery;
- a `doctor` concept.

However, the current dependency and execution graph should not be preserved as-is.

---

## 4. Current architectural gaps

### 4.1 Dynamic discovery is currently eager

The current `_load_commands()` scans `_commands` and imports discovered modules immediately.

Conceptually:

```text
import scikitplot._cli.cli
        │
        ▼
pkgutil.iter_modules(...)
        │
        ├── import greet
        ├── import show_config
        ├── import show_versions
        └── import sysinfo
```

This is dynamic registration, but it is **not lazy command activation**.

The target architecture must separate:

```text
command metadata
```

from:

```text
command implementation import
```

---

### 4.2 Optional presentation dependencies leak into startup

Some command modules import Rich at module import time.

When all command modules are imported during CLI startup, optional presentation dependencies can become effective bootstrap dependencies.

The target CLI must not require a feature renderer merely to process:

```bash
scikitplot --help
scikitplot --version
```

---

### 4.3 Public entry point and dependency policy are not fully aligned

The public console script enters a module that imports Click directly.

Therefore the rebuild must make an explicit dependency decision.

Recommended rule:

> **Click is a small required dependency of the CLI runtime. Heavy library, renderer, integration, plugin, native, network, and scientific dependencies are not bootstrap dependencies.**

Do not attempt to make every dependency optional if doing so makes the bootstrap fragile.

Optimize the expensive and failure-prone boundaries first.

---

### 4.4 Broad exception suppression hides real failures

A command resolver must distinguish:

- command not found;
- alias not found;
- known command failed to import;
- optional dependency missing;
- plugin broken;
- unsupported platform;
- internal programming error.

Do not use broad patterns equivalent to:

```python
try:
    ...
except Exception:
    pass
```

inside command resolution.

---

### 4.5 Logging must not contaminate command output

Structured output must remain structurally valid.

For example:

```bash
scikitplot info --format json | jq .
```

must not receive logging text on stdout.

The CLI must define strict stdout/stderr ownership.

---

### 4.6 Output flags should not conflict

Independent flags such as:

```text
--json
--yaml
--rich
```

allow ambiguous combinations.

The target CLI should normalize structured output under one explicit contract:

```text
--format text|json|yaml
```

Rich styling should normally be a property of human text rendering, not a competing data format.

---

### 4.7 CLI spelling should use shell conventions

Prefer:

```text
show-config
file-path
dark-theme
lib-sample
```

over:

```text
show_config
file_path
dark_theme
lib_sample
```

Python variable names remain snake_case internally.

---

### 4.8 Legacy helper code should be justified, not automatically migrated

The rebuild should classify inherited helpers as:

```text
KEEP
MIGRATE
DEPRECATE
DELETE
```

Do not migrate large compatibility or pip-derived utility modules solely because they already exist.

Every retained helper must have a clear CLI responsibility.

---

# Part II — Design Principles

## 5. Normative language

This guide uses:

- **MUST** — required for correctness or architectural consistency;
- **MUST NOT** — prohibited;
- **SHOULD** — strongly recommended;
- **SHOULD NOT** — normally avoid;
- **MAY** — optional.

---

## 6. Core principles

### 6.1 Discover early, import late

The CLI MAY know that a command exists without importing its implementation.

```text
KNOWN
   ≠
LOADED
```

---

### 6.2 Initialize only when required

Importing a command MUST NOT automatically initialize expensive resources.

Examples:

- thread/process pools;
- model objects;
- network clients;
- MLflow clients;
- GPU contexts;
- memory maps;
- ANN indexes;
- databases;
- remote sessions.

---

### 6.3 CLI code is an adapter

The CLI should convert:

```text
shell input
    ↓
validated request
    ↓
public/internal service API
    ↓
result
    ↓
renderer
```

Business or scientific logic SHOULD NOT live primarily in Click callbacks.

---

### 6.4 Imports should be side-effect light

Importing CLI modules MUST NOT unexpectedly:

- write files;
- create cache directories;
- connect to networks;
- launch servers;
- mutate global environment variables;
- start threads;
- start processes;
- load large datasets;
- initialize GPU/runtime engines;
- import every plugin.

---

### 6.5 Human and machine interfaces are separate contracts

Human output may evolve visually.

Machine output must be conservative and stable.

---

### 6.6 Explicit beats magical

First-party commands SHOULD use an explicit registry.

Do not make a new `.py` file become a public command merely because it exists inside a scanned directory.

---

### 6.7 Safe failure is part of the API

Optional functionality must fail with an actionable CLI error, not an uncontrolled import traceback.

---

### 6.8 Public CLI structure follows workflows, not package internals

Avoid mirroring every Python submodule.

Prefer:

```text
scikitplot config show
scikitplot doctor
scikitplot info
```

over exposing internal module names.

---

# Part III — Target Runtime Architecture

## 7. Big picture

```text
                       USER
                        │
                        ▼
                `scikitplot ...`
                        │
                        ▼
              ┌──────────────────┐
              │  CLI BOOTSTRAP   │
              │ small + stable   │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ COMMAND REGISTRY │
              │ metadata only    │
              └────────┬─────────┘
                       │
              resolve command name
                       │
                       ▼
              ┌──────────────────┐
              │ METADATA COMMAND │
              │ proxy / facade   │
              └────────┬─────────┘
                       │
                invocation only
                       │
                       ▼
              ┌──────────────────┐
              │ COMMAND LOADER   │
              │ import target    │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ CAPABILITY CHECK │
              │ optional/runtime │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ RUNTIME CONTEXT  │
              │ config/log/io    │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ COMMAND ADAPTER  │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ LIBRARY/SERVICE  │
              │      API         │
              └────────┬─────────┘
                       │
                       ▼
                    RESULT
                       │
                       ▼
                   RENDERER
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
           stdout              stderr
```

---

## 8. Three levels of laziness

The architecture defines three independent lazy boundaries.

### L1 — command lazy loading

Do not import a command implementation until the command actually needs execution.

```text
scikitplot --help
        │
        └── MUST NOT import feature implementations
```

---

### L2 — dependency lazy loading

A loaded command module should still avoid importing an optional heavy dependency until the code path needs it.

Example:

```text
plot command adapter
       │
       ├── parse options
       ├── validate paths
       │
       └── import matplotlib only before plot execution
```

---

### L3 — resource lazy initialization

An imported dependency does not imply that its expensive runtime objects should immediately be created.

```text
command imported
      │
      └── resource remains dormant

resource requested
      │
      ▼
initialize resource
```

---

## 9. Strict laziness and Click help

A normal Click lazy `Group` based only on `list_commands()` and `get_command()` can still load immediate child commands while Click renders their short help.

Therefore strict startup laziness requires more than a basic lazy `Group`.

### Required design

The root registry must contain enough metadata to render:

- command name;
- short help;
- hidden status;
- deprecation status;
- category/order if needed;
- aliases if shown.

Top-level help should use lightweight metadata or proxy command objects.

The real command implementation should load only when deeper command-specific behavior is required.

Conceptually:

```text
CommandSpec
   │
   ├── name
   ├── short_help
   ├── target
   ├── aliases
   ├── hidden
   └── capabilities
          │
          ▼
MetadataCommand
          │
   invoke requested?
      │         │
     no        yes
      │         │
 render help    ▼
             load real command
```

This is stricter than simply deferring `importlib.import_module()` in `Group.get_command()`.

---

# Part IV — Package Layout

## 10. Recommended package structure

Start small and add modules only when a real responsibility exists.

```text
scikitplot/_cli/
├── __init__.py
├── __main__.py
├── app.py
│
├── registry.py
├── loader.py
├── context.py
├── capabilities.py
├── errors.py
├── exit_codes.py
├── output.py
├── logging.py
├── options.py
│
├── commands/
│   ├── __init__.py
│   ├── info.py
│   ├── doctor.py
│   ├── config.py
│   └── ...
│
└── plugins/
    ├── __init__.py
    ├── discovery.py
    ├── registry.py
    └── policy.py
```

Possible future modules, only if needed:

```text
completion.py
deprecations.py
platform.py
telemetry.py
profiles.py
resources.py
```

Do not create abstractions before they have behavior.

---

## 11. Responsibilities by module

### `app.py`

Owns:

- root Click application;
- root options;
- root context initialization;
- root help behavior;
- connection to the command registry.

Must remain dependency-light.

---

### `registry.py`

Owns:

- `CommandSpec`;
- built-in command metadata;
- aliases;
- command collision rules;
- deterministic command ordering;
- metadata queries.

MUST NOT import command implementations.

---

### `loader.py`

Owns:

- import target parsing;
- command implementation loading;
- loaded-object validation;
- load error normalization;
- optional load cache.

---

### `context.py`

Owns invocation state such as:

- debug mode;
- verbosity;
- color policy;
- output format;
- configuration view;
- environment abstraction;
- runtime feature switches;
- output streams;
- invocation ID if needed.

---

### `capabilities.py`

Owns:

- optional dependency checks;
- platform checks;
- runtime checks;
- capability descriptions;
- install hints;
- unsupported-capability diagnostics.

Capability metadata should be queryable without importing the heavy implementation whenever practical.

---

### `errors.py`

Owns the CLI error taxonomy.

---

### `exit_codes.py`

Owns stable semantic exit codes.

---

### `output.py`

Owns:

- render mode selection;
- JSON serialization;
- YAML serialization if supported;
- text rendering;
- optional Rich integration;
- safe fallback to plain text;
- machine-output guarantees.

---

### `logging.py`

Owns:

- CLI logging policy;
- handler setup;
- stderr routing;
- verbosity mapping;
- debug formatting;
- log-file setup if supported.

---

### `options.py`

Owns reusable Click option factories/decorators that genuinely apply across commands.

Avoid a giant option catalog whose options are unrelated to most commands.

---

### `commands/`

Contains thin adapters.

A command module SHOULD:

1. define command-local arguments/options;
2. validate CLI-specific input;
3. call a library/service layer;
4. return/format results through common infrastructure.

---

# Part V — Entry Points and Public Invocation

## 12. Console entry point

Recommended future entry point:

```toml
[project.scripts]
scikitplot = "scikitplot._cli.app:cli"
```

Optionally preserve the old import path temporarily with a compatibility shim if external code imports it.

Example:

```python
# scikitplot/_cli/cli.py
from .app import cli

__all__ = ["cli"]
```

Do not duplicate application construction across both files.

---

## 13. `python -m` support

Provide:

```bash
python -m scikitplot._cli
```

through:

```python
# __main__.py
from .app import cli

if __name__ == "__main__":
    # or cli()
    raise SystemExit(cli())
```

Whether `python -m scikitplot` should also invoke the CLI is a separate public API decision.

Do not add that behavior accidentally.

---

# Part VI — Command Registry

## 14. `CommandSpec`

The registry is the architectural center of command discovery.

Recommended initial model:

```python
from dataclasses import dataclass
from typing import Final

@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    target: str
    short_help: str
    aliases: tuple[str, ...] = ()
    hidden: bool = False
    deprecated: bool = False
    capabilities: tuple[str, ...] = ()
```

Possible future fields:

```text
category
order
platforms
experimental
replacement
deprecated_since
remove_after
plugin_distribution
plugin_version
network_required
interactive
```

Do not add fields until the runtime uses them.

---

## 15. Built-in registry

Prefer explicit built-in registration:

```python
BUILTIN_COMMANDS = (
    CommandSpec(
        name="info",
        target="scikitplot._cli.commands.info:cli",
        short_help="Show runtime and installation information.",
    ),
    CommandSpec(
        name="doctor",
        target="scikitplot._cli.commands.doctor:cli",
        short_help="Diagnose the current scikit-plots environment.",
    ),
)
```

Benefits:

- deterministic;
- auditable;
- no filesystem scanning;
- stable ordering;
- predictable documentation;
- explicit public surface;
- easier security review;
- easier deprecation;
- easier capability metadata;
- no accidental command exposure.

---

## 16. Do not use file scanning as the first-party API contract

Avoid making this the primary built-in mechanism:

```python
pkgutil.iter_modules(commands.__path__)
```

File scanning can be useful in controlled tooling, tests, or some plugin systems.

It should not decide which first-party modules become public commands.

---

## 17. Target notation

Use one import target notation consistently:

```text
package.module:attribute
```

Example:

```text
scikitplot._cli.commands.info:cli
```

Do not mix:

```text
package.module.attribute
package.module:attribute
```

unless the loader explicitly supports both for compatibility.

---

# Part VII — Command Loading

## 18. Loader contract

The loader receives a `CommandSpec`, imports its target, resolves the attribute, validates it, and returns a Click command.

Conceptual implementation:

```python
def load_command(spec: CommandSpec) -> click.Command:
    module_name, attr_name = spec.target.split(":", 1)

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise CommandDependencyError.from_import(spec, exc) from exc
    except Exception as exc:
        raise CommandLoadError(spec.name) from exc

    try:
        command = getattr(module, attr_name)
    except AttributeError as exc:
        raise CommandContractError(spec.name, spec.target) from exc

    if not isinstance(command, click.Command):
        raise CommandContractError(spec.name, spec.target)

    return command
```

The final implementation should distinguish missing target modules from missing optional dependencies inside a command.

Do not incorrectly report every `ModuleNotFoundError` as “command module missing.”

---

## 19. Loaded command cache

Command objects MAY be cached within the current process after a successful load.

Requirements:

- cache only validated commands;
- failed imports should not poison unrelated commands;
- tests must be able to reset the cache;
- normal startup must not pre-fill the cache.

---

## 20. Command state machine

A command can be understood as:

```text
REGISTERED
    │
    ▼
RESOLVED
    │
    ▼
CAPABILITY-CHECKED
    │
    ▼
LOADED
    │
    ▼
VALIDATED
    │
    ▼
RUNNING
    │
    ├── success ──► FINISHED
    │
    ├── user error ─► FAILED-CLEANLY
    │
    └── internal error ─► FAILED-INTERNAL
```

Runtime resources have their own lifecycle and should not be conflated with command import state.

---

# Part VIII — Command Taxonomy and UX

## 21. Design commands around user tasks

Good:

```text
scikitplot info
scikitplot doctor
scikitplot config show
scikitplot config get
scikitplot plugins list
```

Avoid blindly producing:

```text
scikitplot module_function_name
```

for every internal function.

---

## 22. Recommended initial public tree

The exact final command tree may evolve, but a clean initial target is:

```text
scikitplot
├── info
├── doctor
├── config
│   ├── show
│   ├── get
│   └── path
│
├── plugins
│   ├── list
│   ├── info
│   └── doctor
│
└── <feature workflows added intentionally>
```

Possible future feature groups:

```text
datasets
metrics
plot
annoy
mlflow
serve
```

Only add them when there is a real stable workflow.

---

## 23. Current command migration candidates

Suggested mapping:

```text
CURRENT              TARGET

show_versions      ┐
sysinfo            ├──► info
show_config        ┘      or config show

doctor             ───► doctor

greet              ───► example/test command,
                         hidden command,
                         or remove from production CLI

st / st2           ───► serve streamlit
                         only if still supported

gr                 ───► serve gradio
                         only if still supported
```

Do not commit to UI/server commands until their package ownership and optional dependency story are clear.

---

## 24. Naming conventions

### Commands

Use lowercase kebab-case:

```text
show-config
show-versions
system-info
```

Prefer short workflow names where clear:

```text
info
doctor
config
plugins
```

---

### Options

Use long kebab-case:

```text
--file-path
--no-color
--cache-dir
--log-level
```

Internal Python parameter:

```python
file_path
```

---

### Short options

Short options are scarce.

Reserve common meanings:

```text
-h    help
-V    version
-v    verbose
-q    quiet
```

Do not assign obscure short flags merely because a long option exists.

---

### Boolean options

For configurable booleans, prefer:

```text
--color / --no-color
--cache / --no-cache
```

when users genuinely need both states.

Avoid:

```python
@click.option("--share", is_flag=True, default=True)
```

because the flag cannot express `False`.

---

## 25. Aliases

Aliases are useful for migration and convenience, but must be explicit.

Example:

```text
show_versions → info
```

Rules:

- aliases MUST be represented in metadata;
- aliases MUST NOT silently shadow real commands;
- aliases SHOULD be visible in diagnostics;
- deprecated aliases SHOULD warn on stderr;
- aliases MUST resolve before loading the implementation.

---

# Part IX — Runtime Context

## 26. One invocation context

Create a typed application context rather than passing loosely structured dictionaries everywhere.

Example:

```python
@dataclass(slots=True)
class CLIContext:
    debug: bool = False
    verbose: int = 0
    quiet: int = 0
    output_format: str = "text"
    color: str = "auto"
    config_path: Path | None = None
```

Possible future fields:

```text
non_interactive
offline
profile
trace
runtime
plugin_policy
```

---

## 27. Context creation must remain cheap

Root context creation MUST NOT:

- load NumPy;
- load SciPy;
- load Matplotlib;
- initialize MLflow;
- scan large directories;
- contact package indexes;
- import plugins;
- open databases.

Configuration parsing itself should be lazy or minimal where practical.

---

## 28. Do not store arbitrary unrelated objects in `ctx.obj`

Use a typed context object or a small set of explicitly scoped context objects.

This improves:

- type checking;
- testability;
- command composition;
- plugin isolation;
- readability.

---

# Part X — Capabilities

## 29. Capability model

A command may exist even when its runtime capability is unavailable.

Example:

```text
annoy command is known
        │
        ▼
environment lacks native extension
        │
        ▼
command remains discoverable
        │
        ▼
invocation explains why unavailable
```

This is better UX than hiding the command entirely.

---

## 30. Capability examples

```text
plotting
annoy-native
mlflow
streamlit
gradio
network
yaml
rich
gpu
wasm
filesystem-write
```

Do not turn every package into a capability.

Capabilities represent user-visible runtime abilities.

---

## 31. Capability state

Recommended model:

```text
AVAILABLE
UNAVAILABLE
UNSUPPORTED
DEGRADED
UNKNOWN
```

Examples:

```text
AVAILABLE
  dependency installed and runtime supported

UNAVAILABLE
  optional dependency missing

UNSUPPORTED
  environment cannot support the capability

DEGRADED
  functionality available with reduced behavior

UNKNOWN
  check intentionally not performed
```

---

## 32. Capability diagnostics

Bad:

```text
ModuleNotFoundError: No module named ...
```

Preferred:

```text
Error: command 'annoy' requires the 'annoy-native' capability.

Status:
  unavailable

Reason:
  native extension is not installed for this environment

Next:
  scikitplot doctor
```

Installation hints MAY be included when they are accurate for the current distribution policy.

Do not guess installation commands for unsupported environments.

---

# Part XI — Output Contract

## 33. stdout

stdout is for requested command results.

Examples:

```text
JSON
YAML
plain text result
generated path
table intended as command output
```

---

## 34. stderr

stderr is for diagnostics:

```text
warnings
progress
logging
deprecation messages
retry notices
debug details
errors
```

---

## 35. Structured output

Recommended common option:

```text
--format text|json|yaml
```

Initial implementation MAY support only:

```text
text
json
```

and add YAML later.

Rules for JSON:

1. stdout MUST contain valid JSON.
2. No logging text may precede or follow it.
3. Tracebacks go to stderr.
4. Warnings go to stderr.
5. Serialization failures are CLI errors.
6. Keys should remain stable when documented as public machine output.

---

## 36. Human renderer

Human text output can support:

- colors;
- tables;
- panels;
- headings;
- Unicode symbols.

But the core command must still work if optional rich rendering is unavailable, unless Rich is deliberately made a required CLI dependency.

Recommended policy:

```text
text + interactive terminal + Rich available
    → rich renderer

text + pipe
    → plain renderer

--color never
    → plain/no ANSI

json
    → structured renderer, never Rich
```

---

## 37. Color policy

Prefer:

```text
--color auto|always|never
```

over only:

```text
--no-color
```

`auto` should consider terminal capability.

Honor widely used environment behavior where appropriate, but document exact precedence before implementing it.

---

# Part XII — Logging

## 38. Logging must be configured once

Do not let every command call its own independent `set_log_level()`.

Recommended lifecycle:

```text
parse root options
      │
      ▼
create CLIContext
      │
      ▼
configure CLI logging once
      │
      ▼
execute command
```

---

## 39. Verbosity semantics

Recommended simple model:

```text
default       WARNING
-v            INFO
-vv           DEBUG

-q            ERROR
-qq           CRITICAL

--debug       DEBUG + developer diagnostics
```

Define conflict behavior.

Recommended:

```text
--debug overrides verbosity
quiet overrides verbose
```

or reject conflicting combinations.

Whichever policy is selected must be tested.

---

## 40. Logging destination

Default CLI logs should go to stderr.

If file logging exists:

```text
--log-file PATH
```

it should be explicit.

Do not create log files just because the CLI module was imported.

---

# Part XIII — Errors

## 41. Error taxonomy

Define a small hierarchy.

Example:

```python
class CLIError(Exception):
    exit_code = 1

class CLIUsageError(CLIError):
    ...

class CLIConfigError(CLIError):
    ...

class CapabilityUnavailable(CLIError):
    ...

class UnsupportedRuntime(CLIError):
    ...

class CommandLoadError(CLIError):
    ...

class CommandContractError(CLIError):
    ...

class PluginLoadError(CLIError):
    ...

class OperationError(CLIError):
    ...
```

Avoid a huge exception taxonomy.

---

## 42. Known vs unknown failures

Known user/runtime failure:

```text
friendly message
no traceback by default
semantic exit code
```

Unexpected internal failure:

```text
concise internal-error message
non-zero exit
traceback only under --debug
cause preserved internally
```

Never discard exception chaining unnecessarily.

---

## 43. Error message structure

Recommended shape:

```text
Error: <short human explanation>

Reason:
  <specific cause>

Next:
  <one or more actionable actions>
```

Use sections only when they add clarity.

Tiny errors can remain one line.

---

# Part XIV — Exit Codes

## 44. Centralize exit codes

Do not invent numbers independently in commands.

Recommended initial semantic range:

```text
0   success

2   command-line usage error

10  configuration error
11  capability unavailable
12  unsupported runtime/platform
13  dependency unavailable

20  invalid input/data
21  resource not found

30  operation failed

40  plugin failure

70  unexpected internal failure
```

Exact numbers may be revised before declaring them stable.

Once documented for automation, changes require compatibility consideration.

---

## 45. Signal exits

Do not reinterpret operating-system signal semantics into arbitrary application codes without reason.

Interrupt and termination behavior should follow platform conventions where practical.

---

# Part XV — Configuration

## 46. Configuration precedence

Select and document one precedence model.

Recommended:

```text
explicit CLI option
      >
environment variable
      >
project configuration
      >
user configuration
      >
built-in default
```

A command MUST NOT invent a different precedence without a strong reason.

---

## 47. Configuration loading should be explainable

A useful future command:

```bash
scikitplot config explain KEY
```

Example:

```text
workers = 4

source:
  environment

candidates:
  CLI option         not set
  environment        4   ← selected
  project config     8
  user config        2
  default            1
```

This substantially improves supportability.

---

## 48. Configuration must not be mutated during reads

Commands such as:

```bash
scikitplot info
scikitplot config show
```

must not rewrite configuration files merely because they read them.

Writes should be explicit.

---

## 49. Secrets

Configuration output MUST redact secrets by default.

Potential secret material includes:

```text
tokens
passwords
credentials
authorization headers
private URLs containing credentials
API keys
session identifiers
```

A diagnostic command should have a centralized redaction policy.

---

# Part XVI — `info` and `doctor`

## 50. `info`

`info` should provide concise runtime facts.

Possible content:

```text
scikit-plots version
Python version
platform
architecture
installation location
selected configuration
available high-level capabilities
```

Keep the default output useful and short.

Use a deeper flag if needed:

```text
--verbose
```

or subcommands if information grows too large.

---

## 51. `doctor`

`doctor` is an active diagnostic command.

It may check:

```text
installation consistency
configuration readability
cache/write paths
optional dependencies
native extensions
plugin metadata
runtime compatibility
selected integrations
```

It should distinguish:

```text
PASS
WARN
FAIL
SKIP
```

---

## 52. Doctor must not become dangerous

By default, `doctor` SHOULD NOT:

- upload data;
- send telemetry;
- modify files;
- repair configuration;
- install dependencies;
- contact arbitrary remote systems;
- import every untrusted third-party plugin implementation.

Active or network checks must be explicit.

Examples:

```text
--network
--deep
--plugins
```

if those behaviors are implemented.

---

# Part XVII — Plugins

## 53. Plugin mechanism

Use Python package entry points for installed third-party CLI extensions.

Possible group:

```toml
[project.entry-points."scikitplot.cli"]
example = "package.cli:command"
```

Discovery should use `importlib.metadata`.

---

## 54. Discovery is not loading

Critical distinction:

```text
discover plugin metadata
        ≠
import plugin Python code
```

Normal startup may discover lightweight entry-point metadata.

It MUST NOT import every plugin merely to start the CLI.

---

## 55. Plugin trust boundary

An installed Python plugin is executable code.

Do not describe entry-point plugins as sandboxed.

A plugin load can execute arbitrary Python with the privileges of the running process.

Therefore:

- load only when needed;
- never silently auto-execute plugin callbacks during discovery;
- make the provider distribution visible;
- support diagnosing plugin load failures;
- isolate plugin failures from unrelated commands where possible.

---

## 56. Collision policy

A plugin MUST NOT silently override a first-party command.

Recommended precedence:

```text
built-in command
    >
plugin command
```

On collision:

```text
plugin command disabled
warning available through plugin diagnostics
```

Alternative explicit namespaces may be introduced later.

---

## 57. Plugin metadata cache

Do not prematurely build a persistent plugin cache.

Start with `importlib.metadata` discovery.

Only add caching if measurement demonstrates a real startup problem.

Persistent caches introduce invalidation and environment-consistency problems.

---

# Part XVIII — Security

## 58. Secure-by-default requirements

The CLI MUST:

- validate filesystem paths before destructive operations;
- avoid constructing shell commands from untrusted strings;
- avoid `shell=True` unless there is a documented unavoidable reason;
- avoid `eval`/`exec` for command dispatch;
- never treat file/module discovery as authorization;
- redact secrets in diagnostics;
- avoid automatic network access during bootstrap;
- preserve safe temporary-file handling;
- make overwrite/destructive behavior explicit;
- avoid importing unrelated plugins;
- preserve exception causes for debugging without leaking secrets by default.

---

## 59. External process execution

If a command launches another process, use argument arrays.

Preferred:

```python
subprocess.run(
    [sys.executable, "-m", "some.module", "--option", value],
    check=True,
)
```

Avoid:

```python
subprocess.run(
    f"python -m some.module --option {value}",
    shell=True,
)
```

unless the command intentionally requires shell semantics and inputs are controlled.

---

## 60. Filesystem writes

A command that writes data SHOULD define:

- destination;
- overwrite behavior;
- atomicity expectations;
- cleanup after failure;
- permissions where relevant.

Useful pattern:

```text
prepare
  ↓
write temporary
  ↓
fsync if required
  ↓
atomic replace
```

Use only where the operation warrants that complexity.

---

## 61. Destructive operations

Destructive operations should support one of:

```text
explicit command verb
explicit --force
interactive confirmation
--dry-run
```

Do not rely only on prompts because CI/non-interactive workflows need deterministic behavior.

---

## 62. Network behavior

No command should unexpectedly access the network because the user requested local information.

Network-dependent commands should clearly document the behavior.

Consider a future global policy:

```text
--offline
```

only if multiple commands genuinely need it.

---

# Part XIX — Runtime Resources, Signals, and Cleanup

## 63. Resource ownership

Every runtime resource should have a clear owner.

Examples:

```text
temporary directory
file handle
network session
thread executor
process pool
database connection
lock
memory map
```

Prefer context managers.

---

## 64. Cleanup

Cleanup must occur for:

- success;
- known errors;
- exceptions;
- keyboard interruption where feasible.

Do not rely solely on interpreter shutdown.

---

## 65. Ctrl+C

`KeyboardInterrupt` should normally result in:

- concise interruption behavior;
- cleanup;
- no giant traceback by default;
- conventional non-success termination.

Debug mode may expose additional details.

---

## 66. Async commands

Do not force the entire CLI into async architecture because one command is asynchronous.

Recommended boundary:

```text
Click callback
    │
    ▼
command-specific async runner
    │
    ▼
async service
```

Avoid nested event-loop hacks.

If async becomes common across many commands, reevaluate centrally.

---

## 67. Concurrency

Commands that create concurrency must:

- expose bounded concurrency;
- define cancellation behavior;
- clean up executors;
- avoid unlimited queues;
- avoid accidental import-time pools.

The bootstrap itself should remain single-process and simple.

---

# Part XX — Native, WASM, and Restricted Runtimes

## 68. Platform-aware capability detection

The CLI should distinguish:

```text
dependency missing
```

from:

```text
capability fundamentally unsupported in this runtime
```

This matters for environments such as:

- CPython desktop/server;
- Windows/macOS/Linux;
- containers;
- free-threaded builds;
- Pyodide;
- Emscripten;
- JupyterLite/xeus-python;
- restricted filesystem/network environments.

---

## 69. Do not pretend every CLI workflow works everywhere

A browser/WASM runtime may not provide a normal OS shell at all.

Therefore CLI code should be portable where reasonable, but CLI availability itself must not be assumed to be meaningful in every runtime.

Library logic should remain reusable independently from CLI transport.

---

## 70. Capability-first messages

Prefer:

```text
This command is unavailable in the current runtime.
```

with a reason.

Avoid exposing obscure loader/import failures to users when the platform limitation is already known.

---

# Part XXI — Shell Completion

## 71. Completion is a bootstrap workload

Shell completion can invoke command discovery frequently.

Therefore it must be:

- fast;
- deterministic;
- side-effect free;
- safe to call repeatedly.

---

## 72. Completion must respect lazy boundaries

Completion SHOULD obtain command names from metadata.

It should not import heavy feature commands simply to list candidates.

Nested option completion may require loading more metadata or a command implementation; keep this boundary explicit.

---

## 73. Dynamic completion

Dynamic values such as:

```text
remote runs
datasets
plugin resources
filesystem-heavy searches
```

should not perform expensive work on every TAB press unless explicitly designed and cached safely.

---

# Part XXII — Performance

## 74. Performance is an architectural contract

Measure at least:

```text
scikitplot --help
scikitplot --version
scikitplot <unknown-command>
shell completion bootstrap
scikitplot info
```

---

## 75. Import budget

Tests should assert that bootstrap paths do not import forbidden heavy modules.

Example conceptual forbidden set:

```text
numpy
scipy
matplotlib
sklearn
mlflow
streamlit
gradio
```

The exact set should reflect actual project dependencies.

Test behavior, not only wall-clock duration.

---

## 76. Time budget

Wall-clock thresholds vary by CI hardware.

Prefer:

1. record baseline;
2. use repeatable benchmark environment;
3. track regression percentage;
4. use generous hard ceilings only for gross regressions.

Do not create flaky tests around extremely small millisecond differences.

---

## 77. Memory budget

At minimum, compare:

```text
Python interpreter baseline
        vs
CLI --help
```

The CLI should not allocate large feature structures during bootstrap.

---

# Part XXIII — Testing Strategy

## 78. Test layers

Use four major layers:

```text
unit
integration
contract
performance/import
```

---

## 79. Registry unit tests

Test:

- unique command names;
- unique aliases;
- valid target syntax;
- deterministic ordering;
- no alias cycles;
- no built-in/plugin silent collision;
- metadata validation.

---

## 80. Loader tests

Test:

- valid target;
- missing module;
- missing target attribute;
- target is not a Click command;
- optional dependency missing;
- import raises unexpected exception;
- repeated load;
- cache reset.

---

## 81. Lazy import contract tests

Critical tests:

```text
scikitplot --help
```

must not import feature modules.

```text
scikitplot --version
```

must not import feature modules.

```text
scikitplot info
```

may import only its required dependency set.

A command should not import unrelated commands.

---

## 82. Help tests

Lazy loading can hide import-order and circular-import bugs.

Run:

```text
scikitplot --help
scikitplot info --help
scikitplot doctor --help
scikitplot config --help
scikitplot config show --help
...
```

for every registered command path.

---

## 83. Output contract tests

For JSON:

```text
stdout parses as JSON
stderr may contain diagnostics
stdout contains no ANSI escape codes
stdout contains no logging prefix
```

For text:

```text
pipe output remains readable
--color never emits no ANSI codes
```

---

## 84. Exit code tests

Each documented failure class should have at least one integration test.

---

## 85. Plugin tests

Test:

- zero plugins installed;
- one healthy plugin;
- broken plugin;
- plugin name collision;
- plugin optional dependency missing;
- plugin discovery does not import plugin;
- unrelated plugin failure does not break built-ins.

---

## 86. Signal/cleanup tests

Where practical:

- interrupt long-running command;
- ensure temporary resources are removed;
- ensure lock is released;
- ensure child process cleanup policy is followed.

---

## 87. Platform matrix

Run core CLI tests on supported Python/platform combinations.

Feature-specific commands may use capability-gated tests.

Do not mark the entire CLI unsupported merely because one optional command cannot run on a platform.

---

# Part XXIV — Dependency Policy

## 88. Dependency tiers

Define dependencies by runtime role.

### Tier 0 — Python standard library

Always available within supported Python.

### Tier 1 — CLI kernel

Small required dependencies needed for the public CLI itself.

Recommended:

```text
click
```

Potentially Rich only if deliberately accepted as a required CLI presentation dependency.

### Tier 2 — optional presentation

Example:

```text
rich
yaml serializer
```

### Tier 3 — feature dependencies

Examples:

```text
matplotlib
mlflow
streamlit
gradio
native ANN extension
```

### Tier 4 — plugins

Third-party installed extensions.

---

## 89. Dependency import rules

```text
Tier 0/1
    allowed during bootstrap

Tier 2
    load when renderer is needed

Tier 3
    load when related command path is used

Tier 4
    discover metadata early if needed;
    import plugin only when activated
```

---

# Part XXV — Compatibility and Deprecation

## 90. CLI is a public interface

Treat these as compatibility-sensitive:

- command names;
- option names;
- option meanings;
- machine-output schema;
- documented exit codes;
- configuration keys;
- environment variables.

---

## 91. Deprecation lifecycle

Suggested lifecycle:

```text
ACTIVE
  ↓
DEPRECATED
  ↓
HIDDEN optional transition
  ↓
REMOVED
```

Deprecation message should include:

```text
what is deprecated
replacement
planned removal version/date if known
```

Do not invent a removal deadline that the project cannot honor.

---

## 92. Compatibility aliases

Use aliases as temporary migration tools when renaming commands.

Example:

```text
show_versions
   ↓
deprecated alias
   ↓
info
```

Do not preserve every historical spelling forever.

---

# Part XXVI — Recommended Initial Reference Skeleton

## 93. `registry.py`

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommandSpec:
    name: str
    target: str
    short_help: str
    aliases: tuple[str, ...] = ()
    hidden: bool = False
    capabilities: tuple[str, ...] = ()


BUILTIN_COMMANDS = (
    CommandSpec(
        name="info",
        target="scikitplot._cli.commands.info:cli",
        short_help="Show runtime and installation information.",
    ),
    CommandSpec(
        name="doctor",
        target="scikitplot._cli.commands.doctor:cli",
        short_help="Diagnose the current environment.",
    ),
)
```

---

## 94. Registry index

Build immutable indexes once:

```python
BY_NAME = {spec.name: spec for spec in BUILTIN_COMMANDS}

BY_ALIAS = {
    alias: spec.name
    for spec in BUILTIN_COMMANDS
    for alias in spec.aliases
}
```

Validate collisions at import/test time.

This registry contains metadata only and is acceptable bootstrap work.

---

## 95. Loader

```python
from __future__ import annotations

import importlib

import click


def load_target(target: str) -> click.Command:
    module_name, attr_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    command = getattr(module, attr_name)

    if not isinstance(command, click.Command):
        raise TypeError(
            f"CLI target {target!r} did not resolve to a Click command."
        )

    return command
```

Production code must normalize errors using the project error taxonomy.

---

## 96. Metadata proxy concept

A lightweight proxy can expose:

```text
name
short help
hidden/deprecated metadata
```

without importing the real command.

On actual invocation it loads the target and delegates execution.

The exact Click subclass design should be implemented only after tests are written for:

```text
--help import behavior
completion import behavior
nested command behavior
Click context propagation
parameter parsing behavior
```

Do not create a proxy that only appears lazy but forces imports through Click help formatting.

---

# Part XXVII — Migration Plan

## 97. Migration rule

Do not perform a big-bang rewrite that deletes the working entry point before replacement behavior is tested.

Build the new kernel beside the old implementation, validate it, then switch the public entry point.

---

## 98. Phase 0 — freeze and characterize current behavior

Inventory:

```text
commands
aliases
options
exit behavior
stdout/stderr
dependencies
environment variables
configuration reads
current tests
```

Capture representative CLI snapshots.

Deliverable:

```text
CURRENT_CLI_BEHAVIOR.md
```

This can be a temporary engineering artifact.

---

## 99. Phase 1 — minimal new kernel

Implement only:

```text
app.py
registry.py
loader.py
errors.py
exit_codes.py
```

Commands:

```text
info
doctor
```

No plugin system yet.

Acceptance:

```text
--help works
--version works
unknown command works
info works
doctor works
heavy modules absent from bootstrap
```

---

## 100. Phase 2 — strict lazy metadata/proxy layer

Implement strict command metadata behavior.

Acceptance:

```text
top-level --help imports no feature command implementation
completion imports no unrelated implementation
invoking one command imports only that command path
```

This is the most important architectural milestone.

---

## 101. Phase 3 — runtime context and common policies

Add:

```text
context.py
logging.py
output.py
options.py
```

Normalize:

```text
-v / -vv
-q / -qq
--debug
--format
--color
```

Acceptance:

```text
JSON stdout is clean
logs use stderr
debug behavior is deterministic
```

---

## 102. Phase 4 — configuration

Add explicit configuration layering and introspection.

Acceptance:

```text
config show
config get
config path
```

Optional later:

```text
config explain
```

---

## 103. Phase 5 — capability system

Introduce only capabilities required by real migrated commands.

Acceptance:

```text
known-but-unavailable command fails cleanly
unsupported platform is distinguishable from missing package
doctor can report capability state
```

---

## 104. Phase 6 — migrate remaining useful commands

For every current command:

```text
KEEP
MIGRATE
DEPRECATE
DELETE
```

Do not automatically port examples or abandoned server launchers.

---

## 105. Phase 7 — plugins

Add entry-point plugin discovery after built-in loading behavior is stable.

Acceptance:

```text
plugin metadata discovery is lazy
plugin imports only on activation
collision policy enforced
broken plugin isolated
```

---

## 106. Phase 8 — completion and deeper performance work

Add completion tests and optimize only measured bottlenecks.

---

# Part XXVIII — Current File Disposition

## 107. `cli.py`

Recommended:

```text
MIGRATE → app.py
```

Keep temporary compatibility shim if needed.

Move out:

```text
command discovery
individual commands
server launch commands
logging setup
```

---

## 108. `_commands/`

Recommended:

```text
MIGRATE → commands/
```

Do not preserve automatic file scanning.

Each command becomes an explicitly registered adapter.

---

## 109. `_cmd_options/_cmd_options_click.py`

Recommended:

```text
PARTIAL MIGRATION
```

Preserve only options that represent real shared CLI policy.

Likely destinations:

```text
options.py
logging.py
context.py
```

Do not keep unrelated option groups simply because they were copied from another CLI design.

---

## 110. `_cmd_options/_cmd_options_optparse.py`

Recommended initial classification:

```text
DEPRECATE / DELETE candidate
```

Before removal, search the complete repository for imports and external compatibility commitments.

Do not migrate ~legacy parser machinery into the new architecture unless an active supported consumer requires it.

---

## 111. `_misc.py`

Recommended:

```text
REVIEW ITEM-BY-ITEM
```

For every helper ask:

```text
Does the CLI use it?
Is it project-specific?
Does the standard library already provide it?
Does it introduce unnecessary dependency or behavior?
```

Move small relevant helpers to focused modules.

Delete or leave outside the new CLI anything unrelated.

---

# Part XXIX — Anti-Patterns

## 112. Forbidden architecture patterns

### Import every command at startup

```python
for module in modules:
    import_module(module)
```

Avoid.

---

### Command auto-registration from every file

```text
drop file into commands/
    ↓
public CLI silently changes
```

Avoid.

---

### Heavy imports at module top level

```python
import matplotlib
import mlflow
import streamlit
```

inside bootstrap-visible modules.

Avoid.

---

### Business logic inside decorators/callbacks

Avoid 300-line Click callbacks.

---

### Broad exception swallowing

```python
except Exception:
    pass
```

Avoid in command loading/resolution.

---

### Logging to stdout

Avoid when stdout is a result channel.

---

### Boolean flag with `default=True` and no negative form

Avoid when users need to disable it.

---

### Automatic plugin execution

Plugin discovery must not equal plugin execution.

---

### Hidden network access

Do not make local commands unexpectedly connect remotely.

---

### Help that initializes the application

`--help` is documentation, not application execution.

---

# Part XXX — Review Checklist

## 113. New command checklist

Before merging a command:

- [ ] Is it a user workflow rather than a direct module mirror?
- [ ] Is the command explicitly registered?
- [ ] Does registry metadata avoid importing the implementation?
- [ ] Are heavy imports deferred?
- [ ] Are optional dependencies represented as capabilities where useful?
- [ ] Does `COMMAND --help` work?
- [ ] Does the command avoid unrelated imports?
- [ ] Is stdout reserved for results?
- [ ] Are logs/warnings on stderr?
- [ ] Is structured output valid?
- [ ] Are paths validated?
- [ ] Are secrets redacted?
- [ ] Are destructive actions explicit?
- [ ] Is cleanup deterministic?
- [ ] Are exit codes tested?
- [ ] Are errors actionable?
- [ ] Are platform restrictions explicit?
- [ ] Is there at least one integration test?

---

## 114. Root CLI checklist

- [ ] `scikitplot --help` is fast.
- [ ] `scikitplot --version` is fast.
- [ ] Top-level help has no feature imports.
- [ ] Unknown command does not load every command.
- [ ] Completion does not initialize features.
- [ ] No network access occurs during bootstrap.
- [ ] No filesystem writes occur during bootstrap.
- [ ] No plugin implementation is auto-imported.
- [ ] Logging is configured once.
- [ ] stdout/stderr policy is enforced.
- [ ] aliases are deterministic.
- [ ] command order is deterministic.

---

# Part XXXI — Architecture Decisions

## 115. Decision summary

### ADR-CLI-001 — Use Click as the CLI kernel

**Decision:** Keep Click as the core CLI framework.

**Reason:** The existing project already uses Click, and Click supports groups, contexts, completion, and lazy command patterns.

**Constraint:** Click must be an intentional runtime dependency if the public console script imports it.

---

### ADR-CLI-002 — Explicit first-party command registry

**Decision:** Built-in commands are registered explicitly through metadata.

**Rejected default:** filesystem/module scanning as the public registry.

---

### ADR-CLI-003 — Strict lazy activation

**Decision:** command existence/help metadata is separated from implementation import.

**Reason:** ordinary lazy loading may still import commands while rendering help.

---

### ADR-CLI-004 — Three lazy layers

**Decision:**

```text
command import
dependency import
resource initialization
```

are separate lazy boundaries.

---

### ADR-CLI-005 — CLI as adapter

**Decision:** scientific/application logic belongs outside Click callbacks.

---

### ADR-CLI-006 — stdout is the result channel

**Decision:** diagnostics and logging go to stderr.

---

### ADR-CLI-007 — Machine output is a stable contract

**Decision:** structured output must be clean, parseable, and documented.

---

### ADR-CLI-008 — Plugins use package metadata

**Decision:** future installed plugins use Python entry points.

**Constraint:** discovery does not imply import or trust.

---

# Part XXXII — Suggested Implementation Order

## 116. First implementation slice

Implement this first:

```text
scikitplot/_cli/
├── __init__.py
├── __main__.py
├── app.py
├── registry.py
├── loader.py
├── errors.py
├── exit_codes.py
└── commands/
    ├── __init__.py
    ├── info.py
    └── doctor.py
```

Do **not** start with:

```text
plugins
YAML
server launchers
complex configuration writes
many feature commands
persistent caches
async abstractions
```

---

## 117. First success scenario

```bash
scikitplot --help
```

Expected:

```text
Usage: scikitplot [OPTIONS] COMMAND [ARGS]...

Commands:
  doctor  Diagnose the current environment.
  info    Show runtime and installation information.
```

Import contract:

```text
Click/kernel modules     yes
registry metadata        yes

Matplotlib               no
SciPy                    no
scikit-learn             no
MLflow                   no
Streamlit                no
Gradio                   no
feature commands         no
plugins                  no implementation imports
```

---

## 118. Second success scenario

```bash
scikitplot info --format json
```

Expected:

```json
{
  "scikitplot": {
    "version": "..."
  },
  "python": {
    "version": "..."
  },
  "platform": {
    "system": "..."
  }
}
```

Contract:

```text
stdout = JSON only
stderr = diagnostics only
exit   = 0
```

---

## 119. Third success scenario

```bash
scikitplot optional-feature ...
```

when dependency is missing:

```text
Error: 'optional-feature' is unavailable.

Reason:
  Required capability 'example' is not available.

Next:
  Run `scikitplot doctor`.
```

No raw traceback unless debug mode requests it.

---

# Part XXXIII — Future-Proof Features

## 120. Add only after the kernel is stable

Potential future improvements:

```text
config explain
capability explain
plugin doctor
command provenance
structured diagnostic bundles
offline policy
dry-run framework
completion metadata cache
command execution tracing
deprecated-command migration hints
feature-specific profiles
```

These are useful extensions, not prerequisites for the first rebuild.

---

## 121. Command provenance

A useful plugin-era feature:

```bash
scikitplot plugins info some-command
```

could show:

```text
command:
  foo

provider:
  distribution-name

version:
  1.2.3

target:
  package.cli:command
```

This makes third-party CLI behavior easier to debug.

---

## 122. Diagnostic bundle

A future safe command might produce a support bundle:

```bash
scikitplot doctor --report PATH
```

Requirements:

- redact secrets;
- no automatic upload;
- deterministic schema;
- user controls destination;
- clearly state collected information.

---

# Part XXXIV — Definition of Done

## 123. Rebuild is architecturally complete when

The new CLI should not be considered complete merely because commands run.

It is complete when all of the following are true:

### Bootstrap

- `--help` works without feature imports.
- `--version` works without feature imports.
- unknown commands fail without importing all commands.
- bootstrap performs no network activity.
- bootstrap performs no unintended filesystem writes.

### Lazy loading

- command metadata is separate from command code;
- one command does not import unrelated commands;
- optional dependencies load only on relevant paths;
- expensive resources initialize only when needed.

### UX

- command and option spelling is consistent;
- help is concise and useful;
- aliases are explicit;
- deprecations are actionable.

### Automation

- JSON output is valid and uncontaminated;
- stdout/stderr rules are enforced;
- exit codes are centralized.

### Reliability

- errors are normalized;
- causes are preserved;
- cleanup is deterministic;
- interruption behavior is tested.

### Extensibility

- built-ins use an explicit registry;
- plugin architecture does not require redesigning the root CLI;
- plugin discovery does not auto-import plugin code.

### Security

- no hidden command execution;
- no silent plugin override;
- no unsafe shell construction;
- diagnostics redact sensitive values.

### Testing

- every command path has a help test;
- lazy import contracts are tested;
- machine output is parsed in tests;
- dependency failure paths are tested;
- supported platform matrix is covered.

---

# Part XXXV — Immediate Next Work

## 124. Next implementation task

The next engineering task should be to build the smallest strict-lazy prototype:

```text
1. `CommandSpec`
2. explicit built-in registry
3. metadata/proxy Click command
4. command loader
5. root application
6. `info`
7. `doctor`
8. import-contract tests
```

Do not migrate every existing helper first.

The prototype should prove the architecture before bulk migration.

---

## 125. Prototype acceptance test

The prototype succeeds if:

```text
A. `scikitplot --help`
   does not import `scikitplot._cli.commands.info`

B. `scikitplot info --help`
   loads only what is required to render command-specific help

C. `scikitplot info`
   loads the real info implementation

D. `scikitplot doctor`
   does not load unrelated feature modules

E. JSON output contains no logging text

F. a broken optional command does not prevent `scikitplot --help`

G. a broken plugin does not prevent built-in commands from running
```

If these conditions cannot be met cleanly, revise the proxy/loader boundary before migrating additional commands.

---

# References

Use primary/official sources when validating implementation behavior.

## Click

- Click documentation
  https://click.palletsprojects.com/

- Complex applications and lazy subcommands
  https://click.palletsprojects.com/en/stable/complex/

- Commands, groups, and contexts
  https://click.palletsprojects.com/en/stable/commands/

- Shell completion
  https://click.palletsprojects.com/en/stable/shell-completion/

## Python

- `importlib.metadata`
  https://docs.python.org/3/library/importlib.metadata.html

- `importlib`
  https://docs.python.org/3/library/importlib.html

- `logging`
  https://docs.python.org/3/library/logging.html

- `subprocess`
  https://docs.python.org/3/library/subprocess.html

- `signal`
  https://docs.python.org/3/library/signal.html

## Python Packaging

- Entry points specification
  https://packaging.python.org/specifications/entry-points/

- Creating and discovering plugins
  https://packaging.python.org/guides/creating-and-discovering-plugins/

- `pyproject.toml` specification
  https://packaging.python.org/en/latest/specifications/pyproject-toml/

---

# Final Design Rule

The CLI must remain understandable from one sentence:

> **The bootstrap knows what commands exist; the runtime loads only the command, dependency, and resources that the user actually activates.**

That rule should remain true as the CLI grows from a few built-in commands to feature groups, optional integrations, native capabilities, and third-party plugins.
