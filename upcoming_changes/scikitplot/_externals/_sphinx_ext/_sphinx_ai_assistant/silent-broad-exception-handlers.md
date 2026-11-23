---
title: "_sphinx_ext/_sphinx_ai_assistant: broad exception handlers that do nothing (16 places)"
status: open
kind: "reliability"
area: "scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant"
discovered_during: "Package sweep, tools/maint_tools/sweep_static.py on main cba9383 (2026-11-24)"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# _sphinx_ext/_sphinx_ai_assistant: broad exception handlers that do nothing (16 places)

## Summary

In the AI assistant proxy, 16 handlers catch `Exception` (or everything) and only `pass` or `continue`, with no log line and no `# noqa` recording a decision.

## Why it matters

Every failure inside those blocks, including bugs (`TypeError`, `KeyError`) and resource errors, disappears; the caller sees a partial result and nobody can tell why.

## Current evidence

- `python -B tools/maint_tools/sweep_static.py --list silent-broad-except` (main cba9383), unacknowledged findings:
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_page_feedback/_service/_forge.py:195`: `except Exception:  # ruff: ignore[blind-except]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_providers/gemini.py:824`: `except Exception:  # ruff: ignore[blind-except, try-except-in-loop]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_providers/gemini.py:829`: `except Exception:  # ruff: ignore[blind-except]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_providers/gemini.py:937`: `except Exception:  # ruff: ignore[blind-except]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_resource_transport.py:245`: `except Exception:  # ruff: ignore[blind-except, try-except-in-loop]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_storage.py:2443`: `except Exception:  # ruff: ignore[blind-except]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_storage.py:2544`: `except Exception:  # ruff: ignore[blind-except]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_zip_artifact.py:156`: `except Exception:  # ruff: ignore[blind-except, try-except-in-loop]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_zip_artifact.py:616`: `except Exception:  # ruff: ignore[blind-except, try-except-in-loop]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_zip_workspace.py:300`: `except Exception:  # ruff: ignore[blind-except]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_zip_workspace.py:801`: `except Exception:  # ruff: ignore[blind-except]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_zip_workspace.py:825`: `except Exception:  # ruff: ignore[blind-except, try-except-in-loop]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_utils/_zip_workspace.py:1164`: `except Exception:  # ruff: ignore[blind-except]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/continue_recovered_trust.py:394`: `except Exception:  # ruff: ignore[blind-except]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/probe_redis_authority.py:87`: `except Exception:  # ruff: ignore[blind-except]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/security/verify_native_status_provenance.py:342`: `except Exception:  # ruff: ignore[blind-except, try-except-in-loop]`

## Root cause / current understanding

Best-effort clean-up and optional steps were written with the broadest handler.

## Expected behavior

Each handler names the exceptions it expects; an unexpected one propagates or is logged at debug level with the operation and path; a deliberate best-effort catch carries `# noqa: BLE001` with the reason.

## Affected paths and ownership

- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant`

## Constraints and non-goals

Keep best-effort clean-up best effort (it must not mask the original error); only narrow and log.

## Edge cases to cover

- Clean-up during an exception already being raised
- Permission errors on read-only files
- Interpreter shutdown

## Proposed direction

Narrow each handler (`OSError`, `ValueError`, ...), log at debug level, and acknowledge the remaining deliberate ones with a reason.

## Verification / acceptance criteria

- `sweep_static.py --list silent-broad-except` prints nothing unacknowledged for the area

## Documentation impact

none

## Release-note promotion

Not required (no shipped behaviour changes).
