---
title: "CI: no dependency vulnerability scan or software bill of materials for wheels"
status: open
kind: "other"
area: "tools"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "tools"
towncrier_type: "feature"
towncrier_fragment: ""
---

# CI: no dependency vulnerability scan or software bill of materials for wheels

## Summary

Wheel and release workflows build and publish packages, but no job scans the resolved dependencies for known vulnerabilities or produces a software bill of materials; attestation steps appear only in the Docker image workflows.

## Why it matters

Users in regulated settings (clinical, finance) need an SBOM and a clean vulnerability report; vulnerable pins in the uv locks go unnoticed.

## Current evidence

- `grep -rln -E 'pip-audit|osv-scanner|cyclonedx|sbom' .github/workflows` finds nothing; `attest` appears only in `ci_docker_image_builder.yml` and the Docker registry workflows (main cba9383).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A scheduled and pull-request job that runs a vulnerability audit on the uv locks and requirements, and a release step that writes a CycloneDX or SPDX SBOM next to the wheels.

## Affected paths and ownership

- `.github/workflows/`
- `docker/env_uv/` (locks)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Vulnerabilities without a fixed version (reported, not failing)
- Vendored code (listed in the SBOM)

## Proposed direction

Use maintained, pinned actions; zizmor and action-validator clean.

## Verification / acceptance criteria

- The job reports on every lock; the release includes an SBOM

## Documentation impact

Security policy (`SECURITY.md`).

## Release-note promotion

Required: `tools` / `feature`, one bullet, when the change lands.
