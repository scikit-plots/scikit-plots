# Project memory

Curated, checked-in facts about this repository that every contributor and
every AI session should start from. This file is read by people and by
agents through `.claude/init` and `AGENTS.md`; Claude Code does not load a
root `MEMORY.md` on its own.

## Rules for this file

- Keep it under 200 lines and 25 KB. Move detail into the owning record and
  leave a one-line pointer here.
- Admit a fact only if it is durable (true for months), verified in this
  checkout, and not already stated by a rule, a skill or a ledger.
- Each fact names its source (a file, a command or a pull request) and the
  date it was last checked (`YYYY-MM-DD`).
- Never write secrets, tokens, account names, personal data, private file
  names or private file contents. A baseline records counts only.
- When a fact stops being true, replace it in the same pull request that
  changed it. Do not keep history here; git keeps it.

## Memory layers (where each kind of knowledge lives)

| Layer | Holds | Where |
|---|---|---|
| Instructions | Disclosure, summary style, read order | `AGENTS.md`, `.claude/init` |
| Rules | Standing, path-scoped instructions | `.claude/rules/*.md`, `plugins/_shared/skills/scikitplot-engineering-rules/` |
| Skills | Procedures loaded on demand | `plugins/_shared/skills/*/SKILL.md` (copied into both plugins) |
| Durable facts | This file | `MEMORY.md` |
| Active state | Open findings, next actions | `upcoming_changes/`, `maintenances/*/todo/todo.md`, `orient.py` |
| Retrieval | The live public API | `python -m scikitplot api <module>` |

Machine-local agent memory (under the user's home directory) is personal and
is never copied into this repository.

## Durable facts

- Public API listing is generated at run time, never stored:
  `python -m scikitplot api scikitplot.corpus`. Source: PR 942, PR 943.
  Checked 2026-11-16.
- `scikitplot.corpus` exports `DocumentReader` and `ZipReader`;
  `DocumentReader.create(path)` returns a reader whose `get_documents()`
  yields documents. Source: `python -m scikitplot api scikitplot.corpus`.
  Checked 2026-11-16.
- Hostile inputs are gated in CI, one child process per sample, 20 s and
  64 MiB ceilings. Source:
  `scikitplot/corpus/_readers/tests/test__hostile_samples.py`. Checked
  2026-11-16.
- Archive and Office readers refuse expansion above 50 output bytes per
  input byte; ZipReader also caps one tree at 256 MiB. Source: PR 938,
  PR 941. Checked 2026-11-16.
- Generators and CI checks must not import the installed package; load
  helpers by file path. Source: lesson CI-12 in
  `maintenances/_plugins/todo/todo.md`. Checked 2026-11-16.
- Plugin versions follow the library version (`0.4.0.post11` becomes
  `0.4.0+post.11`; dev builds add a commit stamp). Source:
  `build_skills_repo.py`, PR 944. Checked 2026-11-16.

## Real-life testing

Procedure: skill `scikitplot-real-data-testing` (both plugins) and rule
`.claude/rules/real-life-testing.md`. Public sources and their terms:
`plugins/_shared/skills/scikitplot-real-data-testing/references/sources.md`.

Baseline, private mixed archive, documents only (md, txt, pdf, pptx, zip),
`reader_snapshot.py snapshot --repeat 2`, Python 3.11, Linux:

| Date | Files | ok | Errors | Unstable | Chunks | Seconds | Peak RSS |
|---|---|---|---|---|---|---|---|
| 2026-11-16 | 145 | 145 | 0 | 0 | 569 | 52 | 302 MiB |

A new run that adds errors or unstable files, or that doubles time or
memory on comparable input, is a regression to investigate before merging.
