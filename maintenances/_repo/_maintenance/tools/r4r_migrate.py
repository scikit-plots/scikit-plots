"""Increment 4R: split tasks/todo.md and tasks/lessons.md into their planes.

Reads both files from a commit (argv[2]) of the repository (argv[1]) so the
source cannot change under the run, and writes into the working tree. Every
section and every rule category is mapped explicitly; an unmapped one stops
the run. The manifest records each part (source line range, length, sha256,
destination) so the move can be proved lossless.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import re
import subprocess
import sys

logger = logging.getLogger(__name__)

REPO = pathlib.Path(sys.argv[1])
COMMIT = sys.argv[2]

CORPUS_H = "maintenances/corpus/_maintenance/history"
REPO_H = "maintenances/_repo/_maintenance/history"
CLEAN_H = "maintenances/cleanprompt/_maintenance/history"
SPHINX_H = "maintenances/_externals/_sphinx_ext/_maintenance/history"
FEEDBACK_H = "maintenances/_externals/_sphinx_ext/_sphinx_feedback/_maintenance/history"
CORPUS_ACTIVE = "maintenances/corpus/todo/todo.md"

#: Section heading (exact) -> destination. Results reviews are mapped on
#: their own, because one ("Results Review - 2026-10-06") reviews the task
#: two sections above it, not the one before it.
TODO_MAP: dict[str, str] = {
    "# Active Tasks": "preamble",
    "## Task: round 33 / Run 4 - structured chunking measured by the harness (R4-2, R4-3)": (
        CORPUS_ACTIVE
    ),
    "## Task: round 33 / Run 3 - ingestion contract (chunker registry, config strictness, spans, cache, logs, resources)": (
        f"{CORPUS_H}/r33_run3_ingestion.md"
    ),
    "## Task: round 33 / Run 2 - retrieval contract (one fusion, one filter, durable identities, one fallback rule)": (
        f"{CORPUS_H}/r33_run2_retrieval.md"
    ),
    "## Task: round 33 / Run 4-A (pulled forward) - corpus evaluation: metrics, sklearn scorers, LLM judges, exportable reports": (
        f"{CORPUS_H}/r33_run4a_evaluation.md"
    ),
    "## Task: round 33 / AI-plane Run 1 - correctness floor (silent wrong answers, a crash, a leak)": (
        f"{CORPUS_H}/r33_run1_correctness.md"
    ),
    "## Task: round 32 - CI red on PR 864: leaked Sphinx state, a py3.9 loop-bound lock, contract drift, a Windows env": (
        f"{REPO_H}/r32_ci_pr864.md"
    ),
    "## Task: round 31 - PR/issue labeling: one policy, generated configs, semantic checks": (
        f"{REPO_H}/r31_labeling.md"
    ),
    "## Task: round 30 - CI collision (again), options regression, forge engine, GitLab, Bitbucket, live-Space check, logging flake": (
        f"{FEEDBACK_H}/r30_feedback_forges.md"
    ),
    "## Task: round 29b - CI collision fix; feedback for any deployment and storage": (
        f"{FEEDBACK_H}/r29b_feedback_storage.md"
    ),
    "## Task: round 29 - page feedback on any site, with or without the AI assistant": (
        f"{FEEDBACK_H}/r29_page_feedback.md"
    ),
    "## Task: cleanprompt round 26 - pattern-risk check (warn by default, user-tuned), custom surrogate sets, generator design": (
        f"{CLEAN_H}/r26_pattern_risk.md"
    ),
    "## Results Review - cleanprompt round 26 - 2026-10-10": (
        f"{CLEAN_H}/r26_pattern_risk.md"
    ),
    "## Task: cleanprompt round 25 - truthful readiness, safe deployment files, Unicode detection view, multi-page guide": (
        f"{CLEAN_H}/r25_readiness.md"
    ),
    "## Results Review - cleanprompt round 25 - 2026-10-10": (
        f"{CLEAN_H}/r25_readiness.md"
    ),
    "## Task: Partial distributions of scikit-plots (`libs/<name>`), in the manner of mlflow-skinny": (
        f"{REPO_H}/partial_distributions.md"
    ),
    "## Task: Round 2 - version-free declarations, scikit-learn 1.3+, Python 3.15, Sphinx extensions, WebAssembly": (
        f"{REPO_H}/r02_declarations.md"
    ),
    "## Results Review - 2026-10-06": f"{REPO_H}/partial_distributions.md",
    "## Results Review - Round 2 - 2026-10-06": f"{REPO_H}/r02_declarations.md",
    "## Task: Round 3 - first CI run: zizmor, verify legs, full test job, Cython annotation report": (
        f"{REPO_H}/r03_first_ci.md"
    ),
    "## Results Review - Round 3 - 2026-10-07": f"{REPO_H}/r03_first_ci.md",
    "## Task: Round 4 - Windows leg, core/parts compatibility number, staging wheels for every platform": (
        f"{REPO_H}/r04_windows_leg.md"
    ),
    "## Results Review - Round 4 - 2026-10-07": f"{REPO_H}/r04_windows_leg.md",
    "## Task: Round 5 - second CI run: what the Windows leg still names, threads at run time, bundle manifest": (
        f"{REPO_H}/r05_second_ci.md"
    ),
    "## Results Review - Round 5 - 2026-10-07": f"{REPO_H}/r05_second_ci.md",
    "## Task: Round 6 - third CI run: the Windows leg only": (
        f"{REPO_H}/r06_third_ci.md"
    ),
    "## Results Review - Round 6 - 2026-10-07": f"{REPO_H}/r06_third_ci.md",
    "## Task: Round 7 - fourth CI run: two tests left on Windows": (
        f"{REPO_H}/r07_fourth_ci.md"
    ),
    "## Results Review - Round 7 - 2026-10-08": f"{REPO_H}/r07_fourth_ci.md",
    "## Task: Round 8 - the annoy wheels: the test step cannot find scikit-plots-skinny": (
        f"{REPO_H}/r08_annoy_wheels.md"
    ),
    "## Task: Round 9 - `_sphinx_ext`: one stack, two import roots, 15 failures": (
        f"{SPHINX_H}/r09_sphinx_ext_stack.md"
    ),
}

RULES = "plugins/scikitplot-dev/skills/_engineering/rules"
#: Rules file -> (title, categories). Each category is declared in one file.
RULE_FILES: dict[str, tuple[str, tuple[str, ...]]] = {
    "verification.md": (
        "Verification and evidence",
        (
            "Verification",
            "Evidence",
            "Diagnostics",
            "Gates",
            "Analysis",
            "Process",
            "Proof",
        ),
    ),
    "testing.md": (
        "Tests",
        (
            "Tests",
            "Timing",
            "Thresholds",
            "Scanners",
            "Shared defaults",
            "Serialisation",
        ),
    ),
    "tooling.md": (
        "Tooling and shell",
        ("Tooling", "Shell", "Parsing", "Git", "Waiting on a background process"),
    ),
    "delivery.md": (
        "Delivery, CI and generated configuration",
        ("Delivery", "CI", "Generated configuration", "Labels", "Labeler globs"),
    ),
    "documentation.md": ("Documentation", ("Documentation", "Docstrings")),
    "numpydocstr.md": ("NumPy docstrings", ("Numpydocstr", "NumPy docstrings")),
    "packaging.md": (
        "Packaging and dependencies",
        (
            "Packaging",
            "Partial distributions",
            "Optional dependencies",
            "Optional SDKs",
            "Build",
            "Data",
            "Paths",
        ),
    ),
    "portability.md": (
        "Portability",
        ("Portability", "Subprocess", "Messages", "Files", "Async"),
    ),
    "contracts.md": (
        "API and data contracts",
        (
            "Validation",
            "Inputs",
            "Options",
            "Contracts",
            "Interfaces",
            "Errors",
            "Fallbacks",
            "Responses",
            "Loaders",
            "Exporters",
            "Identities",
            "Untrusted structured output",
            "Name tables",
            "Design",
            "Protection",
            "Guards",
            "Scope",
            "Caller-supplied resources",
        ),
    ),
    "numerics.md": (
        "Numerics and scoring",
        ("Numerics", "Exactness", "Scoring", "Native kernels"),
    ),
    "native.md": ("Native code", ("C++", "Native memory")),
    "state.md": (
        "Concurrency, caches and stores",
        ("Concurrency", "Caches", "Stores that gain deletion", "SQLite"),
    ),
    "text.md": ("Text, Unicode and markup", ("Unicode", "Spans", "Markup parsers")),
}


def git_show(path: str) -> str:
    """Return the contents of a file at a given path in the target commit."""
    return subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [  # ruff: ignore[start-process-with-partial-path]
            "git",
            "-C",
            str(REPO),
            "show",
            f"{COMMIT}:{path}",
        ],
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")


def sha(text: str) -> str:
    """Return the SHA-256 hex digest of *text* encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_at(text: str, pattern: str) -> list[tuple[int, str]]:
    """Split *text* before each line matching *pattern*; keep line numbers."""
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if re.match(pattern, line)]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    parts = []
    for a, b in zip(starts, [*starts[1:], len(lines)]):
        parts.append((a + 1, "".join(lines[a:b])))
    return parts


def main() -> None:  # ruff: ignore[too-many-branches]
    """Migrate task and lesson content from the source commit into its planes."""
    manifest: dict = {"source_commit": COMMIT, "sources": {}}
    outputs: dict[str, list[str]] = {}

    # ---------------------------------------------------------------- todo
    todo = git_show("tasks/todo.md")
    parts = split_at(todo, r"#{1,2} ")
    records = []
    for line_no, part in parts:
        heading = part.splitlines()[0]
        if heading not in TODO_MAP:
            logger.error("unmapped todo section at line %s: %r", line_no, heading)
            sys.exit(1)
        dest = TODO_MAP[heading]
        records.append(
            {
                "line": line_no,
                "heading": heading,
                "chars": len(part),
                "sha256": sha(part),
                "dest": dest,
            }
        )
        if dest != "preamble":
            outputs.setdefault(dest, []).append(part)
    assert len(records) == len(  # ruff: ignore[assert]
        TODO_MAP,
    ), "a mapped heading did not occur"
    assert "".join(p for _, p in parts) == todo  # ruff: ignore[assert]
    manifest["sources"]["tasks/todo.md"] = {
        "sha256": sha(todo),
        "chars": len(todo),
        "parts": records,
        "preamble": next(p for _, p in parts if p.startswith("# Active Tasks")),
    }

    # ------------------------------------------------------------- lessons
    lessons = git_show("tasks/lessons.md")
    parts = split_at(lessons, r"(### Rule \d+:|## )")
    records = []
    category_file = {c: f for f, (_, cats) in RULE_FILES.items() for c in cats}
    assert len(category_file) == sum(  # ruff: ignore[assert]
        (len(c) for _, c in RULE_FILES.values()),
    ), "a category is declared twice"
    seen_ids = []
    used_categories = set()
    rule_parts: dict[str, list[tuple[int, str]]] = {}
    for line_no, part in parts:
        heading = part.splitlines()[0]
        m = re.match(r"### Rule (\d+): (.+?) - ", heading)
        if m:
            rid, category = int(m.group(1)), m.group(2)
            if category not in category_file:
                logger.error("unmapped rule category %r (Rule %s)", category, rid)
                sys.exit(1)
            used_categories.add(category)
            seen_ids.append(rid)
            dest = f"{RULES}/{category_file[category]}"
            rule_parts.setdefault(dest, []).append((rid, part))
        elif heading in ("# Lessons Learned", "## Active Rules (Currently Applied)"):
            dest = "preamble"
        elif heading in ("## Pattern Analysis", "## Effectiveness Metrics"):
            dest = "plugins/scikitplot-dev/skills/_engineering/PATTERNS.md"
            outputs.setdefault(dest, []).append(part)
        else:
            logger.error("unmapped lessons section at line %s: %r", line_no, heading)
            sys.exit(1)
        records.append(
            {
                "line": line_no,
                "heading": heading,
                "chars": len(part),
                "sha256": sha(part),
                "dest": dest,
            }
        )
    assert "".join(p for _, p in parts) == lessons  # ruff: ignore[assert]
    assert sorted(seen_ids) == list(  # ruff: ignore[assert]
        range(1, len(seen_ids) + 1),
    ), "rule ids not 1..N"
    unused = set(category_file) - used_categories
    assert (  # ruff: ignore[assert]
        not unused
    ), f"declared but unused categories: {sorted(unused)}"
    manifest["sources"]["tasks/lessons.md"] = {
        "sha256": sha(lessons),
        "chars": len(lessons),
        "parts": records,
        "preamble": "".join(
            p
            for _, p in parts
            if p.startswith(("# Lessons Learned", "## Active Rules"))
        ),
    }

    # ------------------------------------------------------------- removed
    removed = git_show("tasks/removed-files.txt")
    dest = f"{REPO_H}/partial_distributions_removed_files.txt"
    manifest["sources"]["tasks/removed-files.txt"] = {
        "sha256": sha(removed),
        "chars": len(removed),
        "parts": [
            {
                "line": 1,
                "heading": removed.splitlines()[0],
                "chars": len(removed),
                "sha256": sha(removed),
                "dest": dest,
            }
        ],
    }

    # --------------------------------------------------------------- write
    def write(rel: str, text: str) -> None:
        path = REPO / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")

    write(dest, removed)
    origin = f"tasks/todo.md at {COMMIT}"
    for rel, texts in outputs.items():
        if rel == CORPUS_ACTIVE:
            continue  # written by hand around the archived Run 4 section
        if rel.endswith("PATTERNS.md"):
            head = (
                "# Patterns and metrics\n\n"
                "Recurring patterns behind the rules in [`rules/`](rules/) and the\n"
                f"review counts per round, moved verbatim from `tasks/lessons.md` at\n"
                f"{COMMIT}. Append new observations under the matching heading.\n\n"
            )
        else:
            head = (
                f"<!-- Archived verbatim from {origin} (increment 4R).\n"
                "     Do not edit: maintenances/_repo/_maintenance/tools/check_records.py\n"
                "     checks every archived part against records_manifest.json. -->\n\n"
            )
        write(rel, head + "".join(texts))
    for rel, items in rule_parts.items():
        name = rel.rsplit("/", 1)[1]
        title, cats = RULE_FILES[name]
        ids = [rid for rid, _ in items]
        assert ids == sorted(ids)  # ruff: ignore[assert]
        head = (
            f"# {title}\n\n"
            f"Categories: {', '.join(cats)}\n\n"
            "Rules keep the global number they were given; a new rule takes the next\n"
            "free number (see [`../SKILL.md`](../SKILL.md)).\n\n"
        )
        write(rel, head + "".join(p for _, p in items))
    # The active Run 4 section is stored for the hand-written file.
    pathlib.Path("/home/claude/r34/run4r/run4_section.md").write_text(
        "".join(outputs[CORPUS_ACTIVE]), encoding="utf-8", newline=""
    )
    manifest_path = "maintenances/_repo/_maintenance/records_manifest.json"
    write(manifest_path, json.dumps(manifest, indent=1, ensure_ascii=True) + "\n")
    logger.info(
        "Migrated source parts: %s",
        json.dumps({k: len(v["parts"]) for k, v in manifest["sources"].items()}),
    )
    logger.info(
        "Rules files: %s",
        {r.rsplit("/", 1)[1]: len(v) for r, v in rule_parts.items()},
    )


if __name__ == "__main__":
    main()
