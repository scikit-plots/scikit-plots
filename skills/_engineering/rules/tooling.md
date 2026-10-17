# Tooling and shell

Categories: Tooling, Shell, Parsing, Git, Waiting on a background process

Rules keep the global number they were given; a new rule takes the next
free number (see [`../SKILL.md`](../SKILL.md)).

### Rule 3: Shell - a process pattern must not match the command that searches for it
- **When:** using `pkill -f` / `pgrep -f`.
- **Then:** bracket the first letter of every alternative (`"[r]un-matrix"`), so the pattern text is not itself a match.
- **Verified by:** the command exits 0/1, not 143/144.
- **Added:** 2026-10-06. Root cause: the pattern occurred literally in the shell's own command line, so the shell killed itself.

### Rule 6: Tooling - generated files must be what the repository's formatter would write
- **When:** a generator emits a file that pre-commit hooks also process (`.py`, code blocks in `.md`).
- **Then:** emit formatter-stable output (exploded literals with trailing commas, double quotes) and keep a test that runs `ruff check` and `ruff format --check` on it.
- **Verified by:** `TestGeneratedFilesPassTheRepositoryLinter` in `libs/_tools/tests/test_generate.py`.
- **Added:** 2026-10-06. Root cause: a hook that rewrites a generated file makes the drift check (`python -m libs._tools check`) fail on every commit.

### Rule 7: Tooling - reading the source tree must not write into it
- **When:** build tooling needs a value from a module inside the package.
- **Then:** `compile()` + `exec()` the file's text instead of importing it, and run tools with `PYTHONDONTWRITEBYTECODE=1`.
- **Verified by:** the "build leaves nothing behind" row of the verify report.
- **Added:** 2026-10-06. Root cause: importing `scikitplot/_distributions.py` by path created `__pycache__` inside the tree, which the next build then staged.

### Rule 8: Shell - a process pattern must not occur anywhere else on the same command line
- **When:** a `pkill -f`/`pgrep -f` shares a command line with any other command.
- **Then:** run the kill as its own command; bracketing the first letter (Rule 3) protects only the pattern's own text, not a later command that contains the same words.
- **Verified by:** the kill command contains no other text that matches its pattern.
- **Added:** 2026-10-06. Root cause: `pkill -f "[l]ibs._tools verify"; ... python -m libs._tools verify ...` in one line: the shell's own command line contained the unbracketed words further on, so it matched and was killed.

### Rule 21: Git - an ignore rule ending in `/` does not cover a symbolic link
- **When:** ignoring a path that may be a directory or a link to one.
- **Then:** write the rule without the trailing slash.
- **Verified by:** `git status --short --untracked-files=all` does not list the link.
- **Added:** 2026-10-07. Root cause: `/*/scikitplot/` ignored the staged directory but not the link of the earlier layout, which could then be committed.

### Rule 33: Parsing - read a tool's output by its grammar, not by where it happens to stop
- **When:** lines are taken from another program's report (pytest's short summary).
- **Then:** match the lines that have the form of an entry (`FAILED|ERROR`, one space, a node id ending in `.py`), over the whole section, and take the section from standard output only.
- **Verified by:** `TestFailureReporting`: a log record that starts with `ERROR`, a multi-line reason, and standard error that imitates a summary.
- **Added:** 2026-10-07. Root cause: two fixes in a row, each for the previous one's blind spot: first every `ERROR` line was an entry, then the list ended at the first line that was not one. On CI pytest prints whole messages there.

### Rule 51: Shell - text that may contain the heredoc delimiter goes in a file, not a heredoc
- **When:** a command embeds a document (a README, a transcript, a shell example) in a heredoc, and that document may itself contain the delimiter word on a line of its own (`END`, `EOF`).
- **Then:** write the content with the file-writing tool to a script in the scratchpad and run the script; never embed such content in `<<'END'`.
- **Verified by:** no heredoc in the session's commands carries a document that contains its delimiter; `grep -n "^END$" <content>` before embedding.
- **Added:** 2026-10-09. Root cause: a Python edit of the cleanprompt README was passed in `<<'END'`; the README's own `<<'END'` example ended the heredoc early, the edit did not run, and the shell executed the following lines, one of them a `pip install` (already satisfied; checked, nothing changed).

### Rule 52: Tooling - an escape written through a tool may arrive as the character it names
- **When:** text written through a tool parameter contains `\uXXXX` meant to stay an escape (invisible characters, confusables, test data for Unicode handling).
- **Then:** write the backslash doubled inside Python source, or generate the character with `chr()`; after writing, scan the file for categories `Cf`/`Zs`/confusable scripts and convert any literal one to an escape.
- **Verified by:** `python3 -c "import unicodedata,sys;t=open(sys.argv[1],encoding='utf-8').read();print([ascii(c) for c in t if ord(c)>127 and unicodedata.category(c) in ('Cf','Zs')])" FILE` prints `[]` for every file written in the round.
- **Added:** 2026-10-09. Root cause: zero-width spaces and a Cyrillic letter landed literally in tests, a probe, a gallery comment and two ledger notes — the exact disguises the round was fixing. Found by the scan, converted before delivery.

### Rule 120: Waiting on a background process - never match its name with a command whose own text contains it
- **When:** a shell loop waits for a background job with `pgrep -f "<pattern>"`.
- **Then:** use `ps -eo pid,args | grep "[p]attern"` (the bracket keeps grep and the loop's own command line from matching), or wait on a PID or an output file's completion marker. The same holds for `pkill -f`: kill by PID taken from the bracket form, never by a pattern the killing command itself contains.
- **Verified by:** the 3c/3d harness waits in this round use the bracket form.
- **Added:** 2026-10-16. Root cause: `while pgrep -f "libs._tools verify"` matched the waiting shell itself, so two waits ran to their timeout after the harness had finished. Recurred the same day in 4a as `pkill -f "rev4a/bin/python"`, which killed its own shell; the rule now names `pkill`. Recurred in 4b in a third form: the bracket check ran in the same command line that launched the job, so it matched the launcher's own text; check in a separate command, and print only the PID and elapsed time. Recurred in 6a/6c as `pkill -f "pytest scikitplot/annoy/tests"` in a command line that also held the pattern: it killed its own shell before the packaging that followed it ran.

### Rule 121: Tooling - the escape conversion of Rule 52 also happens in shell heredocs, and to visible characters
- **When:** any tool input (Write, Edit, or a Bash command, including a quoted `<<'EOF'` heredoc and a Python raw string inside it) carries `\uXXXX` meant to stay an escape in source.
- **Then:** after every write of source, scan the changed files for any character above U+007F, not only `Cf`/`Zs`, and convert each to an escape with a Python step that builds the backslash itself (`"\\u%04x" % ord(c)`). Do this before running gates; the gate's invisible-character scan does not see visible punctuation.
- **Verified by:** `python -c "import sys;t=open(sys.argv[1],encoding='utf-8').read();print(sum(ord(c)>127 for c in t))" FILE` prints `0` for every source file written in the increment (4b: `_structure.py`, `test__structure.py`).
- **Added:** 2026-10-16. Root cause: twice in 4b, `\u2022\u2023\u2043` written inside a raw string in a quoted heredoc arrived as the bullet characters themselves. They are `Po`, so the invisible-character gate passed; ruff's ambiguous-character rule caught the second only because `\u2043` resembles a hyphen.

### Rule 128: Tooling - black given a folder skips what its configuration excludes; give it the files
- **When:** formatting or checking new files with black, in particular tests.
- **Then:** pass each file by name (`for f in ...; do black -q "$f"; done`), as the gates do; a folder argument applies the configuration's `exclude` (this repository's excludes test folders), so `black --check <folder>` passes while every file in it would be reformatted.
- **Verified by:** the gates' black step (`gates.sh`, one file at a time) prints no "NEW black diff" for the increment's files.
- **Added:** 2026-10-17. Root cause: in 5a, `black -q scikitplot/agent` reported "10 files left unchanged" and skipped the four test modules; the gates then reported all four as new black diffs.

### Rule 130: Tooling - an editable meson build rebuilds on import; the venv's `bin` is on `PATH`
- **When:** running anything that imports the package from an editable (meson-python) install: tests, benchmarks, a one-off script.
- **Then:** put the venv's `bin` first on `PATH` (`PATH=<venv>/bin:$PATH <venv>/bin/python ...`); the import runs `ninja`, which calls `cython` and `meson` by name, not through the interpreter that imported.
- **Verified by:** every editable run in 6a and 6c carries the `PATH` prefix; an import after a `.pyx` change rebuilds and the tests see the new code.
- **Added:** 2026-10-17. Root cause: in 6a, calling the venv's `python` directly after editing `kiss_random.pyx` failed the import with "cython: not found": the rebuild ran with the system `PATH`.
