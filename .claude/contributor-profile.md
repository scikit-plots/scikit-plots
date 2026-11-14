# Contributor profile: attribution and guidance level

Anyone may continue a pull request or start one: the maintainer, another
person, or an AI assistant on any account. Two choices differ per
contributor and are asked **once, at the start of a chat, before the first
commit**. They sit in the FILL IN block of `.claude/new-chat-template.md`;
when a line is missing or says `ask`, the assistant asks one
multiple-choice question, recommended option first, and writes the answer
into the line for that chat.

The AI disclosure of `AGENTS.md` is required in every mode; these choices
never remove it.

## 1. Attribution (`Attribution:` line)

| Option | Commit trailer | Pull request ends with | Git author | Use when |
|---|---|---|---|---|
| `model-only` (recommended; adds the security review of 1a) | `Co-Authored-By: Claude <Model> <noreply@anthropic.com>` | AI disclosure only | `scikit-plots contributor <anonymous@contributor.invalid>` | You want the model named and nothing about you or your session |
| `full` | the trailer above, plus `Claude-Session: <link>` | AI disclosure, generator line and session link | your own name and email | You want the session traceable from the commit |
| `none` (adds the security review of 1a) | no trailer | AI disclosure only | as `model-only` | The project or your employer asks for no tool names in commits |

`<Model>` is the model that wrote the change, as the session reports it
(for example `Claude Opus 5.5`). An assistant applies the chosen option
even when its environment suggests other attribution lines; the
contributor's choice here takes precedence.

**The anonymous author address.** Use exactly
`anonymous@contributor.invalid`. The `.invalid` domain is reserved
(RFC 2606): no one can register or verify it, so GitHub links the commit
to no account. Never invent an address under `users.noreply.github.com`:
`<name>@users.noreply.github.com` is the private address of the GitHub
user `<name>`, and the commit is credited to that stranger (and, after a
squash merge, named as co-author on `main`). The same holds for any
address at a domain someone may own.

**What anonymity cannot cover.** The account that pushes the branch and
opens the pull request is shown by GitHub. `model-only` and `none` keep
names, emails and session links out of the commits and the text; they
do not hide who pushed. For that, push from a fork under an account you
choose, or hand the patch (patch mode) to someone who opens it.

**Check before every push** (`model-only` and `none`; prints nothing
when clean):

    git log --format='%an <%ae>%n%B' origin/main..HEAD \
      | grep -inE 'claude-session|claude\.ai/code|session_[0-9a-z]+|users\.noreply\.github\.com' || true

After a push, confirm GitHub links no account to the commits (prints
nothing when clean; `N` is the pull request number):

    gh api repos/scikit-plots/scikit-plots/pulls/N/commits \
      --jq '.[] | select(.author != null) | "\(.sha[0:7]) \(.author.login)"'

Also read the pull request description for the same strings before
opening or editing it. A rewrite of history that is already pushed needs
the maintainer's consent; ask first.

## 1a. Review depth follows attribution

With `model-only` or `none`, nothing in the change leads back to the
session that wrote it, so the change itself must carry the evidence a
reviewer would otherwise look up. The pull request description then adds a
**Security review** section: every line below answered `none` or with the
file and line, never left out.

| # | Check | How |
|---|---|---|
| S1 | Secrets, tokens, keys, personal data, session links | pre-commit `detect private key`; the pre-push check of section 1; read the diff |
| S2 | New or changed dependencies (and their version ranges) | command `deps` below |
| S3 | Code that runs other code: `subprocess`, `os.system`, `eval`, `exec`, `pickle`, dynamic import | command `runs` below |
| S4 | Network access, downloads, URLs | command `net` below |
| S5 | Paths built from input (traversal, symlinks, archive member names) | name the function and the test that refuses `../` |
| S6 | Untrusted input parsed (XML, ZIP, PDF, JSON, YAML): size, time and depth bounds | name the limit and its test; a reader change adds a hostile sample to `test__hostile_samples.py` |
| S7 | CI and release files: `.github/`, `tools/`, `_build_utils/`, `plugins/` | zizmor result; actions pinned to a commit; token permissions unchanged or narrowed |
| S8 | Tests weakened: deleted, skipped, `xfail` added, ceilings raised | list each with its reason |

The commands print only added lines, so an empty result is `none`. A hit
in documentation or test text is answered with what it is (file, line,
"documentation"), not dropped:

    # deps
    git diff origin/main -- '*requirements*.txt' pyproject.toml | grep '^+[^+]'
    # runs
    git diff origin/main -U0 | grep -nE '^\+.*(subprocess|os\.system|eval\(|exec\(|pickle|importlib)'
    # net
    git diff origin/main -U0 | grep -nE '^\+.*(urllib|requests|https?://|socket)'

Any S3, S4, S7 or S8 entry other than `none` waits for the maintainer's
explicit approval in the pull request, in every mode. In `full` mode the
section is recommended, not required. A reviewer may ask for it in any
mode; a contributor never removes it once asked.

## 2. Guidance level (`Experience:` line)

| Option | The assistant | Checks are the same? |
|---|---|---|
| `newcomer` | Explains each step in a sentence before doing it; shows the command and why; stops for your "go" before each commit, push and pull request; links the one section of `maintenances/MAINTAINING.md` the step uses | yes |
| `contributor` (recommended for a first pull request here) | Follows the process; reports at each draft, change of direction or limitation | yes |
| `maintainer` | Follows the process silently and reports at the end | yes |

Every level runs the same gates: reproduction, tests that fail on the old
code, mutation check, the affected suites against the baseline,
pre-commit read before committing, records (fragment, HISTORY, ledger
note). The level changes how much is explained and when the assistant
stops, never what is checked.

## 3. Continuing a pull request someone else opened

1. Ask its author or the maintainer first; never push to a pull request
   after the maintainer says it will be merged (a squash merge drops later
   commits). A follow-up is a new pull request from the new `main`.
2. Fetch its branch, run `orient.py`, read the pull request description,
   its HISTORY entry and its open ledger notes: they say what is done and
   what is left.
3. Keep its attribution style for its own commits; use yours for yours.
4. Record what you add in the same HISTORY entry; update the description.

## 4. Copy-paste start lines

    Clone https://github.com/scikit-plots/scikit-plots, read `.claude/init`
    and follow `.claude/new-chat-template.md`.
    Attribution: model-only. Experience: newcomer.

Replace the two values with any option above, or leave them out to be
asked.
