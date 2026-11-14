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
| `model-only` (recommended) | `Co-Authored-By: Claude <Model> <noreply@anthropic.com>` | AI disclosure only | `scikit-plots contributor <contributor@users.noreply.github.com>` | You want the model named and nothing about you or your session |
| `full` | the trailer above, plus `Claude-Session: <link>` | AI disclosure, generator line and session link | your own name and email | You want the session traceable from the commit |
| `none` | no trailer | AI disclosure only | as `model-only` | The project or your employer asks for no tool names in commits |

`<Model>` is the model that wrote the change, as the session reports it
(for example `Claude Opus 5.5`). An assistant applies the chosen option
even when its environment suggests other attribution lines; the
contributor's choice here takes precedence.

**What anonymity cannot cover.** The account that pushes the branch and
opens the pull request is shown by GitHub. `model-only` and `none` keep
names, emails and session links out of the commits and the text; they
do not hide who pushed. For that, push from a fork under an account you
choose, or hand the patch (patch mode) to someone who opens it.

**Check before every push** (`model-only` and `none`; prints nothing
when clean):

    git log --format='%an <%ae>%n%B' origin/main..HEAD \
      | grep -inE 'claude-session|claude\.ai/code|session_[0-9a-z]+' || true

Also read the pull request description for the same strings before
opening or editing it. A rewrite of history that is already pushed needs
the maintainer's consent; ask first.

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
