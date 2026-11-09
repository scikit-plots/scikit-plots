---
name: scikitplot-pr-review-flow
description: "Run the scikit-plots pull-request review process end to end: orient from the tree, triage, read the change in causal order, verify it with the owning area's checks, record findings with evidence, then post an approved draft review and follow up after merge. Use when asked to review, triage or follow up a scikit-plots pull request, or to decide which pull-request or CI skill to use."
license: BSD-3-Clause
---

# Pull-request review flow

The process for a contributor or maintainer reviewing a scikit-plots pull
request. Each stage names the skill that does it; this skill decides the
order, the gates between stages, and where the result is kept. State lives in
the tree (records, ledger, rules) or on GitHub, never only in the chat.

## Before you start

**First review here?** Read `AGENTS.md` (the "Needs ..." labels),
`maintenances/README.md` (where each area's records live) and
`references/gates-by-path.md` (which checks a change needs). Nothing else is
needed to begin.

**Choose the depth** with the user, by what a wrong merge would cost:

| Depth | When | Stages |
|---|---|---|
| quick | one area, no public API or dependency change, a maintainer's own change, or a time-boxed pass | 1, 3 (the `pr_paths.py` table only), 4, 5 |
| full | anything else: outside contributions, public API, dependencies, CI, published plugin | 1 to 8 |

A quick pass that finds a public API change, a new dependency or a change to
`plugins/scikitplot/` becomes a full one; say so.

**Resume, don't restart.** Each stage leaves a trace another person or a
later session can see; begin at the first stage without one:

| Stage | Its trace |
|---|---|
| 2 | a brief in the user's briefs folder, named `<number>-...` |
| 5 | the findings list given to the user, or ledger notes under `upcoming_changes/` |
| 7 | your pending review on the pull request (`post_pending_review.py --dry-run` reports it and which comments it already holds) |
| 8 | a forecast file named by the run's date |

## Stages

### 1. Orient

```sh
python -B maintenances/_repo/_maintenance/tools/orient.py
PR=123
gh pr view "$PR" --json title,author,labels,state,isDraft,files,headRefName,baseRefName
```

Stop and report, without reviewing, when the pull request or its linked
issue has a "Needs Triage" or another "Needs ..." label (`AGENTS.md`): the
project has not decided yet.

### 2. Triage

- A change whose worth is not settled (a new option, a new dependency, an
  outside contribution with no linked decision): `scikitplot-pr-impact-brief`
  first. The maintainer decides; continue only when told to.
- Otherwise go on.

### 3. Understand

`scikitplot-pr-walkthrough`; its `pr_paths.py` table (owning area and kind of
every changed file) is the first step, and all a quick pass needs. From its Level 0, name the owning area under
`maintenances/`, its skill (`plugins/INDEX.md`), its open findings
(`REVIEW.json`), and the engineering rules the change touches.

### 4. Verify

Check out the branch only with the user's consent. Then, for the paths the
change touches:

- the owning area's tests, and the repository gates of section 5 of
  `.claude/new-chat-template.md` that apply: `references/gates-by-path.md`
  maps each kind of changed path to its checks;
- for a bug fix, the new test fails on the base branch and passes on the
  pull request;
- each claim in the description checked against the code, not repeated.

### 5. Record findings

One line per finding: file and line, what is wrong, the evidence (command
and output, or a reproduction), its severity, and the rule it breaks.
Separate "must change" from "question". A finding that outlives this pull
request becomes a ledger note under `upcoming_changes/` (its README and
template), not a chat remark.

### 6. Decide the action

| Outcome | Next |
|---|---|
| comments for the author | stage 7 |
| a fix the maintainer will make | choose the delivery mode (section 1 of the template) |
| a lesson that applies beyond this change | a numbered engineering rule (`scikitplot-engineering-rules`) |
| ready to merge | say so with the evidence; merging is the maintainer's act |

### 7. Communicate

Agree the exact comments with the user, then `scikitplot-pr-pending-review`.
Before posting, read the comments and reviews already on the pull request and
drop any comment that repeats one, so a second run posts nothing new.

### 8. After merge

When the change touches CI, dependencies or what a scheduled workflow runs:
`scikitplot-ci-failure-forecast` for the next scheduled run, scored after it
runs. When the merge publishes `plugins/scikitplot/`, read the next directory
scan (Rules 170, 171).

## Gates (every stage)

- Read before write: nothing is posted, pushed, labelled or merged without
  the user's approval of that exact action.
- Evidence for every statement about the change.
- One pull request at a time; finish or park it (a ledger note) before the
  next.
- Repeating a stage changes nothing that is already done: no second comment,
  note, finding or rule for the same point.
- Stop and say so when a stage needs a decision that is the maintainer's.
