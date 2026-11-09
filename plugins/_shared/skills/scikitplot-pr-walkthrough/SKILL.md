---
name: scikitplot-pr-walkthrough
description: "Explain a pull request in the order its changes cause each other: the change the author set out to make first, then each change it forced, then cleanup, then tests, CI and docs. Use when asked to walk through, explain, read or understand a pull request, or before reviewing a large diff, instead of reading the files top to bottom as GitHub lists them."
license: BSD-3-Clause
---

# Pull request walkthrough, in causal order

A diff lists files alphabetically. A reader who follows that order meets
consequences before their cause and spends attention on lines that only
exist because of a change further down. This skill reorders the pull request
by cause: one root change, then the chain of changes it made necessary.

The walkthrough explains; it does not judge whether the pull request should
be merged, and it writes nothing to GitHub.

## Steps

### 1. Fetch the pull request without changing the user's checkout

Ask before switching branches: the user may have uncommitted work. When they
agree, check the pull request out; otherwise read it from the API only.

```sh
PR=123
git status --short
gh pr view "$PR" --json title,body,url,state,author,headRefName,baseRefName,labels,files,commits
gh pr diff "$PR"
gh pr view "$PR" --json comments,reviews
gh api "repos/{owner}/{repo}/pulls/$PR/comments"
```

Read the linked issues (`Fixes #N`, `Closes #N`) and their discussion: they
state the intent the code is meant to serve.

### 2. Read past the diff

For every changed function, read the whole function, its callers and what it
calls; for a changed class, its bases and subclasses; for configuration, the
whole section. A diff shows what moved, not what it touches.

### 3. Sort every change into one place

- **Level 0, the intent.** The changes that do what the pull request says it
  does. There is usually one; if there are two unrelated intents, say so,
  since that is a finding in itself.
- **Level 1 to N, the ripple.** Each level holds what the level before
  required: a new parameter at level 0 forces its callers at level 1, whose
  new behaviour forces a fixture at level 2, and so on until nothing is left.
- **Cleanup.** Code the change made dead or simpler.
- **Tests, CI and docs.** Every change whose only reason is one of the above.

Group by concern, not by file: three files that carry one change are one
item. Files a tool regenerates (indexes, generated skills, lock files) are
consequences; name the tool and the source change that produced them, and do
not walk through their lines.

In scikit-plots, also note for the root change: the area that owns it under
`maintenances/`, any `REVIEW.json` finding it closes, the engineering rules it
cites, its release-note fragment under
`docs/source/whats_new/upcoming_changes/`, and whether it changes what is
published to `scikit-plots/skills`.

### 4. Write it

```markdown
# PR #<number>: <title>

Author, branch into base, state, date read.

## Why it exists
One or two paragraphs from the description, the linked issues and the review
discussion.

## Level 0: <the intent in one line>
What changed and how, with links to files and line numbers.

## Level 1: <what level 0 forced>
...

## Cleanup / Tests, CI and docs
...

## Questions for the reviewer
Only real ones: a level whose cause is unclear, a change no level explains,
behaviour the tests do not reach.
```

Draw a small dependency diagram when there are three levels or more.

## Rules

- Give the walkthrough in the reply. Save it to a file only when the user
  asks, at the path they choose (in scikit-plots, under the owning area's
  `maintenances/<area>/_maintenance/`).
- Every statement about the code names a file and a line.
- A change that fits no level is reported, not forced into one.
- Describe a repeated pattern once, then list where it applies.
- Leave the user on whichever branch they agreed to; say which one.
- No praise, no filler: each paragraph tells the reader something to check.
