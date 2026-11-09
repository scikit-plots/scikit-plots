---
name: scikitplot-pr-impact-brief
description: "Gather sourced evidence on whether a pull request is worth carrying (what it changes for users, how many it reaches, how bad the problem is, whether users can already do it, and what it costs to keep) and end with the open questions, never a verdict. Use when asked whether a pull request is worth merging or keeping, to triage open pull requests by value, or for an impact brief on a pull request number or link."
license: BSD-3-Clause
---

# Pull request impact brief

The question is not whether the code is right (that is a review) but whether
the change is worth having. The brief collects evidence for that decision and
stops before it: the maintainer decides.

## Rules that make the brief useful

1. **Read only.** `gh` is used to view: `gh pr view`, `gh pr diff`,
   `gh issue view`, `gh search`, and `gh api` with the default GET. No
   comment, label, review, close or edit.
2. **No verdict.** Not in the brief and not when asked "would you merge it?":
   answer with what is still unresolved.
3. **A source for every fact:** a link, or a file and line. The only
   exception is the reach estimate, which is labelled as an estimate.
4. **What cannot be sourced is listed** under "Unsupported claims" instead of
   being written as fact.
5. **The author's description is a claim, not evidence.** Check each claim
   the case rests on against the code: that the broken path is reachable,
   that the behaviour is as described on the supported versions, that a
   check upstream does not already prevent it.
6. **"Not found" and "could not determine" are answers.** Never fill a gap
   with plausible prose.
7. **One pull request at a time,** chosen by the user, or picked from the
   open list when the user asks for the next one.

## Steps

### 1. Context

```sh
PR=123
gh pr view "$PR" --json title,body,url,author,createdAt,updatedAt,labels,files,additions,deletions,comments,reviews
gh pr diff "$PR"
```

Read every linked issue with its labels and discussion. A label that records
an open decision (in scikit-plots, "Needs Triage" or another "Needs ..."
label) outranks everything else: the project has said the question is open,
and contributing against such an issue is out of process.

### 2. The source

Two jobs no tool does for you:

- **The counterfactual.** Can users get the result today: a sibling function
  that takes this option, an existing composition, a documented workaround?
  Cite where.
- **The central claim.** Find the code the pull request says is wrong and
  confirm it behaves that way on the supported versions.

### 3. Prior art

Search for earlier attempts and the reasons they were closed:

```sh
QUERY="keyword from the title"
gh search prs "$QUERY" --repo "$(gh repo view --json nameWithOwner --jq .nameWithOwner)" --state closed --limit 20
gh search issues "$QUERY" --repo "$(gh repo view --json nameWithOwner --jq .nameWithOwner)" --limit 20
```

## The brief

```markdown
# Impact brief: PR #<number>, <title>

## What it changes
The public behaviour before and after, in two or three sentences, sourced.

## Reach (estimate, not sourced)
Few, some or many users, and the reasoning.

## Severity
What happens to an affected user without the change: wrong result, error,
inconvenience. Sourced.

## Can users do this today?
The counterfactual from step 2, with sources, or "not found".

## Cost to keep
New public surface, new dependency or option, tests and docs to maintain,
interaction with planned work.

## Prior art
Earlier pull requests and issues, and why they ended as they did.

## What the decision hinges on
The two to four questions whose answers would settle it.

## Unsupported claims
Every statement above that could not be sourced.
```

## Conversation

Give the brief in the reply; save it to a file only when the user asks, at
the path they choose. Stay on the same pull request for follow-up questions
until the user moves on. A follow-up answer follows the same rules: sourced,
and no verdict.
