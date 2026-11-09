---
name: scikitplot-pr-pending-review
description: "Place comments the user has agreed on onto specific lines of a GitHub pull request as a pending (draft) review in the user's name, for them to edit and submit in the browser. Use when asked to put comments on lines of a pull request, post review comments as a draft, or add inline explanations to a diff. It posts only text the user approved, checks every line first, and never submits the review."
license: BSD-3-Clause
---

# Pending review on a pull request

This skill moves comments that already exist in the conversation onto the
lines of a pull request, as a review that stays pending until the user
submits it in the browser. It does not decide what to say, and it never
submits: on GitHub, submitting is the user's act.

It writes to GitHub with the user's own `gh` login, so it runs only after the
user has approved the exact comments in the conversation.

## Steps

### 1. Agree the comments

List each comment with its file, line (or range) and text, and get the user's
approval for that list. A comment on the new side of the diff uses the new
file's line number; on a deleted line, the old file's line number with side
`LEFT`.

### 2. Check every line against the diff

Only lines inside a diff hunk (added, deleted or context) take a comment, and
a range must stay inside one hunk.

```sh
PR=123
REPO="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
gh api --paginate "repos/$REPO/pulls/$PR/files" --jq '.[] | {filename, patch}'
```

Each hunk header `@@ -a,b +c,d @@` covers old lines `a` to `a+b-1` and new
lines `c` to `c+d-1`. List every comment with "ok" or the reason it is not
placeable. **If any comment is not placeable, post nothing:** fix it with the
user first.

### 3. Write the review file

Write `review.json` in a scratch folder with the file tool (never in the
repository), with the pull request's head commit:

```sh
gh pr view "$PR" --json headRefOid --jq .headRefOid
```

```json
{
  "commit_id": "<head commit from the command above>",
  "comments": [
    {"path": "src/a.py", "line": 42, "side": "RIGHT", "body": "One line."},
    {"path": "src/a.py", "start_line": 40, "line": 42, "side": "RIGHT", "body": "A range."},
    {"path": "src/b.py", "line": 7, "side": "LEFT", "body": "On a deleted line."}
  ]
}
```

There is no `event` key and no review `body`: without an event the review
stays pending, and a pending review's body is not shown in the browser's
submit dialog. Put a summary in the pull request description or a separate
comment instead.

### 4. Post, once

```sh
gh api --method POST "repos/$REPO/pulls/$PR/reviews" --input review.json --jq '{id, state}'
```

The answer must show `"state": "PENDING"`. Then tell the user to open the
pull request's "Files changed" tab, edit if needed, and submit.

If GitHub answers that the user already has a pending review on this pull
request (only one is allowed per user), add each comment to it instead:

```sh
REVIEW_ID="$(gh api graphql -F owner="${REPO%/*}" -F name="${REPO#*/}" -F number="$PR" -f query='
  query($owner: String!, $name: String!, $number: Int!) {
    repository(owner: $owner, name: $name) {
      pullRequest(number: $number) {
        reviews(states: PENDING, first: 1) { nodes { id } }
      }
    }
  }' --jq '.data.repository.pullRequest.reviews.nodes[0].id')"
echo "$REVIEW_ID"
gh api graphql -F review="$REVIEW_ID" -F path="src/a.py" -F line=42 -F side=RIGHT -F body="One line." -f query='
  mutation($review: ID!, $path: String!, $line: Int!, $side: DiffSide!, $body: String!) {
    addPullRequestReviewThread(input: {pullRequestReviewId: $review, path: $path,
      line: $line, side: $side, body: $body}) { thread { id } }
  }'
```

For a range, add `startLine` (and `startSide`) to the mutation the same way.

## Rules

- Post only what the user approved, word for word.
- All or nothing: one invalid line stops the whole post.
- Never pass `event`, never call the submit endpoint, never approve or
  request changes.
- Post once; if a post fails partway, list what was posted before retrying,
  so nothing is duplicated.
- While a review is pending, GitHub reports `line: null` for its comments;
  they show on the right lines once submitted.
