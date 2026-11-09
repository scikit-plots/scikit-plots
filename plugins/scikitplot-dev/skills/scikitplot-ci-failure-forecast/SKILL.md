---
name: scikitplot-ci-failure-forecast
description: "Forecast which jobs of a scheduled CI run (a nightly or weekly workflow) will fail before it runs, from the last run's failures and what changed since, then diagnose each predicted failure and draft an issue for it; also score the previous forecast against what actually ran. Use when asked what will fail tonight, to forecast or predict the nightly or scheduled CI, or to check how a past forecast did."
license: BSD-3-Clause
---

# Scheduled CI failure forecast

## Programs (ask before each run)

This skill ships the programs below. Before each run, tell the user which one, what it reads, what it changes, and the exact command, and wait for their yes. Never start one unannounced or in the background. Each is optional: the steps further down do the same work by hand. Run them from the repository root; they need Python 3.8 or newer and, where noted, a logged-in `gh`.

| Program | Reads | Changes |
|---|---|---|
| `scripts/ci_runs.py` | scheduled runs and their attempt-1 jobs (`gh`) | the `--out` file, when given |

`baseline` records the newest scheduled run of a workflow (step 1); `score` compares a forecast file, `{"workflow": ..., "baseline_run": ..., "jobs": {"<job>": "<outcome>"}}`, with attempt 1 of the run it forecast and counts hits, misses and false alarms.

```sh
REPO="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
python3 plugins/scikitplot-dev/skills/scikitplot-ci-failure-forecast/scripts/ci_runs.py --repo "$REPO" baseline --workflow nightly.yml
```

A scheduled run tests the default branch against the newest dependencies, so
it breaks for two reasons: a failure that nobody fixed since the last run, or
something new since then (a merged change, a dependency release). Both are
visible before the run. A forecast made beforehand, and scored afterwards,
turns the nightly from a surprise into a checklist.

The forecast reads only; it never fixes, pushes, reruns or posts. Drafted
issues are left for the user to file.

## Steps

### 1. Find the scheduled workflows and their baseline

```sh
REPO="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
grep -l "schedule:" .github/workflows/*
WORKFLOW=nightly.yml
gh run list --repo "$REPO" --workflow "$WORKFLOW" --event schedule --limit 5 --json databaseId,headSha,conclusion,createdAt
RUN="$(gh run list --repo "$REPO" --workflow "$WORKFLOW" --event schedule --limit 1 --json databaseId --jq '.[0].databaseId')"
SHA="$(gh run view "$RUN" --repo "$REPO" --json headSha --jq .headSha)"
gh api --paginate "repos/$REPO/actions/runs/$RUN/attempts/1/jobs" --jq '.jobs[] | {name, conclusion}'
gh run view "$RUN" --repo "$REPO" --attempt 1 --log-failed
```

The baseline is attempt 1: a manual rerun can turn a run green and hide what
failed. Note jobs marked `continue-on-error`: they can fail while the run
reports success, and they are often where breakage appears first, so forecast
them too.

### 2. What changed since the baseline

- **This repository:** every commit merged into the default branch since the
  baseline commit.

  ```sh
  git fetch origin
  git log --oneline "$SHA..origin/HEAD"
  ```

- **Dependencies:** for each dependency the workflow installs without an
  exact version (nightly builds, pre-releases, latest releases), the releases
  published since the baseline run's date, from the package's own release
  page or changelog.

### 3. Forecast each job

Give every job of the workflow one of four outcomes, each with its evidence:

| Outcome | When |
|---|---|
| carries over | it failed at the baseline and no change since addresses its cause |
| recovers | it failed at the baseline and a merged change or a new release fixes the cause (name it) |
| breaks | it passed at the baseline and a change since plausibly breaks it (name the change and the test it reaches) |
| passes | none of the above |

A job only "breaks" with a named cause; a vague worry is not a forecast. When
one test settles a verdict and can be run locally on the changed code, run it
and say so.

### 4. Diagnose and draft

For each job forecast to fail: the failing test or step, the log lines that
show the error, the change or release responsible, and the smallest fix that
would address the cause. Search for an existing issue first and link it;
otherwise draft one (title, body with the evidence) without posting it.

```sh
QUERY="failing test name"
gh search issues "$QUERY" --repo "$REPO" --state open --limit 10
```

### 5. Record and stop

Give the forecast in the reply, as a table of jobs with outcome and evidence,
then the diagnoses and drafted issues. Save it to a file only when the user
asks, at the path they choose, named by the date of the run it forecasts,
with the baseline run id and commit.

## Scoring a past forecast

When the forecast run has happened, compare each job's forecast with its
attempt-1 conclusion, and report hits, misses and false alarms, with one line
per miss on what the forecast could have seen. Keep the score with the
forecast it scores.

## Rules

- Evidence for every outcome other than "passes": a run, a log line, a commit
  or a release.
- Attempt 1 is the truth; later attempts are not.
- Read only: no rerun, no push, no issue or comment posted.
