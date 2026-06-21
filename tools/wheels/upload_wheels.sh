#!/bin/bash

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted from NumPy's wheel upload helper and extended with
# exponential-backoff retry logic (originally in upload_with_retry.sh,
# now merged here so a single `source` is all callers need).
# https://github.com/numpy/numpy/blob/main/tools/wheels/upload_wheels.sh
# https://docs.anaconda.com/anacondaorg/user-guide/packages/standard-python-packages/
#
# WHY RETRY EXISTS
# ────────────────
# Anaconda.org returns HTTP 503 Service Unavailable when its API is under
# high load.  The anaconda CLI propagates that as a Python ServerError and
# exits non-zero, aborting the entire upload even though every artifact is
# perfectly valid.  The failure is 100% transient (the server recovers in
# seconds to minutes), so the correct fix is an exponential-backoff retry
# loop applied per-artifact, NOT a rebuild.
#
# RETRY DESIGN CONSTRAINTS
# ─────────────────────────
# • Retry ONLY on 503 / Service Unavailable / high-load patterns.
#   Auth (401/403), bad-request (400/422), and local I/O errors are permanent
#   and must surface immediately — they must never be masked by retrying.
# • Per-artifact retry budget: one slow file cannot starve the rest.
# • Each artifact is uploaded independently; a 503 on one never aborts others.
# • find + NUL delimiters for safe enumeration (handles spaces in filenames,
#   no shell glob-expansion surprises, deterministic sort order).
#
# ARTIFACT DISCOVERY (in priority order)
# ───────────────────────────────────────
# 1. $ARTIFACTS_PATH env var (set by CI job)       → search that directory
# 2. Auto-detect: ./dist contains *.gz             → sdist path
# 3. Auto-detect: ./wheelhouse contains *.whl      → wheel path
#
# TUNEABLE VIA ENVIRONMENT (optional — sane defaults apply)
# ──────────────────────────────────────────────────────────
#   UPLOAD_MAX_ATTEMPTS  = 5    total attempts per artifact before giving up
#   UPLOAD_RETRY_DELAY   = 30   initial back-off delay in seconds
#   UPLOAD_MAX_DELAY     = 300  back-off ceiling in seconds (5 min)
#   ANACONDA_LABEL       = ""   (optional) Anaconda.org label/channel to upload
#                                to, e.g. "main-pyodide-0.29.4". Passed as
#                                `anaconda upload --label <value>` when set.
#                                Unset or empty (the default) ⇒ no --label
#                                flag at all ⇒ anaconda-client's own default
#                                label ("main") is used — IDENTICAL to this
#                                script's behavior before ANACONDA_LABEL
#                                existed. Callers that never set this variable
#                                (any other CI workflow sourcing this file)
#                                are completely unaffected — this is an
#                                additive, opt-in parameter, not a new
#                                required input.
#
# TIMING PROFILE (defaults)
# ─────────────────────────
#   attempt 1  → fail → wait  30 s
#   attempt 2  → fail → wait  60 s
#   attempt 3  → fail → wait 120 s
#   attempt 4  → fail → wait 240 s
#   attempt 5  → fail → give up  (worst-case ≈ 7.5 min per artifact)

set -Eeuo pipefail

# Initialise trigger flags so `set -u` never fires on unbound variables.
# The POSIX no-op `:` assigns a default without overriding a value already
# exported by the caller (CI matrix env, Travis, etc.).
: "${IS_PUSH:="false"}"
: "${IS_SCHEDULE_DISPATCH:="false"}"

# ANACONDA_LABEL: optional, defaults to empty. Empty ⇒ no --label flag is
# passed to `anaconda upload` ⇒ anaconda-client's own default ("main") is
# used, exactly as before this variable existed. A caller (e.g. a CI matrix
# row) opts in by exporting ANACONDA_LABEL before calling upload_wheels;
# anything that doesn't export it gets the unchanged, pre-existing behavior.
: "${ANACONDA_LABEL:=}"

# ── set_travis_vars ──────────────────────────────────────────────────────────
# Translates Travis-CI event variables into the IS_PUSH / IS_SCHEDULE_DISPATCH
# flags consumed by set_upload_vars.
set_travis_vars() {
    echo "TRAVIS_EVENT_TYPE is $TRAVIS_EVENT_TYPE"
    echo "TRAVIS_TAG is $TRAVIS_TAG"
    if [[ "$TRAVIS_EVENT_TYPE" == "push" && "$TRAVIS_TAG" == v* ]]; then
        IS_PUSH="true"
    else
        IS_PUSH="false"
    fi
    if [[ "$TRAVIS_EVENT_TYPE" == "cron" ]]; then
        IS_SCHEDULE_DISPATCH="true"
    else
        IS_SCHEDULE_DISPATCH="false"
    fi
}

# ── set_upload_vars ──────────────────────────────────────────────────────────
# Exports ANACONDA_UPLOAD, USERNAME, and TOKEN based on event type.
# Must be called before upload_wheels.
set_upload_vars() {
    echo "IS_PUSH is $IS_PUSH"
    echo "IS_SCHEDULE_DISPATCH is $IS_SCHEDULE_DISPATCH"
    if [[ "$IS_PUSH" == "true" ]]; then
        echo "EVENT: push and tag event"
        export ANACONDA_UPLOAD="true"
        export USERNAME="scikit-plots-wheels-staging"
        export TOKEN="$SKPLT_STAGING_UPLOAD_TOKEN"
    elif [[ "$IS_SCHEDULE_DISPATCH" == "true" ]]; then
        echo "EVENT: scheduled or dispatched event"
        export ANACONDA_UPLOAD="true"
        export USERNAME="scikit-plots-wheels-staging-nightly"
        export TOKEN="$SKPLT_STAGING_UPLOAD_TOKEN_NIGHTLY"
    else
        echo "EVENT: non-dispatch event"
        export ANACONDA_UPLOAD="false"
    fi
}

# ── anaconda_upload_with_retry ───────────────────────────────────────────────
# Upload a single artifact with exponential-backoff retry on HTTP 503 only.
#
# All other failure codes (401, 403, 400/422, local I/O) are surfaced
# immediately: retrying them can never succeed and would only delay the error.
#
# Parameters
# ----------
# $1 : path to the artifact (.whl, .tar.gz, or .gz)
#
# Environment (exported by set_upload_vars)
# -----------
# TOKEN               : Anaconda.org API token
# USERNAME            : Anaconda.org target user or organisation
# ANACONDA_LABEL      : (optional) label/channel to upload to. Empty/unset
#                        ⇒ no --label flag ⇒ anaconda-client's default
#                        ("main"). Non-empty ⇒ `--label "$ANACONDA_LABEL"`.
#
# Tuneable (optional env vars with sane defaults)
# --------
# UPLOAD_MAX_ATTEMPTS : total attempts per artifact  (default: 5)
# UPLOAD_RETRY_DELAY  : initial back-off in seconds  (default: 30)
# UPLOAD_MAX_DELAY    : back-off ceiling in seconds  (default: 300)
#
# Returns
# -------
# 0 on success; non-zero on permanent failure or exhausted retry budget.
anaconda_upload_with_retry() {
    local artifact="$1"
    local basename_f; basename_f="$(basename "$artifact")"
    local max_attempts="${UPLOAD_MAX_ATTEMPTS:-5}"
    local delay="${UPLOAD_RETRY_DELAY:-30}"
    local max_delay="${UPLOAD_MAX_DELAY:-300}"
    local attempt=1
    local rc=0
    local out=""
    local next_delay=0  # declared once here; assigned inside the loop below

    # Build the --label arg ONCE per artifact (not per attempt) — the value
    # never changes across retries of the same upload.
    # Empty ANACONDA_LABEL (the default) ⇒ no --label flag ⇒ anaconda-client
    # uploads to its own default label ("main"), matching pre-existing
    # behavior for every caller that doesn't set this variable.
    local -a label_args=()
    local label_display="main (default)"
    if [[ -n "${ANACONDA_LABEL:-}" ]]; then
        label_args=(--label "$ANACONDA_LABEL")
        label_display="$ANACONDA_LABEL"
    fi

    while (( attempt <= max_attempts )); do
        echo "⏫  [${attempt}/${max_attempts}] ${basename_f}  →  label: ${label_display}"

        # Temporarily disable errexit so we can capture both stdout+stderr
        # and the exit code in a single step.  set -e is restored immediately
        # afterwards so the outer errexit contract is never broken.
        set +e
        out=$(anaconda --verbose -q -t "$TOKEN" upload --force \
            "${label_args[@]}" \
            -u "$USERNAME" "$artifact" 2>&1)
        rc=$?
        set -e

        echo "$out"

        # ── Success ──────────────────────────────────────────────────────────
        if (( rc == 0 )); then
            echo "✅  Uploaded: ${basename_f}  →  label: ${label_display}"
            return 0
        fi

        # ── Permanent failure — surface immediately ───────────────────────────
        # Only 503 / Service Unavailable / high-load is transient and worth
        # retrying.  Auth (401/403) and request errors (400/422) will never
        # succeed on retry and must not be silently swallowed.
        if ! echo "$out" | grep -qE "503|Service.Unavailable|high.load"; then
            echo "❌  Non-retryable failure for ${basename_f} (exit ${rc})"
            return $rc
        fi

        # ── Retry budget exhausted ────────────────────────────────────────────
        if (( attempt >= max_attempts )); then
            echo "❌  Giving up after ${max_attempts} attempt(s)" \
                 "(503 persists): ${basename_f}"
            return $rc
        fi

        # ── Back-off then retry ───────────────────────────────────────────────
        # Arithmetic substitution $((…)) always exits 0, so it is safe under
        # set -e.  It also avoids the separate `local` + assignment pattern
        # because next_delay is already declared above.
        next_delay=$(( delay * 2 < max_delay ? delay * 2 : max_delay ))
        echo "⚠️   Anaconda.org 503 — retrying in ${delay}s" \
             "(attempt ${attempt}/${max_attempts}; next back-off ${next_delay}s)"
        sleep "$delay"
        delay=$next_delay

        # Use arithmetic substitution rather than (( attempt++ )).
        # Under set -e, a standalone (( expr )) exits 1 when the expression
        # evaluates to 0.  attempt starts at 1 so (( attempt++ )) would never
        # actually evaluate to 0 here, but $((…)) is unconditionally safe and
        # expresses the intent without relying on that invariant.
        attempt=$(( attempt + 1 ))
    done
}

# ── upload_wheels ────────────────────────────────────────────────────────────
# Install anaconda-client, resolve the artifact directory, then upload each
# file individually via anaconda_upload_with_retry.
#
# A transient 503 on one artifact no longer aborts the rest: all failures are
# collected and a non-zero exit is returned only after every file has been
# attempted.
#
# Environment (set by set_upload_vars and CI job env)
# -----------
# ANACONDA_UPLOAD : must equal "true" to proceed — anything else skips upload
# TOKEN           : Anaconda.org API token         (from set_upload_vars)
# USERNAME        : Anaconda.org target user/org   (from set_upload_vars)
# ARTIFACTS_PATH  : (optional) directory containing artifacts  (set by CI job)
# ANACONDA_LABEL  : (optional) label/channel to upload to. Empty/unset is the
#                    default and preserves the exact pre-existing behavior
#                    (no --label flag). See file header for details. A caller
#                    that wants to upload the SAME artifacts to multiple
#                    labels calls upload_wheels() multiple times, re-exporting
#                    ANACONDA_LABEL to a different value between calls.
# CONDA           : (optional) path to conda installation base (set by runner)
upload_wheels() {
    echo "PWD: ${PWD}"
    ls -lah

    # ── Install anaconda-client ──────────────────────────────────────────────
    # Both conda and pip are tried; || true is intentional because:
    #   • The environment may already have anaconda-client installed.
    #   • conda activate requires shell initialisation that varies by runner.
    #   • pip is a reliable fallback, especially on win-arm64 where
    #     anaconda-client is not available via conda.
    # A missing CLI is caught with a clear "command not found" at upload time.
    if [[ -n "${CONDA:-}" ]]; then
        # Guard against set -u: CONDA may be unset on pip-only runners.
        export PATH="${CONDA}/bin:${PATH}"
    fi
    conda create -n upload -y anaconda-client 2>/dev/null || true
    # shellcheck disable=SC1091
    conda activate upload 2>/dev/null || true
    pip install --no-cache-dir anaconda-client 2>/dev/null || true

    # ── Guard: only upload when explicitly enabled ───────────────────────────
    if [[ "${ANACONDA_UPLOAD:-false}" != "true" ]]; then
        echo "ANACONDA_UPLOAD is not 'true' — skipping upload"
        return 0
    fi

    if [[ -z "${TOKEN:-}" ]]; then
        echo "TOKEN is not set — skipping upload"
        return 0
    fi

    echo "TOKEN found, scanning for artifacts..."

    # ── Artifact directory resolution ─────────────────────────────────────────
    # Priority 1: $ARTIFACTS_PATH from the CI job env (e.g. "wheelhouse",
    #             "dist").  Relative values are resolved against $PWD so that
    #             find receives an absolute path regardless of working directory.
    # Priority 2: Auto-detect ./dist (sdist) or ./wheelhouse (wheels).
    local artifacts_dir=""

    if [[ -n "${ARTIFACTS_PATH:-}" ]]; then
        if [[ "$ARTIFACTS_PATH" == /* ]]; then
            artifacts_dir="$ARTIFACTS_PATH"
        else
            artifacts_dir="${PWD}/${ARTIFACTS_PATH}"
        fi
        echo "Using ARTIFACTS_PATH: ${artifacts_dir}"
    elif compgen -G "./dist/*.gz" > /dev/null 2>&1; then
        echo "Auto-detected sdist(s) in ./dist"
        artifacts_dir="${PWD}/dist"
    elif compgen -G "./wheelhouse/*.whl" > /dev/null 2>&1; then
        echo "Auto-detected wheel(s) in ./wheelhouse"
        artifacts_dir="${PWD}/wheelhouse"
    else
        echo "❌  No artifacts found: ARTIFACTS_PATH is unset and no" \
             "*.gz in ./dist or *.whl in ./wheelhouse"
        return 1
    fi

    if [[ ! -d "$artifacts_dir" ]]; then
        echo "❌  Artifact directory does not exist: ${artifacts_dir}"
        return 1
    fi

    # ── Per-artifact upload loop ──────────────────────────────────────────────
    # find -print0 + IFS= read -r -d '' : NUL-safe; handles filenames with
    #   spaces or special characters without word-splitting or glob expansion.
    # sort -z                            : deterministic, reproducible order.
    # Failures are accumulated (any_failure=1) so that one bad artifact never
    # aborts the rest of the queue.
    local any_failure=0
    local count=0

    while IFS= read -r -d '' artifact; do
        [[ -f "$artifact" ]] || continue
        anaconda_upload_with_retry "$artifact" || any_failure=1
        # $((…)) always exits 0, so it is safe under set -e.
        # (( count++ )) would exit 1 when count==0, which set -e would treat
        # as a failure and abort the script.
        count=$(( count + 1 ))
    done < <(
        find "$artifacts_dir" -maxdepth 1 \
            \( -name '*.whl' -o -name '*.tar.gz' -o -name '*.gz' \) \
            -print0 \
        | sort -z
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    if (( count == 0 )); then
        echo "⚠️   No .whl or .gz files found in ${artifacts_dir}"
        return 1
    fi

    if (( any_failure )); then
        echo "❌  ${count} artifact(s) attempted; one or more failed (see above)"
        return 1
    fi

    echo "✅  All ${count} artifact(s) uploaded successfully  →  label: ${ANACONDA_LABEL:-main (default)}"
    echo "PyPI-style index: https://pypi.anaconda.org/${USERNAME}/simple"
    return 0
}

# upload_wheels() {
#     echo PWD: "${PWD}"
#     # echo "$(ls -lah)"
#     printf "%s\n" "$(ls -lah)"
#     # Ensure conda is available
#     # if ! command -v conda >/dev/null 2>&1; then
#     #     echo "conda not found" >&2
#     #     exit 1
#     # fi
#     # Initialize shell (CRITICAL)
#     # CONDA_BASE="$(conda info --base)"
#     # shellcheck disable=SC1090
#     # source "${CONDA_BASE}/etc/profile.d/conda.sh"
#     # ---------------------------
#     # CHANNEL FIX (IMPORTANT)
#     # ---------------------------
#     # conda config --add channels conda-forge  # primary source (newer builds, ARM support)
#     # conda config --add channels defaults  # (Anaconda channel) → fallback for legacy packages
#     # conda config --set channel_priority strict  # → prevents mixing incompatible builds
#     ## https://github.com/marketplace/actions/build-and-upload-conda-packages
#     ## https://docs.anaconda.com/anacondaorg/user-guide/packages/standard-python-packages/
#     # conda install -qy anaconda-client
#     export PATH=$CONDA/bin:$PATH
#     conda create -n upload -y anaconda-client || true
#     conda activate upload || true  # source activate upload
#     # conda's libmamba sharded repodata cache causes sqlite3 lock races on linux
#     # anaconda-client is unavailable via conda on win-arm64
#     pip install --no-cache-dir anaconda-client || true
#     # -----------------------------
#     # UPLOAD
#     # -----------------------------
#     if [[ "${ANACONDA_UPLOAD:-false}" == true ]]; then
#         if [[ -z "${TOKEN:-}" ]]; then
#             echo "no token set, not uploading"
#             return 0
#         fi
#         echo "TOKEN found, looking files..."
#         # local artifacts_path="${ARTIFACTS_PATH:?ARTIFACTS_PATH must be set}"
#         ## sdists are located under dist folder when built through setup.py
#         ## compgen is a Bash built-in that generates possible completions (filenames, commands, etc.).
#         ## -G uses a glob pattern (like *.gz) and returns matching filenames.
#         ## if ls ./dist/*.gz 1> /dev/null 2>&1; then
#         if compgen -G "./dist/*.gz" > /dev/null; then
#             echo "Found sdist..."
#             ## No quotes if you want globbing (e.g., *.gz) This will expand the glob correctly $ARTIFACTS_PATH/*
#             anaconda -t "${TOKEN}" upload --force -u "${USERNAME}" ./dist/*.gz
#         elif compgen -G "./wheelhouse/*.whl" > /dev/null; then
#             echo "Found wheel..."
#             ## Force a replacement if the remote file already exists -
#             ## nightlies will not have the commit ID in the filename, so
#             ## are named the same (1.X.Y.dev0-<platform/interpreter-tags>)
#             ## No quotes if you want globbing (e.g., *.gz) This will expand the glob correctly $ARTIFACTS_PATH/*
#             anaconda -q -t "${TOKEN}" upload --force -u "${USERNAME}" ./wheelhouse/*.whl
#         else
#             echo "Files do not exist"
#             return 1
#         fi
#         ## Your package is now available at http://anaconda.org/<USERNAME>/<PACKAGE>
#         # export PACKAGE="scikit-plots*"
#         echo "PyPI-style index: https://pypi.anaconda.org/$USERNAME/simple"
#     fi
# }
