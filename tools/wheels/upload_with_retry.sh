# tools/wheels/upload_with_retry.sh
# Source this file — do NOT execute it directly.
#
# WHY THIS EXISTS
# ───────────────
# Anaconda.org returns HTTP 503 Service Unavailable when its API is under
# high load.  The anaconda CLI propagates that as a Python ServerError and
# exits non-zero, aborting the entire upload step even though every wheel
# that was built is perfectly valid.  Because the failure is 100% transient
# (the server recovers in seconds to minutes) the correct fix is an
# exponential-backoff retry loop, NOT a rebuild.
#
# DESIGN CONSTRAINTS
# ──────────────────
# • Retry ONLY on 503 / Service Unavailable / high-load patterns.
#   All other failures (401 auth, 403 forbidden, 400 bad request, local I/O)
#   are permanent at this point and must surface immediately so they are not
#   silently masked by retrying.
# • Per-artifact retry budget so one slow file cannot starve the others.
# • find + NUL delimiters for safe enumeration (no glob expansion issues).
# • No set -euo pipefail at script scope; this file is sourced and must not
#   alter the caller's shell options.
#
# PREREQUISITES (exported by set_upload_vars from upload_wheels.sh)
# ─────────────────────────────────────────────────────────────────
#   TOKEN         — Anaconda.org API token
#   USERNAME      — Anaconda.org target user / organisation
#   ARTIFACTS_PATH — directory containing artifacts (for upload_wheels_with_retry)
#
# TUNEABLE VIA ENVIRONMENT (optional — all have sane defaults)
# ─────────────────────────────────────────────────────────────
#   UPLOAD_MAX_ATTEMPTS  = 5    total attempts per artifact before giving up
#   UPLOAD_RETRY_DELAY   = 30   initial back-off delay in seconds
#   UPLOAD_MAX_DELAY     = 300  back-off ceiling in seconds (5 min)
#
# TIMING PROFILE (defaults)
# ─────────────────────────
#   attempt 1  → fail → wait  30 s
#   attempt 2  → fail → wait  60 s
#   attempt 3  → fail → wait 120 s
#   attempt 4  → fail → wait 240 s
#   attempt 5  → fail → give up  (total worst-case wait ≈ 7.5 min per artifact)

# ── anaconda_upload_with_retry <artifact_path> ──────────────────────────────
# Uploads a single artifact with per-attempt retry on 503 only.
# Returns 0 on success; non-zero on non-retryable error or exhausted retries.
anaconda_upload_with_retry() {
    local artifact="$1"
    local basename_f; basename_f="$(basename "$artifact")"
    local max_attempts="${UPLOAD_MAX_ATTEMPTS:-5}"
    local delay="${UPLOAD_RETRY_DELAY:-30}"
    local max_delay="${UPLOAD_MAX_DELAY:-300}"
    local attempt=1
    local rc=0
    local out=""

    while (( attempt <= max_attempts )); do
        echo "⏫  [${attempt}/${max_attempts}] ${basename_f}"

        # Capture stdout+stderr and exit code without triggering set -e
        set +e
        out=$(anaconda --verbose -q -t "$TOKEN" upload --force -u "$USERNAME" "$artifact" 2>&1)
        rc=$?
        set -e

        echo "$out"

        # ── Success ──────────────────────────────────────────────────────────
        if (( rc == 0 )); then
            echo "✅  Uploaded: ${basename_f}"
            return 0
        fi

        # ── Non-retryable: surface immediately ───────────────────────────────
        # 503 patterns from anaconda's ServerError are the only transient class.
        # Auth errors (401/403), bad-request (400/422), and any local failure
        # will never succeed on retry — propagate them straight away.
        if ! echo "$out" | grep -qE "503|Service.Unavailable|high.load"; then
            echo "❌  Non-retryable upload failure for ${basename_f} (exit ${rc})"
            return $rc
        fi

        # ── Exhausted retry budget ────────────────────────────────────────────
        if (( attempt >= max_attempts )); then
            echo "❌  Giving up after ${max_attempts} attempts" \
                 "(503 persists): ${basename_f}"
            return $rc
        fi

        # ── Wait and retry ────────────────────────────────────────────────────
        local next_delay=$(( delay * 2 < max_delay ? delay * 2 : max_delay ))
        echo "⚠️   Anaconda.org 503 — high load, retrying in ${delay}s" \
             "(attempt ${attempt}/${max_attempts}; next back-off ${next_delay}s)"
        sleep "$delay"
        delay=$next_delay
        (( attempt++ ))
    done
}

# ── upload_wheels_with_retry ─────────────────────────────────────────────────
# Drop-in replacement for upload_wheels from upload_wheels.sh.
# Iterates every .whl and .tar.gz in ARTIFACTS_PATH and calls
# anaconda_upload_with_retry for each, collecting failures so that one bad
# artifact does not abort the remaining uploads.
# Exits 0 only when every artifact succeeded.
upload_wheels_with_retry() {
    local artifacts_path="${ARTIFACTS_PATH:?ARTIFACTS_PATH must be set}"
    local any_failure=0
    local count=0

    # find -print0 + read -d '' : NUL-safe; handles filenames with spaces
    # sort -z       : deterministic upload order (by filename)
    while IFS= read -r -d '' f; do
        [[ -f "$f" ]] || continue
        anaconda_upload_with_retry "$f" || any_failure=1
        (( count++ ))
    done < <(
        find "$artifacts_path" -maxdepth 1 \
            \( -name '*.whl' -o -name '*.tar.gz' \) \
            -print0 \
        | sort -z
    )

    if (( count == 0 )); then
        echo "⚠️   No .whl or .tar.gz files found in ${artifacts_path}"
    fi

    return $any_failure
}
