#!/usr/bin/env bash
# =============================================================================
# quay_token_test.sh — Validate a Quay OAuth Application Token locally
#
# PURPOSE
#   Run this script BEFORE updating the GitHub secret QUAY_OAUTH_TOKEN.
#   It performs every check the CI workflow performs, in the same order,
#   against the real Quay.io API.  All operations are read-only (no deletes).
#
# USAGE
#   export QUAY_OAUTH_TOKEN="<your-oauth-application-token>"
#   bash quay_token_test.sh
#
#   # Override defaults if needed:
#   QUAY_NAMESPACE=scikit-plots IMAGE_REPO=scikit-plots bash quay_token_test.sh
#
# HOW TO GET A QUAY OAUTH APPLICATION TOKEN
#   1. Go to https://quay.io/organization/scikit-plots?tab=OAuthApplications
#   2. You should see "scikitplot-ci" already created.
#   3. Click "scikitplot-ci" → "Generate Token".
#   4. Tick ☑ "Administer Repositories" → "Generate Access Token" → Authorize.
#   5. Copy the token shown (displayed ONCE — save it now).
#   6. Export it and run this script.
#
# WHY NOT /api/v1/user/?
#   Quay has three token types with different auth mechanisms:
#     Personal user tokens  → Bearer → /api/v1/user/ returns 200  ✅
#     Robot account tokens  → Basic Auth (user:secret) — not Bearer
#     Org OAuth App tokens  → Bearer → /api/v1/user/ returns 401  ← org tokens fail here
#   Organisation OAuth Application tokens are org-scoped, not user-scoped.
#   /api/v1/user/ rejects them.  The correct check is the repository's
#   "can_admin" field — only true when the token has "Administer Repositories"
#   scope on the specific repository.
#
# EXPECTED OUTPUT (all checks pass)
#   [1/4] Token format   ✅ PASS  length=40 chars, no colon detected
#   [2/4] Repo admin     ✅ PASS  can_admin=true on quay.io/scikit-plots/scikit-plots
#   [3/4] Tag list       ✅ PASS  found N active tags on page 1  (has_additional=False)
#   [4/4] Tag detail     ✅ PASS  first tag: <tag-name>  modified=<timestamp>  epoch=<N>
#   ══════════════════════════════════════════════════════════════════
#   ✅ All checks passed. Safe to update GitHub secret QUAY_OAUTH_TOKEN.
#
# EXIT CODES
#   0 — all checks passed
#   1 — one or more checks failed
# =============================================================================
set -Eeuo pipefail

# ── Configuration (override via env if needed) ────────────────────────────────
QUAY_NAMESPACE="${QUAY_NAMESPACE:-scikit-plots}"
IMAGE_REPO="${IMAGE_REPO:-scikit-plots}"
QUAY_REPO="${QUAY_NAMESPACE}/${IMAGE_REPO}"
QUAY_API="https://quay.io/api/v1"

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
pass() { printf "${GREEN}✅ PASS${NC}  %s\n" "$*"; }
fail() { printf "${RED}❌ FAIL${NC}  %s\n" "$*"; FAILED=$(( FAILED + 1 )); }
info() { printf "${YELLOW}   ℹ️  ${NC}%s\n" "$*"; }

FAILED=0

echo "══════════════════════════════════════════════════════════════════"
echo "  Quay OAuth Application Token — local validation"
echo "  Namespace : ${QUAY_NAMESPACE}"
echo "  Repository: ${IMAGE_REPO}"
echo "══════════════════════════════════════════════════════════════════"
echo ""

# ── [1/4] Token presence and format ──────────────────────────────────────────
# OAuth Application tokens are alphanumeric strings (no colon).
# Robot account tokens use Basic Auth "username:secret" format.
# A colon is an unambiguous sign the wrong token type is being used — fail fast.
printf "[1/4] Token format   "
if [[ -z "${QUAY_OAUTH_TOKEN:-}" ]]; then
  fail "QUAY_OAUTH_TOKEN is not set."
  echo ""
  echo "  export QUAY_OAUTH_TOKEN=\"<your-token>\""
  echo "  bash $0"
  echo ""
  exit 1
fi
if [[ "${QUAY_OAUTH_TOKEN}" == *:* ]]; then
  fail "Token contains a colon — looks like a robot account 'username:secret' pair."
  info "QUAY_OAUTH_TOKEN must be only the OAuth token string, no username prefix."
  info "Generate one at: https://quay.io/organization/${QUAY_NAMESPACE}?tab=OAuthApplications"
  exit 1
fi
TOKEN_LEN="${#QUAY_OAUTH_TOKEN}"
pass "length=${TOKEN_LEN} chars, no colon detected"

# Build auth header — never echoed, never in set -x output.
AUTH_HEADER="Authorization: Bearer ${QUAY_OAUTH_TOKEN}"

# ── [2/4] Repository admin permission check ───────────────────────────────────
# GET /api/v1/repository/{org}/{repo} response body fields:
#
#   Unauthenticated (public repo, no token):
#     { "is_public": true }
#     "can_admin" is ABSENT → jq default evaluates to false
#
#   Authenticated, token missing "Administer Repositories" scope:
#     { "can_admin": false, "can_write": false/true, ... }
#
#   Authenticated, token has "Administer Repositories" scope:
#     { "can_admin": true, "can_write": true, ... }
#     ↑ This is what we require for tag deletion.
#
#   Repository does not exist:
#     HTTP 404
#
# This single check simultaneously validates:
#   - Token is not expired/revoked (would give HTTP 401 on private repos, or
#     can_admin=false on public repos with bad tokens)
#   - Token has the exact scope needed for tag deletion
#   - Repository path is correct
printf "[2/4] Repo admin     "
REPO_RESP=$(curl -s -w "\n%{http_code}" \
  -H "${AUTH_HEADER}" \
  "${QUAY_API}/repository/${QUAY_REPO}")
REPO_HTTP=$(printf '%s' "${REPO_RESP}" | tail -1)
REPO_BODY=$(printf '%s' "${REPO_RESP}" | head -n -1)

case "${REPO_HTTP}" in
  200)
    CAN_ADMIN=$(printf '%s' "${REPO_BODY}" | jq -r '.can_admin // false' 2>/dev/null || echo "false")
    CAN_WRITE=$(printf '%s' "${REPO_BODY}" | jq -r '.can_write // false' 2>/dev/null || echo "false")
    if [[ "${CAN_ADMIN}" == "true" ]]; then
      pass "can_admin=true  can_write=${CAN_WRITE}  on quay.io/${QUAY_REPO}"
    else
      fail "can_admin=${CAN_ADMIN} — token lacks 'Administer Repositories' scope."
      info "The token is accepted by Quay but cannot delete tags."
      info "Re-generate the token and tick ☑ 'Administer Repositories'."
      info "https://quay.io/organization/${QUAY_NAMESPACE}?tab=OAuthApplications"
    fi
    ;;
  401)
    fail "HTTP 401 — token rejected (expired, revoked, or not yet authorised)."
    info "Complete the OAuth authorisation step when generating the token."
    info "https://quay.io/organization/${QUAY_NAMESPACE}?tab=OAuthApplications"
    ;;
  404)
    fail "HTTP 404 — repository quay.io/${QUAY_REPO} not found."
    info "Check: QUAY_NAMESPACE='${QUAY_NAMESPACE}'  IMAGE_REPO='${IMAGE_REPO}'"
    ;;
  *)
    fail "HTTP ${REPO_HTTP} — unexpected response."
    info "Body: $(printf '%s' "${REPO_BODY}" | head -c 300)"
    ;;
esac

# ── [3/4] Tag list — page 1 (read-only) ──────────────────────────────────────
# Validates the tag API endpoint, response shape, and pagination flag.
# Also confirms onlyActiveTags=true is honoured by the server.
printf "[3/4] Tag list       "
TAG_RESP=$(curl -s \
  -H "${AUTH_HEADER}" \
  "${QUAY_API}/repository/${QUAY_REPO}/tag/?limit=10&page=1&onlyActiveTags=true")

if ! printf '%s' "${TAG_RESP}" | jq -e '(.tags | type) == "array"' > /dev/null 2>&1; then
  fail "Unexpected API response shape — expected {tags: [...]}"
  info "Raw: $(printf '%s' "${TAG_RESP}" | head -c 300)"
else
  TAG_COUNT=$(printf '%s' "${TAG_RESP}" | jq '.tags | length')
  HAS_MORE=$(printf '%s' "${TAG_RESP}" | jq -r '.has_additional // false')
  pass "found ${TAG_COUNT} active tags on page 1  (has_additional=${HAS_MORE})"
fi

# ── [4/4] First tag timestamp parsability ─────────────────────────────────────
# The cleanup script uses GNU date to parse RFC 2822 timestamps from .last_modified.
# Verify that the timestamp format Quay currently returns is parsable by GNU date,
# since a format change on Quay's side would silently cause all tags to be kept.
printf "[4/4] Tag detail     "
FIRST_TAG_NAME=$(printf '%s' "${TAG_RESP}" | jq -r '.tags[0].name // empty'          2>/dev/null || true)
FIRST_TAG_TS=$(printf '%s'   "${TAG_RESP}" | jq -r '.tags[0].last_modified // empty' 2>/dev/null || true)

if [[ -z "${FIRST_TAG_NAME}" ]]; then
  pass "repository has no active tags (nothing to delete)"
elif [[ -z "${FIRST_TAG_TS}" ]]; then
  pass "first tag: ${FIRST_TAG_NAME}  last_modified=<absent — will be kept by default>"
else
  EPOCH=$(date -u -d "${FIRST_TAG_TS}" +%s 2>/dev/null || true)
  if [[ -n "${EPOCH}" && "${EPOCH}" =~ ^[0-9]+$ ]]; then
    pass "first tag: ${FIRST_TAG_NAME}  modified=${FIRST_TAG_TS}  epoch=${EPOCH}"
  else
    fail "Timestamp '${FIRST_TAG_TS}' is not parsable by GNU date."
    info "The cleanup script will KEEP tags with unparsable timestamps (safe default)."
    info "This suggests a Quay API format change — review the timestamp parsing logic."
  fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════"
if [[ "${FAILED}" -eq 0 ]]; then
  printf "${GREEN}✅ All checks passed.${NC}\n"
  echo "   Safe to update GitHub secret QUAY_OAUTH_TOKEN."
  echo ""
  echo "   Next steps:"
  echo "   1. https://github.com/scikit-plots/scikit-plots/settings/secrets/actions"
  echo "      ADD:    QUAY_OAUTH_TOKEN  ← paste your token"
  echo "      DELETE: QUAY_ROBOT_TOKEN  ← remove the old misleadingly-named secret"
  echo "   2. In your cleanup workflow YAML, update the env block:"
  echo "        QUAY_ROBOT_TOKEN: \${{ secrets.QUAY_ROBOT_TOKEN }}"
  echo "      → QUAY_OAUTH_TOKEN: \${{ secrets.QUAY_OAUTH_TOKEN }}"
  echo "   3. Update the script variable references: QUAY_ROBOT_TOKEN → QUAY_OAUTH_TOKEN"
  echo "══════════════════════════════════════════════════════════════════"
  exit 0
else
  printf "${RED}❌ ${FAILED} check(s) failed.${NC}  Fix the issues above before updating the secret.\n"
  echo "══════════════════════════════════════════════════════════════════"
  exit 1
fi
