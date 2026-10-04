#!/usr/bin/env bash
# docker/scripts/tests/test_install_scripts.sh
#
# bash docker/scripts/tests/test_install_scripts.sh
#
# Offline tests for install_nltk.sh and install_spacy.sh: every case uses
# archives built here and a file: base URL, so no network is needed and
# nothing outside a temporary directory is written.

set -Eeuo pipefail

HERE="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
NLTK="$HERE/../install_nltk.sh"
SPACY="$HERE/../install_spacy.sh"
PYTHON="${PYTHON:-python3}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/install_tests.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

passed=0
ok() { passed=$((passed + 1)); printf 'ok %d - %s\n' "$passed" "$1"; }
fail() { printf 'not ok - %s\n' "$1" >&2; exit 1; }

# expect_failure <text the error must contain> <description> <command...>
expect_failure() {
  local needle="$1" label="$2" output
  shift 2
  if output="$("$@" 2>&1)"; then fail "$label: succeeded, expected an error"; fi
  [[ "$output" == *"$needle"* ]] || fail "$label: error did not mention '$needle': $output"
  ok "$label"
}

mkdir -p "$WORK/mirror/corpora" "$WORK/mirror/tokenizers"
"$PYTHON" - "$WORK/mirror" <<'PY'
import sys
import zipfile

root = sys.argv[1]
with zipfile.ZipFile(root + "/corpora/alpha.zip", "w") as bundle:
    bundle.writestr("alpha/data.txt", "one")
with zipfile.ZipFile(root + "/tokenizers/beta.zip", "w") as bundle:
    bundle.writestr("beta/data.txt", "two")
with zipfile.ZipFile(root + "/corpora/escape.zip", "w") as bundle:
    bundle.writestr("../../escaped.txt", "x")
with open(root + "/corpora/junk.zip", "w") as handle:
    handle.write("not a zip")
PY

nltk_local() {
  PYTHON="$PYTHON" NLTK_ALLOW_INSECURE_URL=1 NLTK_VERIFY=0 NLTK_RETRIES=1 \
    NLTK_DATA_BASE_URL="file://$WORK/mirror" NLTK_DATA="$WORK/data" bash "$NLTK" "$@"
}

# --- install_nltk.sh ------------------------------------------------------
NLTK_PACKAGES="corpora/alpha tokenizers/beta" nltk_local >/dev/null 2>&1
[[ "$(cat "$WORK/data/corpora/alpha/data.txt")" == "one" ]] || fail "alpha was not installed"
[[ "$(cat "$WORK/data/tokenizers/beta/data.txt")" == "two" ]] || fail "beta was not installed"
ok "packages are installed under their categories"

second="$(NLTK_PACKAGES="corpora/alpha tokenizers/beta" nltk_local 2>&1)"
[[ "$second" == *"fetched 0 of 2"* ]] || fail "a second run fetched again: $second"
ok "a second run fetches nothing"

echo "changed" >"$WORK/data/corpora/alpha/data.txt"
NLTK_FORCE=1 NLTK_PACKAGES="corpora/alpha" nltk_local >/dev/null 2>&1
[[ "$(cat "$WORK/data/corpora/alpha/data.txt")" == "one" ]] || fail "NLTK_FORCE did not replace the package"
ok "NLTK_FORCE=1 replaces a present package"

[[ -z "$(find "$WORK/data" -name '.incoming-*')" ]] || fail "a staging directory was left behind"
ok "no staging directory is left behind"

NLTK_PACKAGES="corpora/alpha" nltk_local --verify-only >/dev/null 2>&1
ok "--verify-only passes when the package is present"
expect_failure "NLTK data missing" "--verify-only fails when a package is absent" \
  env NLTK_PACKAGES="corpora/absent" PYTHON="$PYTHON" NLTK_DATA="$WORK/data" bash "$NLTK" --verify-only

NLTK_DATA="$WORK/slip" NLTK_PACKAGES="corpora/escape" \
  expect_failure "could not unpack" "an archive entry leaving the data directory is refused" nltk_local
[[ ! -e "$WORK/escaped.txt" && ! -e "$WORK/slip/../escaped.txt" ]] || fail "the escaping entry was written"
ok "nothing was written outside the data directory"

NLTK_PACKAGES="corpora/junk" expect_failure "could not unpack" "a file that is not an archive is refused" nltk_local
NLTK_PACKAGES="corpora/nowhere" expect_failure "download failed" "a missing archive is an error" nltk_local

expect_failure "not a '<category>/<name>' package" "a traversal in a package name is refused" \
  env NLTK_PACKAGES="../../etc/passwd" bash "$NLTK" --list
expect_failure "not a '<category>/<name>' package" "shell syntax in a package name is refused" \
  env NLTK_PACKAGES='corpora/a;b' bash "$NLTK" --list
expect_failure "must start with https://" "a base URL that is not https is refused" \
  env NLTK_DATA_BASE_URL="http://example.com/p" NLTK_DATA="$WORK/none" PYTHON="$PYTHON" bash "$NLTK"
expect_failure "NLTK_RETRIES must be" "a retry count that is not a number is refused" \
  env NLTK_RETRIES="many" bash "$NLTK" --list
expect_failure "unknown argument" "an unknown argument is refused" bash "$NLTK" --nope

[[ "$(NLTK_PACKAGES="corpora/alpha tokenizers/beta" bash "$NLTK" --list | tr '\n' ' ')" == "corpora/alpha tokenizers/beta " ]] \
  || fail "--list did not print the configured packages"
ok "--list prints the configured packages"
[[ "$(bash "$NLTK" --list | wc -l)" -eq 9 ]] || fail "the default package list is not nine entries"
ok "the default list has nine packages"

: >"$WORK/github_env"
GITHUB_ENV="$WORK/github_env" NLTK_PACKAGES="corpora/alpha" nltk_local >/dev/null 2>&1
[[ "$(cat "$WORK/github_env")" == "NLTK_DATA=$WORK/data" ]] || fail "NLTK_DATA was not exported to GITHUB_ENV"
ok "NLTK_DATA is exported for later GitHub Actions steps"

bash "$NLTK" --help | grep -q "install_nltk.sh" || fail "--help printed nothing useful"
ok "--help describes the script"

# --- install_spacy.sh (argument handling; no download) ---------------------
[[ "$(SPACY_MODELS="en_core_web_sm de_core_news_sm" bash "$SPACY" --list | tr '\n' ' ')" == "en_core_web_sm de_core_news_sm " ]] \
  || fail "install_spacy --list did not print the configured pipelines"
ok "install_spacy --list prints the configured pipelines"
expect_failure "not a spaCy pipeline name" "a pip option as a pipeline name is refused" \
  env SPACY_MODELS="--index-url=http://x" bash "$SPACY" --list
expect_failure "not a spaCy pipeline name" "shell syntax in a pipeline name is refused" \
  env SPACY_MODELS='en_core_web_sm;x' bash "$SPACY" --list
expect_failure "SPACY_RETRIES must be" "a retry count that is not a number is refused" \
  env SPACY_RETRIES="0" bash "$SPACY" --list
expect_failure "unknown argument" "an unknown argument is refused by install_spacy" bash "$SPACY" --nope
bash "$SPACY" --help | grep -q "install_spacy.sh" || fail "install_spacy --help printed nothing useful"
ok "install_spacy --help describes the script"

printf '1..%d\nall %d checks passed\n' "$passed" "$passed"
