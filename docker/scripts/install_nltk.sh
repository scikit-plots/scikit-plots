#!/usr/bin/env bash
# docker/scripts/install_nltk.sh
# ===============================================================
# install_nltk.sh — install NLTK data packages without the NLTK downloader
# ===============================================================
# USER NOTES
# - Run it anywhere a shell runs: a GitHub or CircleCI job, a Dockerfile RUN,
#   a devcontainer post-create step, a laptop.
#
#     bash docker/scripts/install_nltk.sh
#     NLTK_PACKAGES="tokenizers/punkt_tab corpora/wordnet" bash docker/scripts/install_nltk.sh
#     bash docker/scripts/install_nltk.sh --list
#     bash docker/scripts/install_nltk.sh --verify-only
#
# - Running it again is cheap: a package that is already there is not fetched.
# - It exits non-zero if any package is missing at the end.
#
# DEV NOTES
# - Why not `python -m nltk.downloader`: NLTK refuses a download that leaves
#   through a proxy, because it cannot pin the address it validated. CircleCI
#   and many corporate networks use one. The downloader then prints "Security
#   Violation" and exits 0, so the job carries on and fails much later, far
#   from the cause. This script fetches the same archives from the same
#   repository with curl, and checks the result.
# - Reproducible: archives come from one commit of nltk/nltk_data
#   (NLTK_DATA_REF), not from a moving branch. Set NLTK_DATA_REF=gh-pages to
#   follow the branch instead.
# - Safe extraction: an archive is tested before it is unpacked, and an entry
#   that would land outside the target directory is refused.
#
# ENV VARS
# - NLTK_DATA             : target directory (default: $HOME/nltk_data)
# - NLTK_PACKAGES         : space-separated "<category>/<name>" list
# - NLTK_DATA_REF         : commit or branch of nltk/nltk_data
# - NLTK_DATA_BASE_URL    : full base URL, overrides NLTK_DATA_REF (mirrors)
# - NLTK_FORCE=1          : fetch even when the package is present
# - NLTK_VERIFY=0         : skip the final check with the nltk library
# - NLTK_RETRIES          : download attempts per archive (default: 5)
# - NLTK_ALLOW_INSECURE_URL=1 : permit a base URL that is not https (tests)
# - PYTHON                : interpreter to use (default: python3, then python)
# ===============================================================

set -Eeuo pipefail

readonly DEFAULT_REF="550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
readonly DEFAULT_PACKAGES="tokenizers/punkt tokenizers/punkt_tab corpora/omw-1.4 corpora/wordnet corpora/words chunkers/maxent_ne_chunker chunkers/maxent_ne_chunker_tab taggers/averaged_perceptron_tagger taggers/averaged_perceptron_tagger_eng"

log() { printf '[install_nltk] %s\n' "$*" >&2; }
die() { printf '[install_nltk] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
install_nltk.sh — install NLTK data packages without the NLTK downloader.

Usage: install_nltk.sh [--list] [--verify-only] [--help]

  --list         print the packages that would be installed and exit
  --verify-only  check that the packages are present; download nothing
  --help         show this text

Configuration is by environment variable (NLTK_DATA, NLTK_PACKAGES,
NLTK_DATA_REF, ...); see the header of this file.
EOF
}

MODE="install"
for argument in "$@"; do
  case "$argument" in
    --list) MODE="list" ;;
    --verify-only) MODE="verify" ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $argument (try --help)" ;;
  esac
done

NLTK_DATA="${NLTK_DATA:-${HOME:?HOME is not set and NLTK_DATA was not given}/nltk_data}"
NLTK_PACKAGES="${NLTK_PACKAGES:-$DEFAULT_PACKAGES}"
NLTK_DATA_REF="${NLTK_DATA_REF:-$DEFAULT_REF}"
NLTK_DATA_BASE_URL="${NLTK_DATA_BASE_URL:-https://raw.githubusercontent.com/nltk/nltk_data/${NLTK_DATA_REF}/packages}"
NLTK_RETRIES="${NLTK_RETRIES:-5}"

# --- validate the configuration before touching anything ------------------
[[ "$NLTK_RETRIES" =~ ^[1-9][0-9]*$ ]] || die "NLTK_RETRIES must be a positive whole number, got '$NLTK_RETRIES'"
[[ "$NLTK_DATA_REF" =~ ^[A-Za-z0-9._/-]+$ ]] || die "NLTK_DATA_REF has characters a git ref cannot have: '$NLTK_DATA_REF'"
if [[ "$NLTK_DATA_BASE_URL" != https://* && "${NLTK_ALLOW_INSECURE_URL:-0}" != "1" ]]; then
  die "NLTK_DATA_BASE_URL must start with https:// (got '$NLTK_DATA_BASE_URL')"
fi

read -r -a PACKAGES <<<"$NLTK_PACKAGES"
[[ "${#PACKAGES[@]}" -gt 0 ]] || die "NLTK_PACKAGES is empty"
for package in "${PACKAGES[@]}"; do
  # One category, one name, nothing that could leave the data directory.
  [[ "$package" =~ ^[a-z_]+/[A-Za-z0-9._-]+$ && "$package" != *..* ]] \
    || die "not a '<category>/<name>' package: '$package'"
done

if [[ "$MODE" == "list" ]]; then
  printf '%s\n' "${PACKAGES[@]}"
  exit 0
fi

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON="python3"
  elif command -v python >/dev/null 2>&1; then PYTHON="python"
  else die "no python interpreter found; set PYTHON"
  fi
fi
command -v "$PYTHON" >/dev/null 2>&1 || die "PYTHON='$PYTHON' is not runnable"

is_installed() {
  # A package is a directory, or for a few packages a single file, named
  # after it under its category.
  local target="$NLTK_DATA/$1"
  [[ -d "$target" && -n "$(ls -A "$target" 2>/dev/null)" ]] || [[ -f "$target" ]]
}

fetch() {
  # fetch <url> <destination>; a file: URL is copied, for tests and mirrors.
  local url="$1" destination="$2"
  if command -v curl >/dev/null 2>&1; then
    local protocols="=https"
    [[ "${NLTK_ALLOW_INSECURE_URL:-0}" == "1" ]] && protocols="=https,http,file"
    curl --fail --silent --show-error --location \
      --proto "$protocols" --retry "$NLTK_RETRIES" --retry-delay 2 --retry-connrefused \
      --connect-timeout 30 --max-time 600 \
      --output "$destination" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --quiet --tries="$NLTK_RETRIES" --timeout=30 --output-document="$destination" "$url"
  else
    die "neither curl nor wget is available"
  fi
}

extract() {
  # extract <archive> <directory>: test the archive, refuse an entry that
  # would be written outside <directory>, then unpack.
  "$PYTHON" - "$1" "$2" <<'PY'
import os
import shutil
import sys
import tempfile
import zipfile

archive, directory = sys.argv[1], os.path.realpath(sys.argv[2])
staging = tempfile.mkdtemp(prefix=".incoming-", dir=directory)
try:
    with zipfile.ZipFile(archive) as bundle:
        damaged = bundle.testzip()
        if damaged is not None:
            sys.exit("corrupt archive member: %s" % damaged)
        for name in bundle.namelist():
            target = os.path.realpath(os.path.join(staging, name))
            if target != staging and not target.startswith(staging + os.sep):
                sys.exit("archive entry leaves the data directory: %r" % name)
        bundle.extractall(staging)
    # Unpacked beside its destination and moved in whole, so an interrupted
    # run never leaves half a package that a later run would take as present.
    for entry in sorted(os.listdir(staging)):
        destination = os.path.join(directory, entry)
        if os.path.isdir(destination) and not os.path.islink(destination):
            shutil.rmtree(destination)
        elif os.path.lexists(destination):
            os.remove(destination)
        os.replace(os.path.join(staging, entry), destination)
except zipfile.BadZipFile as exc:
    sys.exit("not a zip archive: %s" % exc)
finally:
    shutil.rmtree(staging, ignore_errors=True)
PY
}

verify_files() {
  local missing=()
  for package in "${PACKAGES[@]}"; do
    is_installed "$package" || missing+=("$package")
  done
  if [[ "${#missing[@]}" -gt 0 ]]; then
    die "NLTK data missing in $NLTK_DATA: ${missing[*]}"
  fi
}

verify_with_nltk() {
  # Ask the library itself, so that a package it cannot find is caught here
  # and not an hour into a build.
  if ! "$PYTHON" -c 'import nltk' >/dev/null 2>&1; then
    log "the nltk library is not installed for $PYTHON; checked the files only"
    return 0
  fi
  NLTK_DATA="$NLTK_DATA" "$PYTHON" - "${PACKAGES[@]}" <<'PY'
import sys

import nltk

missing = []
for package in sys.argv[1:]:
    try:
        nltk.data.find(package)
    except LookupError:
        try:
            nltk.data.find(package + "/")
        except LookupError:
            missing.append(package)
if missing:
    sys.exit("NLTK cannot find: " + ", ".join(missing))
if "tokenizers/punkt_tab" in sys.argv[1:] or "tokenizers/punkt" in sys.argv[1:]:
    from nltk.tokenize import sent_tokenize

    if sent_tokenize("A. B.") != ["A.", "B."]:
        sys.exit("sentence tokenizer gave an unexpected result")
print("[install_nltk] verified with nltk %s" % nltk.__version__, file=sys.stderr)
PY
}

if [[ "$MODE" == "install" ]]; then
  WORK="$(mktemp -d "${TMPDIR:-/tmp}/install_nltk.XXXXXX")"
  trap 'rm -rf "$WORK"' EXIT
  fetched=0
  for package in "${PACKAGES[@]}"; do
    category="${package%%/*}"
    name="${package##*/}"
    if [[ "${NLTK_FORCE:-0}" != "1" ]] && is_installed "$package"; then
      log "present: $package"
      continue
    fi
    mkdir -p "$NLTK_DATA/$category"
    archive="$WORK/$name.zip"
    log "fetching: $package"
    fetch "$NLTK_DATA_BASE_URL/$category/$name.zip" "$archive" \
      || die "download failed: $NLTK_DATA_BASE_URL/$category/$name.zip"
    extract "$archive" "$NLTK_DATA/$category" || die "could not unpack $package"
    rm -f "$archive"
    fetched=$((fetched + 1))
  done
  log "fetched $fetched of ${#PACKAGES[@]} package(s) into $NLTK_DATA"
fi

verify_files
if [[ "${NLTK_VERIFY:-1}" == "1" ]]; then
  verify_with_nltk
fi

# Let later steps of a GitHub Actions job find the data.
if [[ -n "${GITHUB_ENV:-}" && -w "${GITHUB_ENV}" ]]; then
  printf 'NLTK_DATA=%s\n' "$NLTK_DATA" >>"$GITHUB_ENV"
fi
log "ok: ${#PACKAGES[@]} package(s) in $NLTK_DATA"
