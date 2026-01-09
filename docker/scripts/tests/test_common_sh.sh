#!/usr/bin/env sh
# docker/scripts/tests/test_common_sh.sh
#
# sh docker/scripts/tests/test_common_sh.sh
# shellcheck -s sh docker/scripts/common.sh docker/scripts/tests/test_common_sh.sh

set -eu

HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
# Source the library (must not print anything or install anything)
. "$HERE/../common.sh"

# Basic invariants
[ "${COMMON_SH__LOADED:-0}" = "1" ] || die "Load guard not set"

# Color should be OFF by default (canonical: COMMON_COLOR_MODE=never)
[ "${COMMON_COLOR_MODE:-}" = "never" ] || die "Default COMMON_COLOR_MODE must be 'never'"

# Functions exist (POSIX check: command -v works for functions in sh)
command -v log_info >/dev/null 2>&1 || die "log_info missing"
command -v detect_os >/dev/null 2>&1 || die "detect_os missing"
command -v mktemp_dir >/dev/null 2>&1 || die "mktemp_dir missing"
command -v load_env_kv >/dev/null 2>&1 || die "load_env_kv missing"

# mktemp_dir creates a directory
TMP="$(mktemp_dir common_test)"
[ -d "$TMP" ] || die "mktemp_dir did not create dir: $TMP"
cleanup_dir "$TMP"
[ ! -e "$TMP" ] || die "cleanup_dir failed: $TMP"

log_success "common.sh smoke test OK"
