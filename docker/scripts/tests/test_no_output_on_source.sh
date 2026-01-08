#!/usr/bin/env sh
# docker/scripts/tests/test_no_output_on_source.sh
#
# sh docker/scripts/tests/test_no_output_on_source.sh
set -eu
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"

out="$( ( . "$HERE/../lib/common.sh" ) 2>&1 )"
[ -z "$out" ] || { printf '%s\n' "Unexpected output when sourcing:" "$out" >&2; exit 1; }

printf '%s\n' "OK: sourcing produced no output"
