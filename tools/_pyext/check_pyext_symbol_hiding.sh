#!/bin/bash
#
# check_pyext_symbol_hiding.sh PATHS...
#
# Check that .so files under the given directories export only a
# single public dynamic symbol each.
#

set -e


count_text_symbols() {
    nm -D --defined-only "$1" | wc -l
}

check_symbols() {
    # NUM=`count_text_symbols "$1"`
    NUM=$(count_text_symbols "$1")
    FILENAME=$(basename "$1")

    # special function error handling requires shared state between extension
    # modules that depend on it. There is a shared library encapsulating this
    # state which is an exception.
    if [[ "$FILENAME" == "libsf_error_state.so" ]]; then
        return 0
    fi

    if [[ "$NUM" != "1" ]]; then
        echo "$1: too many public symbols!"
        nm -D --defined-only "$1"
        exit 1
    fi
}

## -r tells read not to treat backslashes as escape characters.
## IFS= prevents leading/trailing whitespace from being trimmed (optional but good practice).
# find "$@" -type f -name '*.so' -print | while read F; do
find "$@" -type f -name '*.so' -print | while IFS= read -r F; do
    check_symbols "$F"
done

echo "Symbol hiding OK"
exit 0
