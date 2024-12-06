#!/bin/bash
######################################################################
## Load Logging Functions
######################################################################
# Function to load utilities
load_logs() {
    local utils_path="${1:-../../../.github/scripts/utils.sh}"  # Use provided path or default
    if [[ -f "$utils_path" ]]; then
        source "$utils_path"
        log "Loaded utilities from $utils_path"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] utils.sh not found at $utils_path" >&2
        exit 1
    fi
}
# Call the function with an optional argument
# Use "$1" if passed when running the script; otherwise, the default path is used.
# load_logs "$1"
# Example usage
# log "Starting script1..."
# success "Script1 executed successfully."