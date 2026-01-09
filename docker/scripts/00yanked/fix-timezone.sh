#!/bin/bash
set -euo pipefail

: <<'DOC'
fix-timezone.sh — Sync container timezone with host (or custom) and fix file timestamps

Usage:
  1. Save this script as fix-timezone.sh
  2. Make it executable:
       chmod +x fix-timezone.sh
  3. Run inside your container:
       ./fix-timezone.sh
  4. Optional: Pass arguments for timezone and work directory:
       ./fix-timezone.sh "UTC" "./"
       ./fix-timezone.sh "Etc/GMT+0000"
       ./fix-timezone.sh "Europe/Istanbul"

Arguments:
  $1 — Timezone name
        Default priority:
          1) First argument passed
          2) /etc/timezone content (if mounted)
          3) date +%Z (timezone abbreviation) or +%z (offset)
          4) Europe/Istanbul (fallback)
  $2 — Directory whose timestamps should be fixed (default: current dir via realpath ./)

Description:
  - Installs tzdata if needed
  - Sets /etc/localtime to the specified timezone
  - Reconfigures system timezone without interactive prompts
  - Updates timestamps of all files in the given directory to "now"
DOC

detect_timezone() {
    if [[ -n "${1:-}" ]]; then
        echo "$1"
    elif [[ -f /etc/timezone ]]; then
        cat /etc/timezone
    else
        local TZ_ABBR=$(date +%Z)
        local TZ_OFFSET=$(date +%z)
        if [[ -f /usr/share/zoneinfo/$TZ_ABBR ]]; then
            echo "$TZ_ABBR"
        else
            if command -v timedatectl &>/dev/null; then
                timedatectl list-timezones | grep -E "$TZ_OFFSET" | head -n 1 || echo "Etc/GMT${TZ_OFFSET}"
            else
                echo "Etc/GMT${TZ_OFFSET}"
            fi
        fi
    fi
}

set_timezone_and_fix_timestamps() {
    local TZ_NAME
    TZ_NAME=$(detect_timezone "${1:-}")
    local WORK_DIR="${2:-$(realpath ./)}"

    echo "🕒 Setting timezone to $TZ_NAME..."

    if ! dpkg -s tzdata &>/dev/null; then
        apt-get update -qq
        DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata > /dev/null
    fi

    if [[ -f "/usr/share/zoneinfo/$TZ_NAME" ]]; then
        ln -sf "/usr/share/zoneinfo/$TZ_NAME" /etc/localtime
    else
        echo "⚠️ Timezone '$TZ_NAME' not found, using UTC."
        ln -sf "/usr/share/zoneinfo/UTC" /etc/localtime
    fi
    dpkg-reconfigure -f noninteractive tzdata > /dev/null

    if [[ -d "$WORK_DIR" ]]; then
        echo "📁 Updating timestamps in $WORK_DIR..."
        # find . -exec touch {} \;
        # find . -exec touch {} +
        find "$WORK_DIR" -exec touch -d "now" {} +
    else
        echo "⚠️ Directory $WORK_DIR not found, skipping timestamp update."
    fi

    echo "✅ Timezone and timestamps updated."
}

set_timezone_and_fix_timestamps "$@"
