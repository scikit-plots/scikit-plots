#!/usr/bin/env bash

# Prevent repeated initialization if nested Bash processes source BASH_ENV.
if [[ "${MAKE_ENV_INITIALIZED:-0}" == "1" ]]; then
	return 0
fi

MAKE_ENV_INITIALIZED=1
export MAKE_ENV_INITIALIZED

TIME_STYLE="${TIME_STYLE:-github}"
TRACE="${TRACE:-0}"

timestamp() {
	case "$TIME_STYLE" in
		github)
			LC_ALL=C date -u '+%a, %d %b %Y %H:%M:%S GMT'
			;;

		plain)
			printf '%s' "${EPOCHREALTIME:-0}"
			;;

		human)
			printf '%(%Y-%m-%dT%H:%M:%S%z)T' -1
			;;

		*)
			printf 'ERROR: Unsupported TIME_STYLE: %s\n' \
				"$TIME_STYLE" >&2
			printf 'Available styles: github, plain, human\n' >&2
			return 2
			;;
	esac
}

log() {
	local level="INFO"

	if [[ "${1:-}" == "DEBUG" ||
	      "${1:-}" == "INFO" ||
	      "${1:-}" == "WARN" ||
	      "${1:-}" == "ERROR" ]]; then
		level="$1"
		shift
	fi

	# printf '[%s] %-5s %s\n' \
	printf '%s %-5s %s\n' \
		"$(timestamp)" \
		"$level" \
		"$*"
}

die() {
	log ERROR "$*"
	return 1
}

# Bash expands PS4 for every xtrace line.
case "$TIME_STYLE" in
	github)
		# PS4='[$(LC_ALL=C date -u "+%a, %d %b %Y %H:%M:%S GMT")] ${BASH_SOURCE[0]:-main}:${LINENO:-0}: '
		PS4='$(LC_ALL=C date -u "+%a, %d %b %Y %H:%M:%S GMT") ${BASH_SOURCE[0]:-main}:${LINENO:-0}: '
		;;

	plain)
		# PS4='[${EPOCHREALTIME:-0}] ${BASH_SOURCE[0]:-main}:${LINENO:-0}: '
		PS4='${EPOCHREALTIME:-0} ${BASH_SOURCE[0]:-main}:${LINENO:-0}: '
		;;

	human)
		# PS4='[$(printf "%(%Y-%m-%dT%H:%M:%S%z)T" -1)] ${BASH_SOURCE[0]:-main}:${LINENO:-0}: '
		PS4='$(printf "%(%Y-%m-%dT%H:%M:%S%z)T" -1) ${BASH_SOURCE[0]:-main}:${LINENO:-0}: '
		;;
esac

if [[ "$TRACE" == "1" ]]; then
	set -x
fi
