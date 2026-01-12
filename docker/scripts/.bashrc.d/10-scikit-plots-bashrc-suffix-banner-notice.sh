#!/usr/bin/env bash
# docker/scripts/.bashrc.d/10-scikit-plots-bashrc-suffix-banner-notice.sh

# ====================================================================
# ________                               _______________
# ___  __/__________________________________  ____/__  /________      __
# __  /  _  _ \_  __ \_  ___/  __ \_  ___/_  /_   __  /_  __ \_ | /| / /
# _  /   /  __/  / / /(__  )/ /_/ /  /   _  __/   _  / / /_/ /_ |/ |/ /
# /_/    \___//_/ /_//____/ \____//_/    /_/      /_/  \____/____/|__/
# ====================================================================
#  ____       _ _    _ _              _       _
# / ___|  ___(_) | _(_) |_      _ __ | | ___ | |_ ___
# \___ \ / __| | |/ / | __|____| '_ \| |/ _ \| __/ __|
#  ___) | (__| |   <| | ||_____| |_) | | (_) | |_\__ \\
# |____/ \___|_|_|\_\_|\__|    | .__/|_|\___/ \__|___/
#                              |_|
# ====================================================================
#  ____           _   _      _   _                     _           _
# / ___|    ___  (_) | | __ (_) | |_           _ __   | |   ___   | |_   ___
# \___ \   / __| | | | |/ / | | | __|  _____  | '_ \  | |  / _ \  | __| / __|
#  ___) | | (__  | | |   <  | | | |_  |_____| | |_) | | | | (_) | | |_  \__ \\
# |____/   \___| |_| |_|\_\ |_|  \__|         | .__/  |_|  \___/   \__| |___/
#                                             |_|
# ====================================================================
# to a global/system file /etc/bash.bashrc
# When bash initializes a non-login interactive bash shell on a Debian/Ubuntu-like system, the shell first reads /etc/bash.bashrc and then reads ~/.bashrc.
# System wide initialization file /etc/bash.bashrc and the standard personal initialization file ~/.bashrc if the shell is interactivе.
# https://github.com/tensorflow/build/blob/master/tensorflow_runtime_dockerfiles/cpu.Dockerfile
# https://github.com/tensorflow/build/blob/master/tensorflow_runtime_dockerfiles/bashrc
# https://github.com/jupyter/docker-stacks/issues/815
# https://www.gnu.org/software/bash/manual/bash.html
# https://linux.die.net/man/1/bash
# ====================================================================

# >>> 10-scikit-plots-bashrc-suffix-banner-notice.sh scikit-plots personal initialization >>>
# ====================================================================

# Interactive shells only (must be first) $- contains shell flags (e.g. himBH).
# case $- in *i*) ;; *) return 0 ;; esac
case $- in *i*) ;; *) return || true ;; esac

## Do not print anything if this is not being used interactively
## bash -ic 'echo hi'
[ -z "$PS1" ] && return

# Avoid double execution if prefix is sourced both from /etc/bashrc.d and ~/.bashrc.d
if [[ -n "${__SCIKIT_PLOTS_BASHRC_PREFIX_BANNER_NOTICE_ONCE:-}" ]]; then
  return 0
fi
__SCIKIT_PLOTS_BASHRC_PREFIX_BANNER_NOTICE_ONCE=1

# >>> ASCII banner for scikit-plots >>>
# ====================================================================
# https://tldp.org/LDP/abs/html/here-docs.html
# cat <<EOF | grep 'b' | tee b.txt  # Pass multi-line string to a pipe in Bash
# cat << EOF [>|>> "$rc_file"] "$VAR"... EOF  # Expanded variable, command
# cat << 'EOF' | envsubst '${VAR}' >> "$rc_file" ..."$VAR"... EOF  # literal string variable, command
# cat << EOF	Starts heredoc, with variable expansion
# cat << 'EOF'	Starts heredoc, no variable expansion as literal string (safe for scripts)
# EOF (or any other delimiter like EOL, MYMARKER, etc.) is just a label used to indicate the end of a here-document.
# 'EOF' (quoted) vs EOF (unquoted): Just controls whether variable expansion and command substitution happens inside the heredoc.
# Use <<EOF and manually escape variables and command substitution you don't want
# cat << 'EOF' | envsubst '${VAR}' — tells it to only expand $VAR, apt-get install -y envsubst
# cat <<EOF "$VAR" OK — double quotes inside block
# >> filename is what redirects the output to the file, either appending (>>) or overwriting (>) it.
# >> "$rc_file": Just controls where the heredoc content goes (append mode).
# The single quotes (') prevent shell variable expansion or escape issues inside the block. So \ stays \, not interpreted.
# All backslashes (\) are safe — no need to double them.
echo -e "\e[1;31m"
cat << 'TF'
 ____       _ _    _ _              _       _
/ ___|  ___(_) | _(_) |_      _ __ | | ___ | |_ ___
\___ \ / __| | |/ / | __|____| '_ \| |/ _ \| __/ __|
 ___) | (__| |   <| | ||_____| |_) | | (_) | |_\__ \
|____/ \___|_|_|\_\_|\__|    | .__/|_|\___/ \__|___/
                             |_|
TF
echo -e "\e[0;33m"

if [[ $EUID -eq 0 ]]; then
  cat <<WARN
WARNING: You are running this container as root, which can cause new files in
mounted volumes to be created as the root user on your host machine.

To avoid this, run the container by specifying your user's userid:

$ docker run -u \$(id -u):\$(id -g) args...
WARN

else
  cat << EXPL
You are running this container as user with ID $(id -u) and group $(id -g),
which should map to the ID and group for your user on the Docker host. Great!
EXPL

fi
# Turn off colors
echo -e "\e[m"
# ====================================================================
# <<< ASCII banner for scikit-plots <<<

# >>> bash-first-run-notice.txt >>>
# ====================================================================
# { command -v sudo >/dev/null && sudo -n true && cp ./docker/scripts/bash-first-run-notice.txt /etc/bash-first-run-notice.txt; } || true
# sudo cp ./docker/scripts/bash-first-run-notice.txt /etc/bash-first-run-notice.txt || true
# rm -rf ~/.first-run-notice-already-displayed ~/.first-run-notice-shown
echo -e "\n\033[1;33m📢  First Run Notice:\033[0m
📝  See Also: cat docker/scripts/bash-first-run-notice.txt
📝  See Also: cat ~/.bash-first-run-notice.txt
"
## Location of the notice file and marker
## echo "Welcome to the container! Here's how to get started..." > ~/.bash-first-run-notice.txt
## "/etc/skel/.bashrc" is a template file used when creating new user accounts on Unix-like systems.
## Try to show a first-run notice file from a list of candidates it hasn't been shown yet or before
MARKER_FILE=~/.first-run-notice-already-displayed  # init_marker
for NOTICE_FILE in \
  "$(realpath .)/docker/scripts/bash-first-run-notice.txt" \
  ~/.bash-first-run-notice.txt \
  ~/bash-first-run-notice.txt \
  "/etc/skel/.bash-first-run-notice.txt" \
  "/etc/skel/bash-first-run-notice.txt" \
  "/etc/.bash-first-run-notice.txt" \
  "/etc/bash-first-run-notice.txt"
do
  if [[ -f "$NOTICE_FILE" && ! -f "$MARKER_FILE" ]]; then
    # cat "$NOTICE_FILE"
    cat -- "$NOTICE_FILE" 2>/dev/null || true
    echo  # Newline after notice

    ## Try to create marker to avoid repeating this notice
    touch "$MARKER_FILE" || {
      echo "⚠️  Could not create marker file at: $MARKER_FILE";
      echo "ℹ️  Please create it manually to avoid repeating this notice.";
    }
    break  # Only show the first found notice
  fi
done

# Optional: show first-run hint (only if notice exists and flag enabled)
# if [[ "${SP_SHOW_NOTICE:-0}" == "1" ]]; then
#   if [[ -f "${HOME}/.bash-first-run-notice.txt" ]]; then
#     cat -- "${HOME}/.bash-first-run-notice.txt" 2>/dev/null || true
#   fi
# fi
# ====================================================================
# <<< bash-first-run-notice.txt <<<

# ====================================================================
# <<< 10-scikit-plots-bashrc-suffix-banner-notice.sh scikit-plots personal initialization <<<
