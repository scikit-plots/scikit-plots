#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# vendor_repo.sh — Deterministic vendoring with reproducibility and integrity
# -----------------------------------------------------------------------------

set -o errexit
set -o nounset
set -euo pipefail

#######################################
# UTILITY: Log message with level
#######################################
log() {
  local level="$1"; shift
  echo "[$level] $*"
}

# ---------------------------------------------------------------------
# Compute deterministic tree hash (excluding lock/readme/gitignore)
# ---------------------------------------------------------------------
# Automatically falls back to Python if `sha256sum` or `find` is missing.
# TREE_INFO=$(...) + split
# TREE_INFO=$(compute_tree_hash "$TARGET_DIR")
# TREE_MODE="${TREE_INFO%% *}"
# TREE_HASH="${TREE_INFO#* }"
# Compute actual tree hash (mode + hash)
# read -r ACTUAL_MODE ACTUAL_HASH < <(compute_tree_hash "$TARGET_DIR")
compute_tree_hash() {
    # Excludes vendor.lock.json, README.md, and .gitignore for reproducibility.
    local EXCLUDES=("vendor.lock.json" "$README_NAME" ".gitignore")
    local exclude_expr=()
    for f in "${EXCLUDES[@]}"; do
        exclude_expr+=(-not -name "$f")
    done
    local mode hash
    local dir="$1"

    # Try Bash + sha256sum pipeline first
    if command -v sha256sum >/dev/null 2>&1 && command -v find >/dev/null 2>&1; then
        mode="bash-sha256sum"
        echo "⚙️  Using $mode mode for tree hash..." >&2
        # Portable find + sort + sha256sum hash deterministically pipeline (to skip excluded files)
        hash=$(
            find "$dir" "${exclude_expr[@]}" -type f -print0 \
            | sort -z \
            | xargs -0 sha256sum \
            | sort \
            | sha256sum \
            | awk '{print $1}'
        )
        echo "$mode $hash"
    else
        # Fallback: compute SHA256 of all files for integrity verification
        mode="python-hashlib"
        echo "⚙️  Falling back to $mode mode for tree hash..." >&2
        hash=$(python - << EOF "$dir"
import hashlib, os, sys
root = sys.argv[1]
exclude = {"vendor.lock.json", "$README_NAME", ".gitignore"}
hasher = hashlib.sha256()
for path, _, files in os.walk(root):
    for f in sorted(files):
        if f in exclude: continue
        full = os.path.join(path, f)
        rel = os.path.relpath(full, root)
        hasher.update(rel.encode())
        with open(full, "rb") as fp:
            while chunk := fp.read(8192):
                hasher.update(chunk)
print(hasher.hexdigest())
EOF
)
        echo "$mode $hash"
    fi
}

#######################################
# CONFIGURABLE VARIABLES
#######################################
# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
# Notice: no quotes around the space-separated paths.
# Each path becomes a separate item in the SRC_SUBDIRS array.
function usage() {
    cat <<'USAGE'
Usage:
  Full vendoring (clone + copy + hash + README):
    vendor_repo.sh --repo-url URL --repo-ref REF --target-dir PATH
                   [--src-subdir "SUBDIR ..."] [--src-subdirs SUBDIR [SUBDIR ...]]
                   [--move-to PATH] [--nested-folder NAME] [--readme-name NAME]
                   [--ensure-init-py|-i] [--future-annotations|-A]
                   [--scope-src-subdirs|-S] [--dry-run|-n]

  Verify only (read-only, exits 2 on drift):
    vendor_repo.sh --target-dir PATH --check

  Refresh only the tree hash (no re-clone):
    vendor_repo.sh --target-dir PATH --update-hash [--ensure-init-py|-i] [--future-annotations|-A]

  Maintenance only on an already-vendored tree (no re-clone):
    vendor_repo.sh --target-dir PATH [--ensure-init-py|-i] [--future-annotations|-A] [--dry-run|-n]

Maintenance flags (usable in the full flow, --update-hash, or standalone):
  -i, --ensure-init-py       Create a missing __init__.py (empty) in every nested
                              folder under --target-dir. Never modifies or
                              recreates one that already exists. Ignored (with a
                              warning) under --check, which stays read-only.
  -A, --future-annotations   Insert 'from __future__ import annotations' as the
                              first statement after any shebang/leading comments/
                              module docstring in every .py file under
                              --target-dir. Skipped for empty files, files with no
                              import statement, files that already have it, and
                              files that don't parse as valid Python. Ignored
                              (with a warning) under --check.
  -S, --scope-src-subdirs    Narrow -i/-A to only the paths named in
                              --src-subdirs (resolved to their actual on-disk
                              location, even after --nested-folder/--move-to),
                              instead of walking the whole --target-dir.
                              Optional; default is the whole target (unchanged
                              behavior). Only has an effect when --src-subdirs
                              was also given in this invocation; otherwise -i/-A
                              fall back to the whole target with a note.
  -n, --dry-run              Report what --ensure-init-py / --future-annotations
                              would change, without writing anything.
USAGE
}
MODE="${MODE:-"update"}"          # default (was mistakenly keyed off REPO_URL before)
REPO_URL="${REPO_URL:-""}"        # Remote Git repo URL
REPO_REF="${REPO_REF:-""}"        # Ref Branch, Tag, or Commit SHA
TARGET_DIR="${TARGET_DIR:-""}"    # Directory to clone into
SRC_SUBDIR="${SRC_SUBDIR:-"."}"   # legacy single subdir (for backward compatibility)
SRC_SUBDIRS=()                    # list of subdirs/files (new, plural form)
README_NAME="README.md"           # README.md
MOVE_TO=""                        # optional move, default: do not move
# Optional nested folder name inside target to move
# --nested-folder "astropy" means only move $TARGET_DIR/astropy → MOVE_TO
NESTED_FOLDER=""                          # optional nested folder to move
ENSURE_INIT_PY="${ENSURE_INIT_PY:-false}"         # --ensure-init-py / -i
FUTURE_ANNOTATIONS="${FUTURE_ANNOTATIONS:-false}" # --future-annotations / -A
SCOPE_SRC_SUBDIRS="${SCOPE_SRC_SUBDIRS:-false}"    # --scope-src-subdirs / -S (default: whole target)
DRY_RUN="${DRY_RUN:-false}"                       # --dry-run / -n

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-url) REPO_URL="$2"; shift 2 ;;
        --repo-ref) REPO_REF="$2"; shift 2 ;;
        --target-dir) TARGET_DIR="$2"; shift 2 ;;
        --move-to) MOVE_TO="$2"; shift 2 ;;
        --nested-folder) NESTED_FOLDER="$2"; shift 2 ;;
        --src-subdir)
            SRC_SUBDIR="$2"
            # Split on spaces to support quoted multi-path
            read -r -a split_subdirs <<<"$2"
            SRC_SUBDIRS+=("${split_subdirs[@]}")
            shift 2
            ;;
        --src-subdirs)
            shift
            # Collect all following args until another --option or end
            subdirs=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                subdirs+=("$1")
                shift
            done
            # Flatten subdirs in case a single quoted string with spaces was passed
            for s in "${subdirs[@]}"; do
                read -r -a parts <<<"$s"
                SRC_SUBDIRS+=("${parts[@]}")
            done
            ;;
        --readme-name) README_NAME="$2"; shift 2 ;;
        --check) MODE="check"; shift ;;
        --update-hash) MODE="update_hash"; shift ;;
        --ensure-init-py|-i) ENSURE_INIT_PY="true"; shift ;;
        --future-annotations|-A) FUTURE_ANNOTATIONS="true"; shift ;;
        --scope-src-subdirs|-S) SCOPE_SRC_SUBDIRS="true"; shift ;;
        --dry-run|-n) DRY_RUN="true"; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

#######################################
# VALIDATE INPUTS
#######################################
[[ -z "$TARGET_DIR" ]] && { echo "❌ --target-dir required."; exit 1; }
TARGET_DIR=$(realpath "$TARGET_DIR")
LOCK_FILE="$TARGET_DIR/vendor.lock.json"
README_FILE="$TARGET_DIR/$README_NAME"
# Determine which path to reference in README instructions
FINAL_TARGET="$TARGET_DIR"  # after possible move

# ---------------------------------------------------------------------
# Copy phase (used later after git clone)
# ---------------------------------------------------------------------
# Globs are resolved by find "$TMP_DIR" -path "$TMP_DIR/$pattern",
# which prevents path traversal (e.g., ../etc/passwd won’t work).
copy_src_paths() {
    local tmp_dir="$1"
    local target_dir="$2"

    if [[ ${#SRC_SUBDIRS[@]} -eq 0 ]]; then
        echo "📦  Copying entire repository..."
        cp -a --preserve=timestamps,mode "$tmp_dir"/. "$target_dir/"
        return
    fi

    echo "📂  Copying specific paths:"
    for pattern in "${SRC_SUBDIRS[@]}"; do
        # Expand globs safely *inside* tmp_dir
        local matches=()
        while IFS= read -r -d '' path; do
            matches+=("$path")
        done < <(find "$tmp_dir" -path "$tmp_dir/$pattern" -print0 2>/dev/null || true)

        if [[ ${#matches[@]} -eq 0 ]]; then
            echo "⚠️  No matches for pattern '$pattern'"
            continue
        fi

        for src in "${matches[@]}"; do
            relpath="${src#$tmp_dir/}"
            mkdir -p "$target_dir/$(dirname "$relpath")"
            # ✅ Preserve timestamps & permissions
            cp -a --preserve=timestamps,mode "$src" "$target_dir/$relpath"
            echo "   - Copied: $relpath"
        done
    done
}

# ---------------------------------------------------------------------
# SCOPING: resolve which paths --ensure-init-py/--future-annotations
# should actually walk, honoring --scope-src-subdirs.
# ---------------------------------------------------------------------
# Default (SCOPE_SRC_SUBDIRS=false, or no --src-subdirs given this run):
# the whole $FINAL_TARGET, i.e. today's existing behavior - unchanged.
#
# With --scope-src-subdirs and a non-empty SRC_SUBDIRS: each requested
# entry is re-resolved against the *actual, current* $FINAL_TARGET (not
# replayed from copy-time bookkeeping), so it stays correct regardless of
# --nested-folder/--move-to having relocated things:
#   - entry == NESTED_FOLDER            -> whole $FINAL_TARGET
#   - entry starts with "NESTED_FOLDER/" -> that prefix is stripped, since
#                                           the move already stripped it
#                                           on disk
#   - otherwise (no --nested-folder, or entry wasn't under it)
#                                        -> matched as-is via the same
#                                           glob-aware find used at copy
#                                           time, so patterns still work
# Entries that resolve to nothing are reported and simply contribute no
# root (rather than failing the whole run).
# ---------------------------------------------------------------------
compute_scope_roots() {
    local final_target="$1"

    if [[ "$SCOPE_SRC_SUBDIRS" != "true" || ${#SRC_SUBDIRS[@]} -eq 0 ]]; then
        if [[ "$SCOPE_SRC_SUBDIRS" == "true" ]]; then
            echo "ℹ️   --scope-src-subdirs had no --src-subdirs to work from in this invocation; using the whole target." >&2
        fi
        printf '%s\0' "$final_target"
        return
    fi

    local roots=()
    for sub in "${SRC_SUBDIRS[@]}"; do
        local effective_sub="$sub"
        if [[ -n "$NESTED_FOLDER" ]]; then
            if [[ "$sub" == "$NESTED_FOLDER" ]]; then
                roots+=("$final_target")
                continue
            elif [[ "$sub" == "$NESTED_FOLDER"/* ]]; then
                effective_sub="${sub#"$NESTED_FOLDER"/}"
            fi
            # else: this entry wasn't under --nested-folder, so it did not
            # move to $final_target with it; fall through and try matching
            # it as-is (covers the "no move happened" / mixed-use cases).
        fi

        local matched=0
        while IFS= read -r -d '' path; do
            roots+=("$path")
            matched=1
        done < <(find "$final_target" -path "$final_target/$effective_sub" -print0 2>/dev/null || true)
        if [[ "$matched" -eq 0 ]]; then
            echo "⚠️   --scope-src-subdirs: '$sub' has no match under $final_target; skipping it for scoping." >&2
        fi
    done

    if [[ ${#roots[@]} -eq 0 ]]; then
        echo "⚠️   --scope-src-subdirs matched nothing at all; falling back to the whole target." >&2
        printf '%s\0' "$final_target"
    else
        printf '%s\0' "${roots[@]}"
    fi
}

# ---------------------------------------------------------------------
# MAINTENANCE: Ensure every nested folder has an __init__.py
# ---------------------------------------------------------------------
# Never modifies or recreates an existing __init__.py (even an empty one
# that already exists is left untouched) - only creates it when missing.
# Hidden directories (.git, .tmp, ...) and __pycache__ are skipped so we
# never write into VCS/cache internals. Accepts one or more root paths
# (see compute_scope_roots above for how those are chosen).
# ---------------------------------------------------------------------
ensure_init_py() {
    local created=0
    local present=0

    log "INFO" "🧩  Ensuring __init__.py under: $*"

    for root in "$@"; do
        [[ -d "$root" ]] || continue
        while IFS= read -r -d '' dir; do
            local init_file="$dir/__init__.py"
            if [[ -e "$init_file" ]]; then
                present=$((present + 1))
                continue
            fi
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "   + [dry-run] Would create: ${init_file#$root/}"
            else
                : > "$init_file"
                echo "   + Created: ${init_file#$root/}"
            fi
            created=$((created + 1))
        done < <(find "$root" \
                    -type d \( -name '.git' -o -name '.tmp' -o -name '__pycache__' -o -name '.*' \) -prune \
                    -o -type d -print0)
    done

    log "INFO" "✅  __init__.py check complete (created: $created, already present: $present)."
}

# ---------------------------------------------------------------------
# MAINTENANCE: Ensure `from __future__ import annotations` is present
# ---------------------------------------------------------------------
#   - Skipped entirely for empty files, and for files with no import
#     statement anywhere in them (nothing to future-annotate).
#   - Skipped (left untouched) if the import is already present anywhere
#     in the file - inserted at most once, ever.
#   - Skipped (left untouched, reported) if the file does not parse as
#     valid Python - never guess at a syntax fix.
#   - Inserted as the first statement after any shebang / leading
#     comments / module docstring, which is also the only syntactically
#     legal place for a `__future__` import in Python - so it can never
#     land inside a class, function, or method body.
#   - Surrounded by exactly one blank line on each side; pre-existing
#     blank lines at the insertion point are collapsed first so re-runs
#     stay idempotent (no growing whitespace on repeated invocations).
# Accepts one or more root paths (see compute_scope_roots above for how
# those are chosen when --scope-src-subdirs narrows the walk).
# ---------------------------------------------------------------------
ensure_future_annotations() {
    log "INFO" "🧬  Ensuring 'from __future__ import annotations' under: $*"
    python - "$DRY_RUN" "$@" <<'PYEOF'
import ast
import os
import re
import sys

DRY_RUN, ROOTS = sys.argv[1] == "true", sys.argv[2:]
SKIP_DIRS = {".git", ".tmp", "__pycache__", ".hg", ".svn"}
FUTURE_RE = re.compile(r"^[ \t]*from[ \t]+__future__[ \t]+import[ \t]+.*\bannotations\b")


def has_future_annotations(source: str) -> bool:
    return any(FUTURE_RE.match(line) for line in source.splitlines())


def has_any_import(tree: ast.Module) -> bool:
    return any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))


def insertion_row(tree: ast.Module) -> int:
    """0-indexed line count of the module preamble. Shebang and comments
    are not part of the AST at all; only a leading docstring (the sole
    statement Python allows before a __future__ import) extends it."""
    if not tree.body:
        return 0
    first = tree.body[0]
    is_docstring = (
        isinstance(first, ast.Expr)
        and isinstance(getattr(first, "value", None), ast.Constant)
        and isinstance(first.value.value, str)
    )
    return first.end_lineno if is_docstring else first.lineno - 1


def process(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
        encoding = "utf-8"
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as fh:
            source = fh.read()
        encoding = "latin-1"

    if not source.strip():
        return "empty"
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "syntax-error"
    if not has_any_import(tree):
        return "no-import"
    if has_future_annotations(source):
        return "already-present"

    lines = source.splitlines()
    row = insertion_row(tree)
    prefix, suffix = lines[:row], lines[row:]
    while prefix and prefix[-1].strip() == "":
        prefix.pop()
    while suffix and suffix[0].strip() == "":
        suffix.pop(0)
    new_source = "\n".join(prefix + ["", "from __future__ import annotations", ""] + suffix)
    if not new_source.endswith("\n"):
        new_source += "\n"

    if not DRY_RUN:
        with open(path, "w", encoding=encoding, newline="") as fh:
            fh.write(new_source)
    return "updated"


def iter_py_files(root: str):
    """Yield .py file paths under `root` (or `root` itself if it already
    names a .py file - a --src-subdirs entry can be a single file)."""
    if os.path.isfile(root):
        if root.endswith(".py"):
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def main() -> int:
    counts = {}
    seen = set()  # de-dup in case scope roots overlap/nest
    for root in ROOTS:
        display_base = root if os.path.isdir(root) else (os.path.dirname(root) or ".")
        for full in iter_py_files(root):
            full = os.path.realpath(full)
            if full in seen:
                continue
            seen.add(full)
            rel = os.path.relpath(full, display_base)
            try:
                status = process(full)
            except Exception as exc:  # one bad file must never abort the whole tree
                status = "error"
                print(f"   ! Skipped ({exc.__class__.__name__}: {exc}): {rel}", file=sys.stderr)
            counts[status] = counts.get(status, 0) + 1
            if status == "updated":
                tag = "[dry-run] Would update" if DRY_RUN else "Updated"
                print(f"   + {tag}: {rel}")
            elif status == "syntax-error":
                print(f"   ! Skipped (does not parse, left untouched): {rel}", file=sys.stderr)

    print(
        "[INFO] future-annotations summary: "
        f"updated={counts.get('updated', 0)} "
        f"already_present={counts.get('already-present', 0)} "
        f"no_import={counts.get('no-import', 0)} "
        f"empty={counts.get('empty', 0)} "
        f"syntax_error={counts.get('syntax-error', 0)} "
        f"error={counts.get('error', 0)}"
    )
    return 0


sys.exit(main())
PYEOF
}

#######################################
# Integrity check mode
#######################################
if [[ "$MODE" == "check" ]]; then
    if [[ "$ENSURE_INIT_PY" == "true" || "$FUTURE_ANNOTATIONS" == "true" ]]; then
        echo "⚠️  --ensure-init-py/--future-annotations are ignored under --check (check stays read-only)." >&2
    fi
    echo "🔍  Running integrity check on $TARGET_DIR..."
    if [[ ! -f "$LOCK_FILE" ]]; then
        echo "❌  No vendor.lock.json found; cannot verify."
        exit 1
    fi

    # robust extraction
    if command -v jq >/dev/null 2>&1; then
        EXPECTED_HASH=$(jq -r '.tree_hash' "$LOCK_FILE")
    else
        echo "⚠  jq not found, using fallback JSON parser or python"
        EXPECTED_HASH=$(python -c "import json,sys; print(json.load(open('$LOCK_FILE'))['tree_hash'])")
    fi

    # Compute actual tree hash (mode + hash)
    read -r ACTUAL_MODE ACTUAL_HASH <<<"$(compute_tree_hash "$TARGET_DIR")"

    echo "🔍 Verification mode: $ACTUAL_MODE"
    if [[ "$EXPECTED_HASH" == "$ACTUAL_HASH" ]]; then
        echo "✅  Verified: Tree hash matches ($EXPECTED_HASH)"
        exit 0
    else
        echo "❌  Drift detected!"
        echo "   • Expected: $EXPECTED_HASH"
        echo "   •   Actual: $ACTUAL_HASH"
        echo "   •    Mode : $ACTUAL_MODE"
        exit 2
    fi
fi

#######################################
# Update only the tree hash (no git clone)
#######################################
# bash ./tools/maint_tools/vendor_repo.sh \
#   --target-dir "../scikitplot/cexternals/NumCpp" \
#   --update-hash

# Escape a string for safe use in a sed replacement (handles / and &)
sed_repl_escape() {
  printf '%s' "$1" | sed 's/[\/&]/\\&/g';
}

# ---------------------------------------------------------------------
# Write recomputed tree_mode/tree_hash (+ timestamp) into vendor.lock.json
# and README.md. Shared by --update-hash and standalone maintenance mode
# so the two never drift apart.
# ---------------------------------------------------------------------
update_lock_and_readme_hash() {
    local new_mode="$1"
    local new_hash="$2"
    local now
    now="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

    if command -v jq >/dev/null 2>&1; then
        local tmpfile
        tmpfile=$(mktemp)
        jq --arg mode "$new_mode" --arg hash "$new_hash" --arg now "$now" \
           '.tree_mode=$mode | .tree_hash=$hash | .generated_utc=$now' \
           "$LOCK_FILE" >"$tmpfile" && mv "$tmpfile" "$LOCK_FILE"
    else
        python - "$LOCK_FILE" "$new_mode" "$new_hash" <<'EOF'
import json, sys, datetime, tempfile, os
path, mode, h = sys.argv[1:4]
data = json.load(open(path))
data["tree_mode"] = mode
data["tree_hash"] = h
data["generated_utc"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
tmp = tempfile.NamedTemporaryFile('w', delete=False)
json.dump(data, tmp, indent=2)
tmp.close()
os.replace(tmp.name, path)
EOF
    fi

    # Update README.md if it exists
    if [ -f "$README_FILE" ]; then
      local new_mode_esc new_hash_esc now_esc
      new_mode_esc="$(sed_repl_escape "$new_mode")"
      new_hash_esc="$(sed_repl_escape "$new_hash")"
      now_esc="$(sed_repl_escape "$now")"

      # --- New table format ---
      # --- Legacy bullet format (if still present) ---
      sed -i.bak -E \
        \
        -e "s/^(\|[[:space:]]*Tree Mode[^|]*\|)[[:space:]]*[^|]*(\|\|[[:space:]]*)$/\1 $new_mode_esc \2/" \
        -e "s/^(\|[[:space:]]*Tree Hash[^|]*\|)[[:space:]]*[^|]*(\|\|[[:space:]]*)$/\1 $new_hash_esc \2/" \
        -e "s/^(\|[[:space:]]*Retrieved[^|]*\|)[[:space:]]*[^|]*(\|\|[[:space:]]*)$/\1 $now_esc \2/" \
        \
        -e "s/^(- Tree Mode:).*/\1  $new_mode_esc/" \
        -e "s/^(- Tree Hash:).*/\1  $new_hash_esc/" \
        -e "s/^(- Retrieved:).*/\1  $now_esc/" \
        \
        "$README_FILE" && rm -f "$README_FILE.bak"

      echo "📘 Updated README.md with new hash."
    fi
}

if [ "$MODE" = "update_hash" ]; then
    echo "🔁  Recomputing tree hash for $TARGET_DIR..."

    if [ ! -f "$LOCK_FILE" ]; then
        echo "❌  No vendor.lock.json found; cannot update hash."
        exit 1
    fi

    if [[ "$ENSURE_INIT_PY" == "true" || "$FUTURE_ANNOTATIONS" == "true" ]]; then
        mapfile -d '' -t scope_roots < <(compute_scope_roots "$TARGET_DIR")
        [[ "$ENSURE_INIT_PY" == "true" ]] && ensure_init_py "${scope_roots[@]}"
        [[ "$FUTURE_ANNOTATIONS" == "true" ]] && ensure_future_annotations "${scope_roots[@]}"
    fi

    read -r NEW_MODE NEW_HASH <<<"$(compute_tree_hash "$TARGET_DIR")"
    echo "🔐  New Tree Hash: $NEW_HASH ($NEW_MODE)"

    update_lock_and_readme_hash "$NEW_MODE" "$NEW_HASH"

    echo "✅  Updated vendor.lock.json and $README_NAME with recomputed hash."
    exit 0
fi

#######################################
# Maintenance-only mode (no clone): apply --ensure-init-py / --future-annotations
# to an already-vendored tree, then refresh the lock file + README if present.
#######################################
if [[ "$MODE" == "update" && -z "$REPO_URL" && -z "$REPO_REF" \
      && ( "$ENSURE_INIT_PY" == "true" || "$FUTURE_ANNOTATIONS" == "true" ) ]]; then
    echo "🛠️   Maintenance mode (no clone) on $TARGET_DIR..."

    mapfile -d '' -t scope_roots < <(compute_scope_roots "$TARGET_DIR")
    [[ "$ENSURE_INIT_PY" == "true" ]] && ensure_init_py "${scope_roots[@]}"
    [[ "$FUTURE_ANNOTATIONS" == "true" ]] && ensure_future_annotations "${scope_roots[@]}"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "ℹ️   Dry run: no files were written; lock file/README left untouched."
        exit 0
    fi

    if [[ -f "$LOCK_FILE" ]]; then
        echo "🔁  Refreshing tree hash after maintenance..."
        read -r NEW_MODE NEW_HASH <<<"$(compute_tree_hash "$TARGET_DIR")"
        echo "🔐  New Tree Hash: $NEW_HASH ($NEW_MODE)"
        update_lock_and_readme_hash "$NEW_MODE" "$NEW_HASH"
    else
        echo "ℹ️   No vendor.lock.json found under $TARGET_DIR; skipping hash refresh."
    fi

    echo "✅  Maintenance complete."
    exit 0
fi

#######################################
# Update (vendoring) mode
#######################################
# --- Step 1: Validate Inputs ---
[[ -z "$REPO_URL" ]] && { echo "❌  --repo-url required."; exit 1; }
[[ -z "$REPO_REF" ]] && { echo "❌  --repo-ref required."; exit 1; }

# ---------------------------------------------------------------------
# Function: Determine if provided ref appears to be a commit hash
# ---------------------------------------------------------------------
is_commit_hash() {
  [[ "$REPO_REF" =~ ^[a-fA-F0-9]{7,40}$ ]]
}
# ---------------------------------------------------------------------
# STEP 1: Reference validation (branches/tags only exists remotely)
# We SKIP this step if ref is a commit hash because commits are not listed in tags/heads.
# ---------------------------------------------------------------------
ref_exists_remotely() {
  git ls-remote --exit-code --tags "$REPO_URL" "refs/tags/$REPO_REF" >/dev/null 2>&1 ||
  git ls-remote --exit-code --heads "$REPO_URL" "$REPO_REF" >/dev/null 2>&1
}
if ! is_commit_hash; then
  if ! git ls-remote --exit-code --tags "$REPO_URL" "refs/tags/$REPO_REF" >/dev/null 2>&1; then
    if ! git ls-remote --exit-code --heads "$REPO_URL" "$REPO_REF" >/dev/null 2>&1; then
        echo "❌  Repository ref '$REPO_REF' is not a tag or branch in $REPO_URL"
        echo "ℹ️  If this is a commit hash, please confirm it exists."
        exit 1
    fi
  fi
else
  echo "ℹ️  Detected commit hash: skipping tag/branch remote validation."
fi

TMP_DIR="$TARGET_DIR/.tmp"
rm -rf "$TMP_DIR" "$TARGET_DIR"
mkdir -p "$TARGET_DIR" "$TMP_DIR"

# ---------------------------------------------------------------------
# CLONE DEFAULT BRANCH (LATEST)
# ---------------------------------------------------------------------
clone_default_branch() {
  log "INFO" "Cloning default branch (shallow)"
  git clone --depth 1 "$REPO_URL" "$TMP_DIR"
}

# ---------------------------------------------------------------------
# CLONE BY SPECIFIC COMMIT
# ---------------------------------------------------------------------
clone_specific_commit() {
  log "INFO" "Cloning specific commit: $REPO_REF"
  # Initialize repo with main as default branch
  git init -b main "$TMP_DIR"
  pushd "$TMP_DIR" >/dev/null
  git remote add origin "$REPO_URL"
  # the entire repository contents as they were at that commit.
  git fetch --depth 1 origin "$REPO_REF"   # FETCHES ONLY FILES AT THAT COMMIT
  # Checkout commit in detached HEAD
  git checkout "$REPO_REF"                 # CHECKS OUT SNAPSHOT OF THAT COMMIT
  popd >/dev/null
  log "INFO" "✅  Checked out commit $REPO_REF in detached HEAD mode on branch 'main'."
}

# ---------------------------------------------------------------------
# CLONE BY BRANCH OR TAG
# ---------------------------------------------------------------------
clone_branch_or_tag() {
  log "INFO" "Cloning branch/tag: $REPO_REF"
  # entire tree at that point
  git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$TMP_DIR"
}

# ---------------------------------------------------------------------
# STEP 2: Execute appropriate clone mode
# ---------------------------------------------------------------------
if [[ -z "$REPO_REF" ]]; then
  clone_default_branch
elif is_commit_hash; then
  clone_specific_commit
else
  if ! ref_exists_remotely; then
    echo "❌  Ref '$REPO_REF' not found in $REPO_URL"
    exit 1
  fi
  clone_branch_or_tag
fi

#######################################
# FINAL CONFIRMATION
#######################################
pushd "$TMP_DIR" >/dev/null
HASH=$(git rev-parse HEAD)
popd >/dev/null

log "SUCCESS" "Repository successfully checked out to: $TMP_DIR"
echo "📦  Checked out commit $HASH from $REPO_URL"

# --- Step 3: Move or Copy files exactly-deterministically ---
if [[ ${#SRC_SUBDIRS[@]} -eq 0 ]]; then
    echo "📦  Copying entire repository..."
    cp -a --preserve=timestamps,mode "$TMP_DIR"/. "$TARGET_DIR/"
else
    # Verify all requested paths exist before copying. Uses the same
    # pattern-aware `find -path` matching as copy_src_paths below, so a
    # legitimate glob entry can never fail here while still matching there.
    for sub in "${SRC_SUBDIRS[@]}"; do
        if ! find "$TMP_DIR" -path "$TMP_DIR/$sub" -print -quit 2>/dev/null | grep -q .; then
            echo "❌  Path '$sub' not found in repo." >&2
            exit 1
        fi
    done
    copy_src_paths "$TMP_DIR" "$TARGET_DIR"
fi

# Copy LICENSE files, under ifdef NESTED_FOLDER
if [[ -n "$NESTED_FOLDER" ]]; then
    LICENSE_TARGET="$TARGET_DIR/$NESTED_FOLDER"
else
    LICENSE_TARGET="$TARGET_DIR"
fi
cp -a --preserve=timestamps,mode "$TMP_DIR"/LICENSE* "$LICENSE_TARGET/" 2>/dev/null || true
rm -rf "$TMP_DIR"

# --- Step 3b: Move if requested (with safety check) ---
if [[ -n "${MOVE_TO:-}" ]]; then
    MOVE_TO=$(realpath "$MOVE_TO")
    echo "📦  Moving vendored content from $TARGET_DIR to $MOVE_TO ..."

    rm -rf "$MOVE_TO"
    mkdir -p "$(dirname "$MOVE_TO")"

    if [[ -n "$NESTED_FOLDER" ]]; then
        NESTED_PATH="$TARGET_DIR/$NESTED_FOLDER"
        # Safety check: ensure nested folder is within target
        if [[ "$NESTED_PATH" != "$TARGET_DIR"* ]]; then
            echo "❌  Error: --nested-folder '$NESTED_FOLDER' points outside target!"
            exit 1
        fi
        if [[ -d "$NESTED_PATH" ]]; then
            # Move only the nested folder
            mv "$NESTED_PATH" "$MOVE_TO"
            # Remove empty parent directories if needed
            rmdir --ignore-fail-on-non-empty "$TARGET_DIR" 2>/dev/null || true
        else
            echo "❌  Nested folder '$NESTED_FOLDER' not found in target."
            exit 1
        fi
    else
        # Move entire target folder
        mv "$TARGET_DIR" "$MOVE_TO"
    fi

    # Update TARGET_DIR path for following steps (README, tree hash)
    LOCK_FILE="$MOVE_TO/vendor.lock.json"
    README_FILE="$MOVE_TO/$README_NAME"
    FINAL_TARGET="$MOVE_TO"  # after possible move
fi

# --- Step 3c: Maintenance passes on the final vendored tree (optional) ---
if [[ "$ENSURE_INIT_PY" == "true" || "$FUTURE_ANNOTATIONS" == "true" ]]; then
    mapfile -d '' -t scope_roots < <(compute_scope_roots "$FINAL_TARGET")
    [[ "$ENSURE_INIT_PY" == "true" ]] && ensure_init_py "${scope_roots[@]}"
    [[ "$FUTURE_ANNOTATIONS" == "true" ]] && ensure_future_annotations "${scope_roots[@]}"
fi

# --- Step 4: Compute SHA256 fingerprint-hash of the vendored tree ---
read -r TREE_MODE TREE_HASH <<<"$(compute_tree_hash "$FINAL_TARGET")"

# --- Step 5: Save-Write metadata lockfile ---
cat <<EOF > "$LOCK_FILE"
{
  "repository": "$REPO_URL",
  "version": "$REPO_REF",
  "commit_hash": "$HASH",
  "tree_mode": "$TREE_MODE",
  "tree_hash": "$TREE_HASH",
  "generated_utc": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
}
EOF

code_block() {
  local code="$1"
  printf '```bash\n%s\n```\n' "$code"
}
code_block_p() {
  local code="$1"
  printf '~~~bash\n%s\n~~~\n' "$code"
}

# --- Step 5: Record provenance exactly README.md ---
# Build the "how to reproduce" command, only including flags actually in use
# (an unconditional --move-to/--nested-folder/--src-subdirs with an empty
# value produced misleading copy-paste instructions before).
EXTRA_FLAGS=""
[[ "$ENSURE_INIT_PY" == "true" ]] && EXTRA_FLAGS+=" --ensure-init-py"
[[ "$FUTURE_ANNOTATIONS" == "true" ]] && EXTRA_FLAGS+=" --future-annotations"
[[ "$SCOPE_SRC_SUBDIRS" == "true" ]] && EXTRA_FLAGS+=" --scope-src-subdirs"

REPRO_CMD="bash ./tools/maint_tools/vendor_repo.sh \\
  --repo-url $REPO_URL \\
  --repo-ref $REPO_REF \\
  --target-dir $TARGET_DIR"
[[ -n "$MOVE_TO" ]] && REPRO_CMD+=" \\
  --move-to $MOVE_TO"
[[ -n "$NESTED_FOLDER" ]] && REPRO_CMD+=" \\
  --nested-folder $NESTED_FOLDER"
if [[ ${#SRC_SUBDIRS[@]} -gt 0 ]]; then
  REPRO_CMD+=" \\
  --src-subdirs ${SRC_SUBDIRS[*]}"
fi
REPRO_CMD+=" \\
  --readme-name $README_NAME${EXTRA_FLAGS}"

cat <<EOF > "$README_FILE"
Vendored repository information
===============================

|     |     |     |
| --: | :-- | --- |
| Repository (Remote Git repo URL)         : | $REPO_URL ||
| Version (Ref Branch, Tag, or Commit SHA) : | $REPO_REF ||
| Commit                                   : | $HASH ||
| Tree Mode                                : | $TREE_MODE ||
| Tree Hash                                : | $TREE_HASH ||
| Retrieved                                : | $(date -u +'%Y-%m-%dT%H:%M:%SZ') ||

To update (git clone), run:

$(code_block_p "$REPRO_CMD")

To update only the tree hash (no git clone):

$(code_block_p "bash ./tools/maint_tools/vendor_repo.sh \\
  --target-dir $FINAL_TARGET \\
  --update-hash")

To add missing __init__.py files and/or future-annotations imports without re-cloning:

$(code_block_p "bash ./tools/maint_tools/vendor_repo.sh \\
  --target-dir $FINAL_TARGET \\
  --ensure-init-py \\
  --future-annotations")

To verify in CI:

$(code_block_p "bash ./tools/maint_tools/vendor_repo.sh --target-dir $FINAL_TARGET --check")

$(code_block_p "# python ./tools/maint_tools/verify_vendor.py "./scikitplot/"
python ./tools/maint_tools/verify_vendor.py \"$FINAL_TARGET\"  # --json --pretty")
EOF

echo "✅  Vendoring complete (commit: $HASH)"
echo "🔐  Integrity fingerprint: $TREE_HASH"
