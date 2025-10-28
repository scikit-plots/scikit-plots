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
    local excludes="( -name vendor.lock.json -o -name README.md -o -name .gitignore ) -prune -o"
    local EXCLUDES=("vendor.lock.json" "README.md" ".gitignore")
    local exclude_expr=()
    for f in "${EXCLUDES[@]}"; do
        exclude_expr+=(-not -name "$f")
    done
    local mode hash
    local dir="$1"

    # Try Bash + sha256sum pipeline first
    # ACTUAL_HASH=$(git -C "$TARGET_DIR" rev-parse HEAD 2>/dev/null || echo "none")
    # ACTUAL_HASH=$(find "$TARGET_DIR" -type f -exec sha256sum {} \; | sort | sha256sum | awk '{print $1}')
    if command -v sha256sum >/dev/null 2>&1 && command -v find >/dev/null 2>&1; then
        local mode="bash-sha256sum"
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
        # Fallback Compute SHA256 of all files for integrity verification
        local mode="python-hashlib"
        echo "⚙️  Falling back to $mode mode for tree hash..." >&2
        local hash
        # hash=$(python - "$dir" <<'EOF'
        hash=$(python - <<'EOF' "$dir"
import hashlib, os, sys
root = sys.argv[1]
exclude = {"vendor.lock.json", "README.md", ".gitignore"}
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
    echo "Usage: $0 --repo-url REPO_URL --repo-ref TAG --target-dir PATH [--src-subdir SUBDIR] [--readme-name NAME] [--check]"
}
MODE="${REPO_URL:-"update"}"      # default
REPO_URL="${REPO_URL:-""}"        # Remote Git repo URL
REPO_REF="${REPO_REF:-""}"        # Ref Branch, Tag, or Commit SHA
TARGET_DIR="${TARGET_DIR:-""}"    # Directory to clone into
SRC_SUBDIR="${SRC_SUBDIR:-"."}"   # legacy single subdir (for backward compatibility)
SRC_SUBDIRS=()                    # list of subdirs/files (new, plural form)
README_NAME="vendor_repo.sh"
MOVE_TO=""                        # optional move, default: do not move
# Optional nested folder name inside target to move
# --nested-folder "astropy" means only move $TARGET_DIR/astropy → MOVE_TO
NESTED_FOLDER=""  # optional nested folder to move

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
        --update-hash) MODE="update_hash"; shift ;;   # 👈 NEW
        --help|-h)
            # echo "Usage: $0 --repo-url REPO_URL --repo-ref TAG --target-dir PATH [--src-subdir SUBDIR] [--readme-name NAME] [--check]"
            usage;
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

#######################################
# VALIDATE INPUTS
#######################################
[[ -z "$TARGET_DIR" ]] && { echo "❌ --target-dir required."; exit 1; }
# TARGET_DIR=$(basename "$REPO_URL" .git)
TARGET_DIR=$(realpath "$TARGET_DIR")
LOCK_FILE="$TARGET_DIR/vendor.lock.json"
README_FILE="$TARGET_DIR/README.md"
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
        echo "📦 Copying entire repository..."
        # cp -a "$tmp_dir"/. "$target_dir/"
        # cp -a --preserve=all "$src" "$target_dir/$relpath"
        cp -a --preserve=timestamps,mode "$tmp_dir"/. "$target_dir/"
        return
    fi

    echo "📂 Copying specific paths:"
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
            # cp -a "$src" "$target_dir/$relpath"
            # ✅ Preserve timestamps & permissions
            cp -a --preserve=timestamps,mode "$src" "$target_dir/$relpath"
            echo "   - Copied: $relpath"
        done
    done
}

#######################################
# Integrity check mode
#######################################
# echo "$MODE"
if [[ "$MODE" == "check" ]]; then
    echo "🔍 Running integrity check on $TARGET_DIR..."
    # [[ -f "$LOCK_FILE" ]] || { echo "❌ No vendor.lock.json found; cannot verify."; exit 1; }
    if [[ ! -f "$LOCK_FILE" ]]; then
        echo "❌ No vendor.lock.json found; cannot verify."
        exit 1
    fi

    # robust extraction
    if command -v jq >/dev/null 2>&1; then
        # EXPECTED_HASH=$(jq -r '.commit_hash' "$LOCK_FILE")
        EXPECTED_HASH=$(jq -r '.tree_hash' "$LOCK_FILE")
    else
        echo "⚠ jq not found, using fallback JSON parser or python"
        # EXPECTED_HASH=$(grep -o '"commit_hash": *"[^"]*"' "$LOCK_FILE" | sed -E 's/.*"commit_hash": *"([^"]*)".*/\1/')
        EXPECTED_HASH=$(python -c "import json,sys; print(json.load(open('$LOCK_FILE'))['tree_hash'])")
    fi

    # Compute actual tree hash (mode + hash)
    # read -r ACTUAL_MODE ACTUAL_HASH < <(compute_tree_hash "$TARGET_DIR")
    read -r ACTUAL_MODE ACTUAL_HASH <<<"$(compute_tree_hash "$TARGET_DIR")"

    echo "🔍 Verification mode: $ACTUAL_MODE"
    if [[ "$EXPECTED_HASH" == "$ACTUAL_HASH" ]]; then
        # echo "✅ Verified: Commit hash matches ($EXPECTED_HASH)"
        echo "✅ Verified: Tree hash matches ($EXPECTED_HASH)"
        exit 0
    else
        echo "❌ Drift detected!"
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
if [ "$MODE" = "update_hash" ]; then
    echo "🔁 Recomputing tree hash for $TARGET_DIR..."

    # [[ -f "$LOCK_FILE" ]] || { echo "❌ No vendor.lock.json found; cannot update hash."; exit 1; }
    if [ ! -f "$LOCK_FILE" ]; then
        echo "❌ No vendor.lock.json found; cannot update hash."
        exit 1
    fi

    read -r NEW_MODE NEW_HASH <<<"$(compute_tree_hash "$TARGET_DIR")"
    echo "🔐 New Tree Hash: $NEW_HASH ($NEW_MODE)"

    if command -v jq >/dev/null 2>&1; then
        tmpfile=$(mktemp)
        jq --arg mode "$NEW_MODE" --arg hash "$NEW_HASH" \
           '.tree_mode=$mode | .tree_hash=$hash | .generated_utc="'$(date -u +'%Y-%m-%dT%H:%M:%SZ')'"' \
           "$LOCK_FILE" >"$tmpfile" && mv "$tmpfile" "$LOCK_FILE"
    else
        python - "$LOCK_FILE" "$NEW_MODE" "$NEW_HASH" <<'EOF'
import json, sys, datetime, tempfile, os
path, mode, h = sys.argv[1:4]
data = json.load(open(path))
data["tree_mode"] = mode
data["tree_hash"] = h
data["generated_utc"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")  # '%Y-%m-%dT%H:%M:%SZ'
tmp = tempfile.NamedTemporaryFile('w', delete=False)
json.dump(data, tmp, indent=2)
tmp.close()
os.replace(tmp.name, path)
EOF
    fi

    # Update README.md if exists
    if [ -f "$TARGET_DIR/README.md" ]; then
        sed -i.bak -E \
            -e "s/^(- Tree Mode:).*/\1  $NEW_MODE/" \
            -e "s/^(- Tree Hash:).*/\1  $NEW_HASH/" \
            -e "s/^(- Retrieved:).*/\1  $(date -u +'%Y-%m-%dT%H:%M:%SZ')/" \
            "$TARGET_DIR/README.md" && rm -f "$TARGET_DIR/README.md.bak"
        echo "📘 Updated README.md with new hash."
    fi

    echo "✅ Updated vendor.lock.json and README.md with recomputed hash."
    exit 0
fi

#######################################
# Update (vendoring) mode
#######################################
# --- Step 1: Validate Inputs ---
# [[ -z "$REPO_URL" || -z "$REPO_REF" ]] && { echo "❌ --repo-url and --repo-ref required for update mode."; exit 1; }
[[ -z "$REPO_URL" ]] && { echo "❌ --repo-url required."; exit 1; }
[[ -z "$REPO_REF" ]] && { echo "❌ --repo-ref required."; exit 1; }

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
        echo "❌ Repository ref '$REPO_REF' is not a tag or branch in $REPO_URL"
        echo "ℹ️  If this is a commit hash, please confirm it exists."
        exit 1
    fi
  fi
else
  echo "ℹ️ Detected commit hash: skipping tag/branch remote validation."
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
  # git init "$TMP_DIR"
  git init -b main "$TMP_DIR"
  # git -C "$TMP_DIR" remote add origin "$REPO_URL"
  # git -C "$TMP_DIR" fetch --depth 1 origin "$REPO_REF"
  # git -C "$TMP_DIR" checkout "$REPO_REF" || { echo "❌ Commit $REPO_REF not found."; exit 1; }
  # cd "$TMP_DIR"
  pushd "$TMP_DIR" >/dev/null
  git remote add origin "$REPO_URL"
  # the entire repository contents as they were at that commit.
  git fetch --depth 1 origin "$REPO_REF"   # FETCHES ONLY FILES AT THAT COMMIT
  # Checkout commit in detached HEAD
  git checkout "$REPO_REF"                 # CHECKS OUT SNAPSHOT OF THAT COMMIT
  popd >/dev/null
  log "INFO" "✅ Checked out commit $REPO_REF in detached HEAD mode on branch 'main'."
}

# ---------------------------------------------------------------------
# CLONE BY BRANCH OR TAG
# ---------------------------------------------------------------------
clone_branch_or_tag() {
  log "INFO" "Cloning branch/tag: $REPO_REF"
  # entire tree at that point
  git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$TMP_DIR"
}

# --- Step 2: Clone and get exact commit hash ---
# --depth 1 → shallow clone (faster, minimal history)
# --branch tag_or_branch -> only supports branch names or tags — not commit hashes.
# git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$TMP_DIR"

# ---------------------------------------------------------------------
# STEP 2: Execute appropriate clone mode
# ---------------------------------------------------------------------
if [[ -z "$REPO_REF" ]]; then
  clone_default_branch
elif is_commit_hash; then
  clone_specific_commit
else
  if ! ref_exists_remotely; then
    echo "❌ Ref '$REPO_REF' not found in $REPO_URL"
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
echo "📦 Checked out commit $HASH from $REPO_URL"

# --- Step 3: Move or Copy files exactly-deterministically ---
# if [[ -n "$SRC_SUBDIR" && ! -d "$TMP/$SRC_SUBDIR" ]]; then
# if [[ ! -f "$TMP/LICENSE" && ! -f "$TMP/LICENSE.txt" ]]; then
# SRC_PATH="${SRC_SUBDIR:+$TMP_DIR/$SRC_SUBDIR}"
# # Verify required files exist
# [[ -d "${SRC_PATH:-}" ]] || { echo "❌ Subdir '$SRC_SUBDIR' not found."; exit 1; }
# cp -a "$SRC_PATH"/. "$TARGET_DIR/"
if [[ ${#SRC_SUBDIRS[@]} -eq 0 ]]; then
    echo "📦 Copying entire repository..."
    cp -a --preserve=timestamps,mode "$TMP_DIR"/. "$TARGET_DIR/"
else
    # Verify all requested paths exist before copying
    for sub in "${SRC_SUBDIRS[@]}"; do
        local_path="$TMP_DIR/$sub"
        [[ -e "$local_path" ]] || { echo "❌ Path '$sub' not found in repo."; exit 1; }
    done
    # Copy all requested paths at once
    # cp -a "$local_path" "$TARGET_DIR/$sub"
    copy_src_paths "$TMP_DIR" "$TARGET_DIR"
fi

sync_source_tree() {
    local tmp_dir="$1"
    local target_dir="$2"
    if [[ ${#SRC_SUBDIRS[@]} -eq 0 ]]; then
        log INFO "Copying entire repository..."
        rsync -a --no-perms --no-owner --no-group "$tmp_dir"/. "$target_dir"
    else
        log INFO "Copying selective paths: ${SRC_SUBDIRS[*]}"
        copy_src_paths "$tmp_dir" "$target_dir"
    fi
}


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
    echo "📦 Moving vendored content from $TARGET_DIR to $MOVE_TO ..."

    rm -rf "$MOVE_TO"
    mkdir -p "$(dirname "$MOVE_TO")"

    if [[ -n "$NESTED_FOLDER" ]]; then
        NESTED_PATH="$TARGET_DIR/$NESTED_FOLDER"
        # Safety check: ensure nested folder is within target
        if [[ "$NESTED_PATH" != "$TARGET_DIR"* ]]; then
            echo "❌ Error: --nested-folder '$NESTED_FOLDER' points outside target!"
            exit 1
        fi
        if [[ -d "$NESTED_PATH" ]]; then
            # Move only the nested folder
            mv "$NESTED_PATH" "$MOVE_TO"
            # Remove empty parent directories if needed
            rmdir --ignore-fail-on-non-empty "$TARGET_DIR" 2>/dev/null || true
        else
            echo "❌ Nested folder '$NESTED_FOLDER' not found in target."
            exit 1
        fi
    else
        # Move entire target folder
        mv "$TARGET_DIR" "$MOVE_TO"
    fi

    # Update TARGET_DIR path for following steps (README, tree hash)
    LOCK_FILE="$MOVE_TO/vendor.lock.json"
    README_FILE="$MOVE_TO/README.md"
    FINAL_TARGET="$MOVE_TO"  # after possible move
fi

# --- # Step 4: Compute SHA256 fingerprint-hash of the vendored tree ---
# read -r TREE_MODE TREE_HASH < <(compute_tree_hash "$TARGET_DIR")
read -r TREE_MODE TREE_HASH <<<"$(compute_tree_hash "$FINAL_TARGET")"

# --- Step 5: Save-Write metadata lockfile ---
# cat >"$LOCK_FILE" <<EOF
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

# --- Step 5: Record provenance exactly README.md ---
# {echo "Vendored repository information"} > "$TARGET_DIR/README.md"
# cat >"$TARGET_DIR/README.md" <<EOF
cat <<EOF > "$README_FILE"
Vendored repository information
===============================

- Repository: $REPO_URL  # Remote Git repo URL
- Version:    $REPO_REF  # Ref Branch, Tag, or Commit SHA
- Commit:     $HASH
- Tree Mode:  $TREE_MODE
- Tree Hash:  $TREE_HASH
- Retrieved:  $(date -u +'%Y-%m-%dT%H:%M:%SZ')

To update (git clone), run:
  bash ./tools/maint_tools/$README_NAME \\
    --repo-url "$REPO_URL" \\
    --repo-ref "$REPO_REF" \\
    --target-dir "$TARGET_DIR" \\
    --move-to "$MOVE_TO" \\
    --nested-folder "$NESTED_FOLDER" \\
    --src-subdirs "${SRC_SUBDIRS[*]}" \\
    --readme-name "$README_NAME"

To update only the tree hash (no git clone):
  bash ./tools/maint_tools/$README_NAME \\
    --target-dir "$FINAL_TARGET" \\
    --update-hash

To verify in CI:
  bash ./tools/maint_tools/$README_NAME --target-dir "$FINAL_TARGET" --check
  python ./tools/maint_tools/verify_vendor.py "$FINAL_TARGET"  # --json --pretty
EOF

echo "✅ Vendoring complete (commit: $HASH)"
echo "🔐 Integrity fingerprint: $TREE_HASH"
