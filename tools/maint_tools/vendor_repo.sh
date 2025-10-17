#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# vendor_repo.sh — Deterministic vendoring with reproducibility and integrity
# -----------------------------------------------------------------------------

set -o errexit
set -o nounset
set -euo pipefail

# ---------------------------------------------------------------------
# Compute deterministic tree hash (excluding lock/readme/gitignore)
# ---------------------------------------------------------------------
# Automatically falls back to Python if `sha256sum` or `find` is missing.
# TREE_INFO=$(...) + split
# TREE_INFO=$(compute_tree_hash "$TARGET")
# TREE_MODE="${TREE_INFO%% *}"
# TREE_HASH="${TREE_INFO#* }"
# Compute actual tree hash (mode + hash)
# read -r ACTUAL_MODE ACTUAL_HASH < <(compute_tree_hash "$TARGET")
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
    # ACTUAL_HASH=$(git -C "$TARGET" rev-parse HEAD 2>/dev/null || echo "none")
    # ACTUAL_HASH=$(find "$TARGET" -type f -exec sha256sum {} \; | sort | sha256sum | awk '{print $1}')
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

# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
function usage() {
    echo "Usage: $0 --url URL --version TAG --target PATH [--src-subdir SUBDIR] [--readme-name NAME] [--check]"
}
MODE="update" # default
URL="" VERSION="" TARGET="" SRC_SUBDIR="" README_NAME="vendor_repo.sh"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --url) URL="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --src-subdir) SRC_SUBDIR="$2"; shift 2 ;;
        --readme-name) README_NAME="$2"; shift 2 ;;
        --check) MODE="check"; shift ;;
        --update-hash) MODE="update_hash"; shift ;;   # 👈 NEW
        --help|-h)
            # echo "Usage: $0 --url URL --version TAG --target PATH [--src-subdir SUBDIR] [--readme-name NAME] [--check]"
            usage;
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -z "$TARGET" ]] && { echo "❌ --target required."; exit 1; }
TARGET=$(realpath "$TARGET")
LOCK_FILE="$TARGET/vendor.lock.json"

# ---------------------------------------------------------------------
# Integrity check mode
# ---------------------------------------------------------------------
# echo "$MODE"
if [[ "$MODE" == "check" ]]; then
    echo "🔍 Running integrity check on $TARGET..."
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
        EXPECTED_HASH=$(python -c "import json,sys;print(json.load(open('$LOCK_FILE'))['tree_hash'])")
    fi

    # Compute actual tree hash (mode + hash)
    # read -r ACTUAL_MODE ACTUAL_HASH < <(compute_tree_hash "$TARGET")
    read -r ACTUAL_MODE ACTUAL_HASH <<<"$(compute_tree_hash "$TARGET")"

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

# ---------------------------------------------------------------------
# Update only the tree hash (no git clone)
# ---------------------------------------------------------------------
# bash ./tools/maint_tools/vendor_repo.sh \
#   --target "../scikitplot/cexternals/NumCpp" \
#   --update-hash
if [ "$MODE" = "update_hash" ]; then
    echo "🔁 Recomputing tree hash for $TARGET..."

    if [ ! -f "$LOCK_FILE" ]; then
        echo "❌ No vendor.lock.json found; cannot update hash."
        exit 1
    fi

    read -r NEW_MODE NEW_HASH <<<"$(compute_tree_hash "$TARGET")"
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
    if [ -f "$TARGET/README.md" ]; then
        sed -i.bak -E \
            -e "s/^(- Tree Mode:).*/\1  $NEW_MODE/" \
            -e "s/^(- Tree Hash:).*/\1  $NEW_HASH/" \
            -e "s/^(- Retrieved:).*/\1  $(date -u +'%Y-%m-%dT%H:%M:%SZ')/" \
            "$TARGET/README.md" && rm -f "$TARGET/README.md.bak"
        echo "📘 Updated README.md with new hash."
    fi

    echo "✅ Updated vendor.lock.json and README.md with recomputed hash."
    exit 0
fi

# ---------------------------------------------------------------------
# Update (vendoring) mode
# ---------------------------------------------------------------------
# [[ -z "$URL" || -z "$VERSION" ]] && { echo "❌ --url and --version required for update mode."; exit 1; }
[ -z "$URL" ] && { echo "❌ --url required."; exit 1; }
[ -z "$VERSION" ] && { echo "❌ --version required."; exit 1; }

TMP_DIR="$TARGET/.tmp"
rm -rf "$TMP_DIR" "$TARGET"
mkdir -p "$TARGET" "$TMP_DIR"

# --- Step 1: Verify that the tag/branch exists remotely ---
if ! git ls-remote --exit-code --tags "$URL" "refs/tags/$VERSION" >/dev/null 2>&1; then
    if ! git ls-remote --exit-code --heads "$URL" "$VERSION" >/dev/null 2>&1; then
        echo "❌ Version '$VERSION' not found in $URL"
        exit 1
    fi
fi

# --- Step 2: Clone and get exact commit hash ---
git clone --depth 1 --branch "$VERSION" "$URL" "$TMP_DIR"
pushd "$TMP_DIR" >/dev/null
HASH=$(git rev-parse HEAD)
popd >/dev/null
echo "📦 Checked out commit $HASH from $URL"

# --- Step 3: Move or Copy files exactly-deterministically ---
SRC_PATH="${SRC_SUBDIR:+$TMP_DIR/$SRC_SUBDIR}"
# Verify required files exist
# if [[ -n "$SRC_SUBDIR" && ! -d "$TMP/$SRC_SUBDIR" ]]; then
# if [[ ! -f "$TMP/LICENSE" && ! -f "$TMP/LICENSE.txt" ]]; then
[[ -d "${SRC_PATH:-}" ]] || { echo "❌ Subdir '$SRC_SUBDIR' not found."; exit 1; }

cp -a "$SRC_PATH"/. "$TARGET/"
cp -a "$TMP_DIR"/LICENSE* "$TARGET/" 2>/dev/null || true
rm -rf "$TMP_DIR"

# --- # Step 4: Compute SHA256 fingerprint-hash of the vendored tree ---
# read -r TREE_MODE TREE_HASH < <(compute_tree_hash "$TARGET")
read -r TREE_MODE TREE_HASH <<<"$(compute_tree_hash "$TARGET")"

# --- Step 5: Save-Write metadata lockfile ---
# cat >"$LOCK_FILE" <<EOF
cat <<EOF > "$LOCK_FILE"
{
  "repository": "$URL",
  "version": "$VERSION",
  "commit_hash": "$HASH",
  "tree_mode": "$TREE_MODE",
  "tree_hash": "$TREE_HASH",
  "generated_utc": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
}
EOF

# --- Step 5: Record provenance exactly README.md ---
# cat >"$TARGET/README.md" <<EOF
cat <<EOF > "$TARGET/README.md"
Vendored repository information
===============================

- Repository: $URL
- Version:    $VERSION
- Commit:     $HASH
- Tree Mode:  $TREE_MODE
- Tree Hash:  $TREE_HASH
- Retrieved:  $(date -u +'%Y-%m-%dT%H:%M:%SZ')

To update (git clone), run:
  bash ./tools/maint_tools/$README_NAME \\
    --url "$URL" \\
    --version "$VERSION" \\
    --target "$TARGET" \\
    --src-subdir "$SRC_SUBDIR" \\
    --readme-name "$README_NAME"

To update only the tree hash (no git clone):
  bash ./tools/maint_tools/$README_NAME \\
    --target "$TARGET" \\
    --update-hash

To verify in CI:
  bash ./tools/maint_tools/$README_NAME --target "$TARGET" --check
  python ./tools/maint_tools/verify_vendor.py "$TARGET"  # --json --pretty
EOF

echo "✅ Vendoring complete (commit: $HASH)"
echo "🔐 Integrity fingerprint: $TREE_HASH"
