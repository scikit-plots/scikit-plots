
python - <<'PY'
from pathlib import Path

path = Path("scikitplot/__init__.py")
data = path.read_bytes()

# Canonical source encoding.
data.decode("utf-8")
print("PASS: UTF-8 decode")

# Reproduce legacy Windows locale behavior.
text = data.decode("cp1252")
print("PASS: cp1252 decode")

# Ensure cp1252 interpretation is byte-preserving.
assert text.encode("cp1252") == data
print("PASS: cp1252 round trip")
PY

python - <<'PY'
import locale
import sys
from pathlib import Path

path = Path("scikitplot/__init__.py")
data = path.read_bytes()

print("Python:", sys.version)
print("Native encoding:", locale.getpreferredencoding(False))
print("UTF-8 mode:", sys.flags.utf8_mode)

# Source must always be valid UTF-8.
data.decode("utf-8")
print("PASS: UTF-8")

# Compatibility check for legacy Python 3.8 / delvewheel / Windows.
cp1252_text = data.decode("cp1252")
assert cp1252_text.encode("cp1252") == data
print("PASS: Windows cp1252 compatibility")
PY

python - <<'PY'
from pathlib import Path

root = Path("scikitplot")

failed = []

for path in sorted(root.rglob("*.py")):
    data = path.read_bytes()

    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        failed.append((path, "utf-8", exc))
        continue

    try:
        text = data.decode("cp1252")
        assert text.encode("cp1252") == data
    except (UnicodeDecodeError, UnicodeEncodeError, AssertionError) as exc:
        failed.append((path, "cp1252", exc))

if failed:
    print("FAIL:")
    for path, encoding, exc in failed:
        print(f"  {path}: {encoding}: {exc}")
    raise SystemExit(1)

print("PASS: all Python files are valid UTF-8")
print("PASS: all Python files are Windows cp1252-decodable")
PY

# All Python files:
#     valid UTF-8
# Top-level scikitplot/__init__.py:
#     valid UTF-8
#     cp1252-safe for legacy delvewheel
# Windows repair:
#     python -X utf8 -m delvewheel repair

# Your whole-tree test should instead test only UTF-8
python - <<'PY'
from pathlib import Path

root = Path("scikitplot")
failed = []

for path in sorted(root.rglob("*.py")):
    try:
        path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        failed.append((path, exc))

if failed:
    print("FAIL: invalid UTF-8 Python files:")
    for path, exc in failed:
        print(f"  {path}: {exc}")
    raise SystemExit(1)

print("PASS: all Python files are valid UTF-8")
PY


# Then separately test the legacy delvewheel patch target:
python - <<'PY'
from pathlib import Path

path = Path("scikitplot/__init__.py")
data = path.read_bytes()

# Python source invariant.
data.decode("utf-8")
print("PASS: top-level __init__.py is valid UTF-8")

# Legacy delvewheel/Python 3.8 Windows regression check.
text = data.decode("cp1252")
assert text.encode("cp1252") == data
print("PASS: top-level __init__.py is cp1252-safe")
PY
