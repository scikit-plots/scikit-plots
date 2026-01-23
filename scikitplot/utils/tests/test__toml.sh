python - <<'PY'
from scikitplot.utils import _toml

print("READ:", _toml.TOML_READ_SUPPORT, _toml.TOML_READ_SOURCE)
print("WRITE:", _toml.TOML_WRITE_SUPPORT, _toml.TOML_WRITE_SOURCE)

# Import should not emit logs or raise even if toml backends are missing.
# Only calling read_toml/write_toml should raise when unsupported.
try:
    _toml.read_toml("does_not_exist.toml")
except Exception as e:
    print("read_toml error (expected for missing file):", type(e).__name__)
PY
