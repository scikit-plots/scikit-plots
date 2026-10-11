from __future__ import annotations

import subprocess
import sys
import textwrap


def test_import_does_not_load_optional_accelerators() -> None:
    source = textwrap.dedent(
        """
        import sys
        before = set(sys.modules)
        import scikitplot.levenshtein
        loaded = set(sys.modules) - before
        assert not any(m == "rapidfuzz" or m.startswith("rapidfuzz.") for m in loaded)
        assert not any(m == "Levenshtein" or m.startswith("Levenshtein.") for m in loaded)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
