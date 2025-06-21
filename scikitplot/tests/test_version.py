import re

import scikitplot
import scikitplot.version


def test_valid_scikitplot_version():
    """
    Ensure `scikitplot.__version__` follows a valid PEP 440 version string format:
    https://peps.python.org/pep-0440/

    Supports:
    - X.Y[.Z]
    - aN / bN / rcN
    - .postN
    - .devN
    - +gitYYYYMMDD.hash
    """
    version = scikitplot.__version__

    # PEP 440 version components
    base = r"\d+\.\d+(?:\.\d+)?"              # major.minor(.patch)
    pre = r"(a|b|rc)\d+"                      # pre-release
    post = r"post\d+"                         # post-release
    dev = r"dev\d+"                           # dev-release
    local = r"git\d{8}\.[0-9a-f]{7}"          # local version (git)

    # Combine full PEP 440 regex
    pep440_regex = (
        rf"^{base}"
        rf"(?:(?:{pre})|(?:\.{post})|(?:\.{dev}))*"   # optional pre/post/dev
        rf"(?:\+{local})?$"                          # optional +local
    )

    match = re.match(pep440_regex, version)
    assert match is not None, (
        f"❌ Version '{version}' does not match expected PEP 440 format.\n"
        f"Expected pattern: {pep440_regex}"
    )

    assert version, "❌ Version string is empty or missing."
    print(f"✅ Valid version string: {version}")


def test_version_submodule_members():
    """
    Validate the expected public API of `scikitplot.version`.

    Ensures key attributes like `version`, `full_version`, etc., exist and are accessible.
    """
    expected_attrs = (
        "version",
        "full_version",
        "short_version",
        "git_revision",
        "release",
    )

    for attr in expected_attrs:
        assert hasattr(scikitplot.version, attr), (
            f"❌ Missing attribute '{attr}' in `scikitplot.version`."
        )
        value = getattr(scikitplot.version, attr)
        assert value is not None, (
            f"❌ Attribute '{attr}' in `scikitplot.version` is None."
        )

    # expected_types = {
    #     "version": str,
    #     "full_version": str,
    #     "short_version": str,
    #     "git_revision": str,
    #     "release": bool,
    # }

    # for attr, expected_type in expected_types.items():
    #     value = getattr(scikitplot.version, attr, None)
    #     assert isinstance(value, expected_type), (
    #         f"❌ `{attr}` should be of type {expected_type.__name__}, got {type(value).__name__}"
    #     )
