import re
from unittest import mock

import pytest

import scikitplot
import scikitplot.version


def is_valid_version(version: str) -> bool:
    """
    Ensure `scikitplot.__version__` follows a valid PEP 440 version string format:
    https://peps.python.org/pep-0440/

    Supports:
    - X.Y[.Z]
    - aN / bN / rcN
    - .postN
    - .devN
    - +gitYYYYMMDD.hash


    PEP 440’s canonical form does not technically allow two-part versions
    like 0.5.dev0, but many projects use them.

    - ✅ 0.5.0.dev0+git.20250621.3d9d503 (fully PEP 440 compliant)
    - 🚫 0.5.dev0+git.20250621.3d9d503 (not strictly compliant but allow it)

    Supports both:
    - 'X.Y.Z[.devN|.postN|aN|bN|rcN][+gitYYYYMMDD.commit]'
    - 'X.Y[.devN][+gitYYYYMMDD.commit]' (loosened to allow no patch version)

    Returns
    -------
    True if the version string matches the PEP 440-compatible regex.
    """

    # PEP 440 version components
    # Base version: major.minor(.patch)
    base = r"\d+\.\d+(?:\.\d+)?"              # major.minor(.patch) X.Y or X.Y.Z e.g., 0.5 or 0.5.0
    # Pre-release, post-release, dev-release
    pre = r"(?:a|b|rc)\d+"                    # pre-release
    post = r"post\d+"                         # post-release
    dev = r"dev\d+"                           # dev-release
    # Optional local version identifier: allow optional dot after 'git'
    # e.g. +git20250621.abcdef1 or +git.20250621.abcdef1
    local = r"git\.?\d{8}\.[0-9a-f]{7}"       # local version (git) e.g., +git20250621.abcdef1

    # Combine full PEP 440 regex pattern (match start to end)
    pattern = (
        # base version
        rf"^{base}"
        # optional pre/post/dev with dots where required
        rf"(?:(?:{pre})|(?:\.{post})|(?:\.{dev}))*"
        # optional local version +git...
        rf"(?:\+{local})?$"
    )
    return re.match(pattern, version) is not None


valid_versions = [
    ("0.5.dev0+git20250621.abcdef1", "dev release, no dot in local part"),
    ("0.5.dev0+git.20250621.abcdef1", "dev release, dot in local part"),
    ("0.5.0.dev0+git.20250621.abcdef1", "patch + dev + local"),
    ("0.5.0rc1", "release candidate"),
    ("0.5.0a1", "alpha release"),
    ("0.5.0b2", "beta release"),
    ("0.5.0.post1", "post release"),
    ("0.5.0", "final release"),
    ("1.2", "two-part release"),
    ("1.2.3", "three-part release"),
    # ("1!2.3.4", "epoch notation - optional"),
]
invalid_versions = [
    ("0.5.dev+git.20250621.abcdef1", "missing dev digit"),
    ("0.5.0rc", "missing digit after rc"),
    ("version0.5.0", "invalid prefix"),
    ("0.5.dev0+git.20250621.abcd", "short hash in local part"),
    ("0.5.dev0+git20250621abcdefg", "missing dot in local version"),
    ("0.5.0..dev0", "double dots"),
    ("0.5.0 ", "trailing space"),
    (" 0.5.0", "leading space"),
    ("0.5.0+local!", "invalid local char"),
]


def patch_version_and_test(version: str):
    with mock.patch("scikitplot.__version__", version):
        assert is_valid_version(scikitplot.__version__), f"Mocked version '{version}' is invalid"


@pytest.mark.parametrize("version,desc", valid_versions, ids=[v[1] for v in valid_versions])
def test_mock_versions(version, desc) -> None:
    patch_version_and_test(version)


@pytest.mark.parametrize("version,desc", invalid_versions, ids=[v[1] for v in invalid_versions])
def test_invalid_versions(version, desc):
    assert not is_valid_version(version), f"Bad version '{version}' passed validation ({desc})"


def test_valid_scikitplot_version():
    """
    Ensure `scikitplot.__version__` follows a valid PEP 440 version format,
    with relaxed support for two-part versions like '0.5.dev0'
    and allowing optional dot after 'git'.
    """
    version = scikitplot.__version__
    assert version is not None, f"Attribute '{version}' is None"
    assert is_valid_version(version), f"Real version '{version}' is invalid"


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


# def test_version_submodule_members() -> None:
#     expected_attrs = {
#         "version": str,
#         "full_version": str,
#         "short_version": str,
#         "git_revision": str,
#         "release": bool,
#     }

#     for attr, expected_type in expected_attrs.items():
#         assert hasattr(scikitplot.version, attr), f"Missing attribute '{attr}'"
#         value = getattr(scikitplot.version, attr)
#         assert value is not None, f"Attribute '{attr}' is None"
#         assert isinstance(value, expected_type), (
#             f"Attribute '{attr}' expected type {expected_type.__name__} but got {type(value).__name__}"
#         )
