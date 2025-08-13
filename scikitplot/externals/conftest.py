# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the scikit-learn project.
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/externals

# Do not collect any tests in externals. This is more robust than using
# --ignore because --ignore needs a path and it is not convenient to pass in
# the externals path (very long install-dependent path in site-packages) when
# using --pyargs
def pytest_ignore_collect(collection_path, config):
    return True
