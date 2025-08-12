# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

from collections.abc import Callable

def _preprocess_data(
    func: Callable | None = ...,
    *,
    replace_names: list[str] | None = ...,
    label_namer: str | None = ...,
) -> Callable: ...
