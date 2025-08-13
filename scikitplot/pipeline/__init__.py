# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the feature_engine project.
# https://github.com/feature-engine/feature_engine/blob/main/feature_engine/pipeline/pipeline.py

"""pipeline."""

from .pipeline import Pipeline, make_pipeline

__all__ = ["Pipeline", "make_pipeline"]
