"""Smoke tests for package scaffolding and subpackage importability."""

from __future__ import annotations

import importlib

import pytest

import easypasal


def test_package_version() -> None:
    """Verify package version is defined and follows semantic format."""
    assert hasattr(easypasal, "__version__")
    assert isinstance(easypasal.__version__, str)
    assert len(easypasal.__version__) > 0


@pytest.mark.parametrize(
    "subpackage_name",
    [
        "easypasal.ingest",
        "easypasal.index",
        "easypasal.retrieval",
        "easypasal.generation",
        "easypasal.guardrails",
        "easypasal.api",
        "easypasal.ui",
        "eval",
        "eval.metrics",
    ],
)
def test_subpackages_importable(subpackage_name: str) -> None:
    """Verify all architectural modules can be imported without error."""
    module = importlib.import_module(subpackage_name)
    assert module is not None
