"""Tests for echotype_lab.__init__ and package-level imports."""

from __future__ import annotations


def test_package_version():
    import echotype_lab

    assert echotype_lab.__version__ == "0.1.0"


def test_submodule_imports():
    """Verify all sub-packages are importable."""
    import echotype_lab.audio
    import echotype_lab.features
    import echotype_lab.models
    import echotype_lab.evaluation
    import echotype_lab.visualization
    import echotype_lab.consent
