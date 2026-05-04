"""Smoke tests for ``pxr_reduce`` import wiring."""

import pxr_reduce


def test_package_wires_prsoxr_loader() -> None:
    """Assert the package imports and re-exports ``PrsoxrLoader``.

    Verifies ``src`` layout resolution under pytest, stable ``__all__``, and
    that the primary public symbol remains importable without beamline data.
    """
    assert "PrsoxrLoader" in pxr_reduce.__all__
    assert pxr_reduce.PrsoxrLoader.__name__ == "PrsoxrLoader"
