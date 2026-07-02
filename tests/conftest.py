"""Shared pytest fixtures, including synthetic FITS generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


def write_synthetic_fits(
    path: Path,
    image: np.ndarray,
    header: dict | None = None,
) -> Path:
    """Write a FITS file matching the BL11.0.1.2 layout (metadata HDU 0, image HDU 2).

    Args:
        path: Destination path for the FITS file.
        image: 2D pixel array to store in HDU 2.
        header: Optional header keyword/value pairs for HDU 0.

    Returns:
        The path written.
    """
    primary = fits.PrimaryHDU()
    if header:
        for key, value in header.items():
            primary.header[key] = value
    hdul = fits.HDUList(
        [
            primary,
            fits.ImageHDU(data=np.zeros((2, 2))),  # HDU 1 (unused by loader)
            fits.ImageHDU(data=image.astype(np.int32)),  # HDU 2 (pixel data)
        ]
    )
    hdul.writeto(path, overwrite=True)
    return path


@pytest.fixture
def fits_writer():
    """Return the :func:`write_synthetic_fits` helper for direct use in tests."""
    return write_synthetic_fits


def _beam_image(peak, size=61, center=(30, 30)):
    """A gaussian-ish beam blob on a low background."""
    img = np.full((size, size), 5.0)
    yy, xx = np.mgrid[0:size, 0:size]
    r2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
    img += peak * np.exp(-r2 / (2 * 3.0**2))
    return img


def _frame_header(sam_th, sam_z, energy=250.0, polarization=100.0):
    """Full raw header for a synthetic frame."""
    return {
        "Beamline Energy": energy,
        "EPU Polarization": polarization,
        "Sample Theta": sam_th,
        "CCD Theta": 2 * sam_th,
        "Sample X": 0.0,
        "Sample Y": 0.0,
        "Sample Z": sam_z,
        "EXPOSURE": 1.0,
        "Higher Order Suppressor": 5.0,
        "Upstream JJ Vert Aperture": 0.1,
        "Upstream JJ Horz Aperture": 0.1,
        "Beam Current": 500.0,
        "AI 3 Izero": 1.0,
    }


@pytest.fixture
def processed_loader_factory(tmp_path):
    """Return a factory that builds and processes a PXRLoader of synthetic data."""
    from pxr_reduce.config import ReductionConfig
    from pxr_reduce.core import PXRLoader

    def _make(sample="MF999A", energy=250.0, polarization=100.0, subdir=None):
        base = tmp_path / subdir if subdir else tmp_path
        base.mkdir(parents=True, exist_ok=True)
        peaks = [10000.0, 10000.0, 4000.0, 2000.0, 900.0, 400.0]
        sam_z = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        sam_th = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        files = []
        for i, (peak, z, th) in enumerate(zip(peaks, sam_z, sam_th)):
            path = base / f"{sample}_{i}.fits"
            write_synthetic_fits(
                path, _beam_image(peak), _frame_header(th, z, energy, polarization)
            )
            files.append(path)
        loader = PXRLoader(files, ReductionConfig(roi_height=9, roi_width=9))
        loader.process()
        return loader

    return _make


@pytest.fixture
def frame_builders():
    """Return ``(beam_image, frame_header)`` helpers for building synthetic frames."""
    return _beam_image, _frame_header


@pytest.fixture
def synthetic_scan_folder(tmp_path):
    """Write a 6-frame synthetic scan into a folder and return the folder path."""

    def _make(sample="MF999A", energy=250.0, polarization=100.0):
        peaks = [10000.0, 10000.0, 4000.0, 2000.0, 900.0, 400.0]
        sam_z = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        sam_th = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        for i, (peak, z, th) in enumerate(zip(peaks, sam_z, sam_th)):
            write_synthetic_fits(
                tmp_path / f"{sample}_{i}.fits",
                _beam_image(peak),
                _frame_header(th, z, energy, polarization),
            )
        return tmp_path

    return _make


@pytest.fixture
def synthetic_fits_factory(tmp_path):
    """Return a factory that writes synthetic FITS files into a temp dir."""

    def _make(index: int, image: np.ndarray, header: dict | None = None) -> Path:
        path = tmp_path / f"sample_{index}.fits"
        return write_synthetic_fits(path, image, header)

    return _make
