"""pxr-reduce: tools for reducing polarized X-ray reflectivity data.

Public API:

* :class:`~pxr_reduce.core.PXRLoader` - load and reduce a series of FITS frames.
* :class:`~pxr_reduce.config.ReductionConfig` - typed reduction parameters.
* :class:`~pxr_reduce.dataset.ReducedDataset` - export/combine reduced results.
* :mod:`pxr_reduce.detectors` - detector specifications and registry.
* :mod:`pxr_reduce.viewer` - metadata-driven image viewer.
"""

from pxr_reduce.config import ReductionConfig
from pxr_reduce.core import PXRLoader
from pxr_reduce.dataset import ReducedDataset
from pxr_reduce.detectors import (
    DetectorSpec,
    available_detectors,
    get_detector,
    register_detector,
)

__author__ = "Thomas Ferron"
__email__ = "tjferron@lbl.gov"

__all__ = [
    "PXRLoader",
    "ReductionConfig",
    "ReducedDataset",
    "DetectorSpec",
    "available_detectors",
    "get_detector",
    "register_detector",
]
