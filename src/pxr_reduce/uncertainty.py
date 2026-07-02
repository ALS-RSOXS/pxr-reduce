"""Uncertainty model and error propagation for PXR reduction.

This module implements the per-point uncertainty pipeline described in the design
doc (section 7). Per-pixel variances come from the detector object
(:meth:`~pxr_reduce.detectors.DetectorSpec.pixel_variance_adu`), keeping all
detector physics encapsulated there; this module handles the propagation through
ROI summation, dark subtraction, normalization, and stitch scaling.

Uncertainties are represented as ``Value`` pairs of (value, sigma). All functions
are pure and operate on scalars/arrays so they can be unit-tested in isolation
and reused by the reduction stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pxr_reduce.detectors import DetectorSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Value:
    """A scalar measurement with a 1-sigma uncertainty.

    Args:
        value: The measured value.
        sigma: The 1-sigma (standard) uncertainty; non-negative.
    """

    value: float
    sigma: float

    @property
    def rel(self) -> float:
        """Relative uncertainty (sigma / value); 0.0 when value is 0."""
        return self.sigma / self.value if self.value != 0 else 0.0


def roi_variance(
    roi: NDArray[np.floating], detector: DetectorSpec, exposure_s: float
) -> float:
    """Total variance (ADU^2) of the summed counts in an ROI.

    Summed counts have a variance equal to the sum of per-pixel variances,
    because pixel noise is independent.

    Args:
        roi: The ROI sub-image in ADU.
        detector: Detector supplying the per-pixel noise model.
        exposure_s: Exposure time in seconds (for the dark-current term).

    Returns:
        Variance of the ROI sum in ADU^2.
    """
    return float(detector.pixel_variance_adu(roi, exposure_s).sum())


def net_counts(
    spot: NDArray[np.floating],
    dark: NDArray[np.floating],
    detector: DetectorSpec,
    exposure_s: float,
) -> Value:
    """Background-subtracted counts and their uncertainty for one frame.

    Computes ``sum(spot) - sum(dark)`` with variance
    ``var(spot) + var(dark)`` (independent regions add in quadrature).

    Args:
        spot: Beam ROI sub-image in ADU.
        dark: Dark ROI sub-image in ADU (same shape as ``spot``).
        detector: Detector supplying the per-pixel noise model.
        exposure_s: Exposure time in seconds.

    Returns:
        A :class:`Value` of net counts and its 1-sigma uncertainty in ADU.
    """
    net = float(np.asarray(spot).sum() - np.asarray(dark).sum())
    var = roi_variance(spot, detector, exposure_s) + roi_variance(
        dark, detector, exposure_s
    )
    return Value(net, float(np.sqrt(var)))


def scale(v: Value, factor: float) -> Value:
    """Multiply a value by an exact (uncertainty-free) factor.

    Used for linear normalizations such as dividing by ``exposure * beam_current``
    where the divisor is treated as exact.

    Args:
        v: The value to scale.
        factor: Exact multiplicative factor.

    Returns:
        The scaled value with proportionally scaled uncertainty.
    """
    return Value(v.value * factor, abs(factor) * v.sigma)


def ratio(numerator: Value, denominator: Value) -> Value:
    """Divide two values, propagating relative uncertainties in quadrature.

    Args:
        numerator: The numerator value/uncertainty.
        denominator: The denominator value/uncertainty.

    Returns:
        ``numerator / denominator`` with propagated uncertainty. Returns a zero
        Value when the denominator value is zero.
    """
    if denominator.value == 0:
        logger.warning("Division by zero-valued denominator; returning Value(0, 0).")
        return Value(0.0, 0.0)
    result = numerator.value / denominator.value
    rel = np.sqrt(numerator.rel**2 + denominator.rel**2)
    return Value(result, abs(result) * rel)


def product(a: Value, b: Value) -> Value:
    """Multiply two values, propagating relative uncertainties in quadrature.

    Args:
        a: First factor.
        b: Second factor.

    Returns:
        ``a * b`` with propagated uncertainty.
    """
    result = a.value * b.value
    rel = np.sqrt(a.rel**2 + b.rel**2)
    return Value(result, abs(result) * rel)


def apply_scale_factor(r: Value, scale_factor: Value) -> Value:
    """Apply a stitch scale factor to a reflectivity point.

    Reflectivity is divided by the scale factor, propagating the scale factor's
    uncertainty. This is the step omitted (commented out) in the original loader,
    which caused ``R_err`` to ignore stitch-ratio uncertainty.

    Args:
        r: The reflectivity value/uncertainty before scaling.
        scale_factor: The cumulative stitch scale factor and its uncertainty.

    Returns:
        The scaled reflectivity with fully propagated uncertainty.
    """
    return ratio(r, scale_factor)
