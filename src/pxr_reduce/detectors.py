"""Detector specifications and registry.

All detector-specific properties (pixel size, bit depth, gain, noise terms) and
the per-pixel noise physics live here, encapsulated in :class:`DetectorSpec`.
Reduction code should never hardcode a detector constant; instead it holds a
``DetectorSpec`` instance and asks it for values. Swapping detectors is then a
matter of selecting a different spec from the registry.

.. warning::
    The gain, read-noise, dark-current, and bias values below are PLACEHOLDERS.
    Replace them with measured values for each detector before trusting absolute
    uncertainties. Each placeholder field is flagged in its definition.

To add a new detector, build a :class:`DetectorSpec` and call
:func:`register_detector`, or add it to :data:`_BUILTIN_DETECTORS`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorSpec:
    """Immutable description of an area detector and its noise model.

    All noise terms are expressed in detector counts (ADU) so that reduction can
    operate directly on raw images. ``gain`` bridges to electrons for shot-noise
    calculations. Values marked PLACEHOLDER must be measured per detector.

    Args:
        name: Short unique identifier used as the registry key.
        description: Human-readable description of the detector/hardware.
        pixel_size_mm: Physical size of one square pixel in millimetres.
        bit_depth: ADC bit depth; full-scale saturation is ``2**bit_depth - 1``.
        gain_e_per_adu: Detector gain in electrons per ADU (PLACEHOLDER).
        read_noise_adu: RMS read noise per pixel in ADU (PLACEHOLDER).
        dark_current_adu_per_s: Mean dark current per pixel in ADU/s (PLACEHOLDER).
        bias_adu: Electronic bias/offset level per pixel in ADU (PLACEHOLDER).
        full_well_adu: Optional full-well capacity in ADU; falls back to the ADC
            saturation value when omitted.
        extras: Free-form dictionary for detector-specific metadata that does not
            warrant a dedicated field (e.g. serial number, firmware).
    """

    name: str
    description: str
    pixel_size_mm: float
    bit_depth: int = 16
    # --- PLACEHOLDER noise parameters: replace with measured values -----------
    gain_e_per_adu: float = 1.0  # PLACEHOLDER
    read_noise_adu: float = 0.0  # PLACEHOLDER
    dark_current_adu_per_s: float = 0.0  # PLACEHOLDER
    bias_adu: float = 0.0  # PLACEHOLDER
    # -------------------------------------------------------------------------
    full_well_adu: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def saturation_adu(self) -> int:
        """Full-scale ADC saturation value in ADU."""
        return 2**self.bit_depth - 1

    @property
    def has_measured_noise(self) -> bool:
        """True if the noise model has been given non-placeholder values.

        Used to warn callers when absolute uncertainties are being computed from
        unmeasured placeholder parameters.
        """
        return (
            self.gain_e_per_adu != 1.0
            or self.read_noise_adu != 0.0
            or self.dark_current_adu_per_s != 0.0
            or self.bias_adu != 0.0
        )

    def pixel_variance_adu(
        self, signal_adu: NDArray[np.floating], exposure_s: float
    ) -> NDArray[np.floating]:
        """Per-pixel variance of a raw image in ADU^2.

        Combines photon shot noise, read noise, and dark-current shot noise:

            var = (signal - bias) / gain    # Poisson shot noise in ADU^2
                + read_noise^2               # read noise
                + dark_current * exposure    # dark shot noise

        The shot-noise term divides by gain because Poisson variance is defined
        in electrons (var_e = N_e = signal_adu * gain), which converts back to
        ADU^2 as ``signal_adu / gain``. Negative signal (after bias removal) is
        clipped to zero before taking the Poisson term.

        Args:
            signal_adu: Raw pixel values in ADU.
            exposure_s: Exposure time in seconds for the dark-current term.

        Returns:
            Array of per-pixel variances in ADU^2, same shape as ``signal_adu``.
        """
        signal = np.asarray(signal_adu, dtype=float)
        shot_e = np.clip((signal - self.bias_adu) * self.gain_e_per_adu, 0.0, None)
        shot_var_adu2 = shot_e / (self.gain_e_per_adu**2)
        dark_var_adu2 = self.dark_current_adu_per_s * exposure_s
        return shot_var_adu2 + self.read_noise_adu**2 + dark_var_adu2

    def is_saturated(self, image: NDArray[np.floating], threshold: float = 1.0) -> bool:
        """Return True if the image approaches the detector saturation level.

        Args:
            image: Raw image in ADU.
            threshold: How close (in ADU) the peak pixel must be to
                :attr:`saturation_adu` to count as saturated.

        Returns:
            True if the brightest pixel is within ``threshold`` ADU of saturation.
        """
        return bool((self.saturation_adu - np.asarray(image).max()) < threshold)

    def to_header_dict(self) -> dict[str, Any]:
        """Return a flat, serializable mapping for export headers."""
        return {
            "detector_name": self.name,
            "detector_description": self.description,
            "detector_pixel_size_mm": self.pixel_size_mm,
            "detector_bit_depth": self.bit_depth,
            "detector_saturation_adu": self.saturation_adu,
            "detector_gain_e_per_adu": self.gain_e_per_adu,
            "detector_read_noise_adu": self.read_noise_adu,
            "detector_dark_current_adu_per_s": self.dark_current_adu_per_s,
            "detector_bias_adu": self.bias_adu,
            "detector_full_well_adu": self.full_well_adu,
            "detector_noise_measured": self.has_measured_noise,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Built-in detectors. The "default" entry reproduces the constants previously
# hardcoded in the loader (0.027 mm pixels, 16-bit) with PLACEHOLDER noise.
_BUILTIN_DETECTORS: dict[str, DetectorSpec] = {
    "default": DetectorSpec(
        name="default",
        description="Generic 16-bit CCD/CMOS placeholder (BL11.0.1.2 legacy defaults)",
        pixel_size_mm=0.027,
        bit_depth=16,
    ),
    "cmos_11012": DetectorSpec(
        name="cmos_11012",
        description="RSoXS CMOS area detector at BL11.0.1.2 (PLACEHOLDER noise specs)",
        pixel_size_mm=0.027,
        bit_depth=16,
        # PLACEHOLDER values below — replace with measured detector specifications.
        gain_e_per_adu=1.0,
        read_noise_adu=0.0,
        dark_current_adu_per_s=0.0,
        bias_adu=0.0,
    ),
}

_registry: dict[str, DetectorSpec] = dict(_BUILTIN_DETECTORS)


def register_detector(spec: DetectorSpec, *, overwrite: bool = False) -> None:
    """Register a detector spec in the global registry.

    Args:
        spec: The detector specification to register.
        overwrite: If False, refuse to replace an existing name.

    Raises:
        ValueError: If ``spec.name`` is already registered and ``overwrite`` is
            False.
    """
    if spec.name in _registry and not overwrite:
        raise ValueError(
            f"Detector {spec.name!r} is already registered; pass overwrite=True to replace it."
        )
    logger.info("Registering detector %r", spec.name)
    _registry[spec.name] = spec


def get_detector(detector: str | DetectorSpec) -> DetectorSpec:
    """Resolve a detector name or spec to a :class:`DetectorSpec`.

    Args:
        detector: Either a registered detector name or a ``DetectorSpec`` (which
            is returned unchanged).

    Returns:
        The resolved detector specification.

    Raises:
        KeyError: If a name is given that is not in the registry.
    """
    if isinstance(detector, DetectorSpec):
        return detector
    try:
        return _registry[detector]
    except KeyError:
        raise KeyError(
            f"Unknown detector {detector!r}. Available: {sorted(_registry)}"
        ) from None


def available_detectors() -> list[str]:
    """Return the sorted names of all registered detectors."""
    return sorted(_registry)


def with_noise(
    detector: str | DetectorSpec,
    *,
    name: str | None = None,
    gain_e_per_adu: float | None = None,
    read_noise_adu: float | None = None,
    dark_current_adu_per_s: float | None = None,
    bias_adu: float | None = None,
) -> DetectorSpec:
    """Return a copy of a detector with updated noise parameters.

    Convenience for creating a measured variant of a built-in placeholder
    detector without mutating the original (specs are immutable).

    Args:
        detector: Base detector name or spec to copy from.
        name: Optional new name for the derived spec.
        gain_e_per_adu: Overriding gain, if provided.
        read_noise_adu: Overriding read noise, if provided.
        dark_current_adu_per_s: Overriding dark current, if provided.
        bias_adu: Overriding bias level, if provided.

    Returns:
        A new :class:`DetectorSpec` with the requested overrides applied.
    """
    base = get_detector(detector)
    updates: dict[str, Any] = {}
    if name is not None:
        updates["name"] = name
    if gain_e_per_adu is not None:
        updates["gain_e_per_adu"] = gain_e_per_adu
    if read_noise_adu is not None:
        updates["read_noise_adu"] = read_noise_adu
    if dark_current_adu_per_s is not None:
        updates["dark_current_adu_per_s"] = dark_current_adu_per_s
    if bias_adu is not None:
        updates["bias_adu"] = bias_adu
    return replace(base, **updates)
