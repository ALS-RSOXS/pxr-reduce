# Configuration reference

All reduction parameters live in the `ReductionConfig` dataclass
([`pxr_reduce.config`](api-reference.md#pxr_reduceconfig)). Construct one and pass
it to `PXRLoader`; every field is also serialized into the `.dat` export header so
each result records exactly how it was produced.

```python
from pxr_reduce import ReductionConfig

config = ReductionConfig(
    detector="cmos_11012",
    roi_height=40,
    roi_width=40,
    energy_offset=0.2,
)
```

Validation happens on construction: `darkside` must be `"LHS"`/`"RHS"`,
`energy_resolution` must be positive, and `roi_height`/`roi_width` must be
positive.

> **Note on ROI defaults.** From the CLI, `--roi-height`/`--roi-width` default to
> the values below (they only override when you pass them). When
> `roi_from_beam_fit=True`, the ROI is computed from the direct-beam fit and these
> values are replaced.

---

## Detector

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `detector` | `str \| DetectorSpec` | `"default"` | Which detector to use. A registered name (see `pxr-reduce list-detectors`) or a `DetectorSpec` instance. Detector-specific values (pixel size, bit depth, gain, read noise, dark current, bias) live on the spec, **not** here. See [Detectors](#detector-specifications). |

## Metadata & geometry

| Parameter | Type | Default | Units | Meaning |
|---|---|---|---|---|
| `exposure_offset` | `float` | `0.00389278` | s | Added to each frame's exposure time to account for the physical shutter open/close time. Measured once; rarely changed. |
| `energy_resolution` | `float` | `20.0` | 1/eV | Photon energy is rounded to the nearest `1/energy_resolution` eV (default: nearest 0.05 eV). Groups near-identical energies together. Must be > 0. |
| `energy_offset` | `float` | `0.0` | eV | Additive correction applied to the recorded photon energy. |
| `sam_th_offset` | `float \| None` | `None` | deg | Fixed offset added to sample theta. If `None` and `sam_th_correction` is True, it is determined automatically from the measurement geometry. |
| `sam_th_correction` | `bool` | `True` | — | When `sam_th_offset` is `None`, auto-determine the offset assuming a theta–2theta geometry (from the first data-collection angle after `sam_z` moves into the beam). |

## Image processing

| Parameter | Type | Default | Units | Meaning |
|---|---|---|---|---|
| `roi_height` | `int` | `40` | px | Height of the beam ROI integrated on each frame. Must be > 0. Replaced when `roi_from_beam_fit=True`. |
| `roi_width` | `int` | `40` | px | Width of the beam ROI. Must be > 0. Replaced when `roi_from_beam_fit=True`. |
| `trim_x` | `int` | `20` | px | Pixels removed from each **vertical** edge (axis 0) before processing, to drop detector border artifacts. |
| `trim_y` | `int` | `20` | px | Pixels removed from each **horizontal** edge (axis 1) before processing. |
| `filter_size` | `int` | `3` | px | Kernel size of the median filter used as the dezinger reference. |
| `dezinger` | `bool` | `True` | — | If True, median-filter and dezinger each frame (removes hot pixels/cosmic rays). If False, skip it for a much faster but noisier reduction. |
| `mask_threshold` | `int` | `90` | counts | Mean-image counts above which a pixel is treated as a persistent beam location when building the integration mask. |
| `mask_max_frames` | `int` | `200` | frames | Maximum number of frames read to build the mask; frames are evenly subsampled above this count. `0` uses all frames. |
| `drift_distance` | `int` | `25` | px | Radius by which mask seed regions are expanded, allowing the beam to drift between frames without leaving the mask. |
| `dark_pix_offset` | `int` | `20` | px | Gap between the beam ROI and the dark ROI used for background subtraction. |
| `darkside` | `"LHS" \| "RHS"` | `"LHS"` | — | Preferred side of the beam to place the dark ROI. Falls back to the opposite side when there is not enough room. |
| `saturate_threshold` | `float` | `2.0` | counts | A frame is flagged saturated if its peak is within this many counts of the detector's full-scale value. |

## ROI from the direct beam

| Parameter | Type | Default | Units | Meaning |
|---|---|---|---|---|
| `roi_from_beam_fit` | `bool` | `False` | — | If True, size the ROI from a 2D-moments fit of the direct-beam (i0) frames instead of using `roi_height`/`roi_width`. The fitted ROI replaces those values during `process()`. |
| `roi_n_sigma` | `float` | `3.0` | σ | ROI half-extent in beam sigmas (ROI dimension ≈ `2 · roi_n_sigma · sigma`). |
| `roi_fit_window` | `int` | `50` | px | Size of the window cropped around the beam peak for the moments fit. Should comfortably contain the beam. |

## Stitching & scaling

| Parameter | Type | Default | Units | Meaning |
|---|---|---|---|---|
| `stitch_cutoff` | `float` | `1.003` | ratio | Minimum spot/dark counts ratio for a point to be eligible as a stitch-overlap point (rejects near-background points). |
| `stitch_mark_tol` | `float` | `1e-5` | — | Minimum tracked-motor motion that marks a stitch boundary between segments. |
| `new_scan_marker` | `float` | `15.0` | deg | A `sam_th` jump larger than this marks the start of a new scan (used to segment a multi-scan dataset). |
| `drop_failed_stitch` | `bool` | `True` | — | If True, drop points after a stitch that could not be matched (no safe overlap angle). |

---

## Detector specifications

Detector-specific constants live in a `DetectorSpec`
([`pxr_reduce.detectors`](api-reference.md#pxr_reducedetectors)), selected via the
`detector` config field. This keeps the reduction independent of any one detector.

| Field | Type | Meaning |
|---|---|---|
| `name` | `str` | Registry key / identifier. |
| `description` | `str` | Human-readable description. |
| `pixel_size_mm` | `float` | Physical size of one square pixel (mm). |
| `bit_depth` | `int` | ADC bit depth; saturation = `2**bit_depth - 1`. |
| `gain_e_per_adu` | `float` | Detector gain, electrons per ADU. **Placeholder** until measured. |
| `read_noise_adu` | `float` | RMS read noise per pixel (ADU). **Placeholder** until measured. |
| `dark_current_adu_per_s` | `float` | Mean dark current per pixel (ADU/s). **Placeholder** until measured. |
| `bias_adu` | `float` | Electronic bias/offset per pixel (ADU). **Placeholder** until measured. |
| `full_well_adu` | `float \| None` | Optional full-well capacity; defaults to the ADC saturation value. |
| `extras` | `dict` | Free-form detector metadata (serial number, firmware, ...). |

Built-in detectors:

- **`default`** — generic 16-bit, 0.027 mm pixels, placeholder noise. Reproduces
  the constants used by the legacy loader.
- **`cmos_11012`** — the RSoXS CMOS detector, placeholder noise specs.

> **Placeholder noise.** Until real gain / read noise / dark current / bias are
> supplied, `has_measured_noise` is `False`, absolute `R_err` values are not
> physical, and the export header is tagged `[PLACEHOLDER noise specs]`.
> Reflectivity `R` and `q` are unaffected.

### Registering a measured detector

```python
from pxr_reduce.detectors import with_noise, register_detector

measured = with_noise(
    "cmos_11012",
    name="cmos_11012_measured",
    gain_e_per_adu=2.1,
    read_noise_adu=1.8,
    dark_current_adu_per_s=0.05,
    bias_adu=100.0,
)
register_detector(measured)

config = ReductionConfig(detector="cmos_11012_measured")
```

See the [uncertainty model](api-reference.md#pxr_reduceuncertainty) for how these
values propagate into `R_err`.
