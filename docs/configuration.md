# Configuration reference

There are two configuration surfaces:

- **`ReductionConfig`** ([`pxr_reduce.config`](api-reference.md#pxr_reduceconfig)) —
  the reduction *parameters* (detector, ROI, tracking, stitching). Construct one
  in Python and pass it to `PXRLoader`; every field is serialized into the `.dat`
  export header so each result records exactly how it was produced.
- **`RunConfig`** ([`pxr_reduce.run_config`](api-reference.md#pxr_reducerun_config)) —
  a single editable **TOML** file describing a whole *batch*: where the data lives,
  which tracker/export options to use, the `ReductionConfig` values, and a map of
  sample names to scan IDs. This is what `pxr-reduce batch` reads. See
  [Batch runs (TOML)](#batch-runs-toml) below.

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

> **Beam tracking.** `PXRLoader.process()` uses the **simple** tracker
> (median-filter + local-argmax, cropped to the search region): direct-beam and
> segment-start frames are located by the global peak of the full frame, and every
> other frame within `drift_distance` (or `search_radius`) of the previous
> position. The older SNR-gated tracker is available as the deprecated
> `PXRLoader.process_snr()`; the parameters it alone uses are listed under
> [Deprecated parameters](#deprecated-parameters).

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
| `filter_size` | `int` | `5` | px | Kernel size of the median filter used for beam finding and as the dezinger reference. |
| `dezinger` | `bool` | `True` | — | If True, median-filter and dezinger each frame (removes hot pixels/cosmic rays). If False, skip it for a much faster but noisier reduction. |
| `drift_distance` | `int` | `45` | px | Default beam search radius: how far the beam may move between consecutive frames. The simple tracker searches within this radius of the previous position (overridden per run by `search_radius`). |
| `dark_pix_offset` | `int` | `50` | px | Gap between the beam ROI and the dark ROI used for background subtraction. |
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
| `stitch_condition_columns` | `tuple[str, ...]` | `("hos", "exposure", "slits_vert", "slits_horz")` | — | Metadata columns whose change (beyond `stitch_condition_tol`) marks a stitch boundary, alongside a `sam_th` back-step. Missing columns are ignored. |
| `stitch_condition_tol` | `float` | `0.0` | — | A watched column must change by more than this to count as a condition change. Metadata is pre-rounded, so `0.0` means "any real change". |
| `stitch_theta_backstep` | `float` | `0.001` | deg | A `sam_th` decrease larger than this between consecutive reflectivity frames marks a stitch boundary. |
| `new_scan_marker` | `float` | `15.0` | deg | A `sam_th` jump larger than this marks the start of a new scan (used to segment a multi-scan dataset). |
| `drop_failed_stitch` | `bool` | `True` | — | If True, drop points after a stitch that could not be matched (no safe overlap angle). |

Stitch boundaries are detected between consecutive *reflectivity* frames (i0
frames are excluded, and the i0→first-measurement step is never a boundary): a
boundary is marked where `sam_th` steps back into already-measured angles **or**
any watched condition changes. Use
[`diagnose_stitches`](api-reference.md#pxr_reducereduction) /
`PXRLoader.diagnose_stitches()` to see which boundary fired, what triggered it,
how many overlap points it used, and the fitted scale.

---

## Deprecated parameters

These fields still exist on `ReductionConfig` (and feed the deprecated
`PXRLoader.process_snr()` SNR-gated tracker) but the standard `process()` does not
use them, so they are **omitted from the bundled `default_config.toml`**. Add them
back under `[reduction]` only if you run the base tracker.

| Parameter | Default | Was used for |
|---|---|---|
| `mask_threshold` | `80` | SNR tracker's static integration mask. |
| `mask_max_frames` | `200` | Frames subsampled to build that mask. |
| `beam_snr_min` | `3.0` | SNR tracker's per-frame detection gate. |
| `track_smoothing` | `True` | SNR tracker's per-scan trajectory smoothing. |
| `track_poly_order` | `3` | Polynomial order of that trajectory fit. |
| `centroid_radius` | `8` | Centroid window around the SNR-tracker peak. |
| `stitch_mark_tol` | `1e-5` | Legacy stitch marking; replaced by condition-aware detection (fully unused). |

---

## Batch runs (TOML)

`pxr-reduce batch` reads a `RunConfig` from a TOML file (resolved as
`--config` → `./reduction_config.toml` → built-in defaults). Generate a
documented starter with `pxr-reduce init-config`. Sections:

**`[paths]`**

| Key | Type | Default | Meaning |
|---|---|---|---|
| `parent_dir` | path | `"data"` | Folder searched recursively for FITS scans. |
| `results_root` | path | `"results"` | Where `<sample>.dat` + plots are written. |
| `fits_glob` | `str` | `"*.fits"` | Glob for FITS files under `parent_dir`. |
| `scan_number_width` | `int` | `5` | Digit width of the static scan-ID block in filenames (distinguishes it from the frame index). |
| `scan_number_regex` | `str \| None` | `None` | Optional regex with a `scan` group overriding the width-based rule. |

**`[tracking]`**

| Key | Type | Default | Meaning |
|---|---|---|---|
| `tracker` | `"simple" \| "base"` | `"simple"` | Beam tracker. `"base"` is the deprecated SNR-gated tracker. |
| `search_radius` | `int \| None` | `None` | Local search radius (px); `None` uses `reduction.drift_distance`. |
| `filter_size` | `int \| None` | `None` | Beam-finding median kernel; `None` uses `reduction.filter_size`. |

**`[export]`**

| Key | Type | Default | Meaning |
|---|---|---|---|
| `angle_decimals` | `int` | `4` | Decimals kept for `sam_th`; also sets the angular step propagated onto q for its significant-figure rounding. |
| `plots` | `bool` | `True` | Write I-vs-q PNGs alongside each `.dat`. |
| `apply_scale` | `bool` | `True` | Apply stitch scaling (False = quick reduction). |
| `drop_duplicates` | `bool` | `True` | Average points sharing (sam_th, energy, polarization). |

**`[reduction]`** — any `ReductionConfig` field from this page.

**`[samples]`** — `name = [scan IDs]`. A sample pools every frame from all listed
scans into one reduction, written to `results_root/<name>.dat`:

```toml
[samples]
B1A1_NEdge_XRR = [89854, 89855]
B1A1_XRR_P100 = [17344]
```

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
