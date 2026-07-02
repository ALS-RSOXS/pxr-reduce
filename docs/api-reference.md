# API reference

Public modules, classes, and functions. Signatures use Python type hints; keyword
-only arguments follow `*`. For parameter *meanings* of the reduction settings see
the [Configuration reference](configuration.md).

## Top-level exports

```python
from pxr_reduce import (
    PXRLoader,          # load + reduce a series of frames
    ReductionConfig,    # typed reduction parameters
    ReducedDataset,     # export / combine reduced results
    DetectorSpec,       # detector specification
    get_detector, register_detector, available_detectors,
)
```

---

## `pxr_reduce.core`

### `class PXRLoader`

Orchestrates loading, processing, and reduction. Holds the scalar metadata table
and a lazy image store.

```python
PXRLoader(files: list[str | Path],
          config: ReductionConfig | None = None,
          *, auto_process: bool = False,
          cache_size: int = 64)
```

**Attributes**

| Attribute | Type | Description |
|---|---|---|
| `data` | `DataFrame` | Scalar metadata + per-frame counts (no images). |
| `config` | `ReductionConfig` | Active configuration. |
| `mask` | `ndarray[bool] \| None` | Integration mask (set by `process`). |
| `name` | `str` | Inferred sample name. |
| `path` | `Path` | Source directory. |
| `beam_shape` | `BeamShape \| None` | Fitted beam shape when `roi_from_beam_fit`. |
| `sam_th_offset_applied` | `float` | Sample-theta offset applied. |
| `data_processed` | `bool` | Whether `process()` has run. |

**Methods**

| Method | Description |
|---|---|
| `process() -> None` | Build the mask and integrate every frame. |
| `reduce(*, apply_scale=True, drop_duplicates=True) -> DataFrame` | Reduce to the 1D curve. `apply_scale=False` skips stitching (quick mode). |
| `query(**conditions) -> DataFrame` | Filter metadata rows; scalar = equality, 2-tuple = inclusive range. |
| `get_image(fits_index) -> ndarray` | Raw frame (lazy-loaded). |
| `get_clean_image(fits_index) -> ndarray` | Trimmed + dezingered frame. |
| `__call__(**kwargs)` | Alias for `reduce`. |

`reduce()` returns columns: `scan, energy, polarization, sam_th, q, R, R_err`.

---

## `pxr_reduce.config`

### `class ReductionConfig`

Typed dataclass of all reduction parameters — see the
[Configuration reference](configuration.md) for every field. Key methods:

| Method | Description |
|---|---|
| `detector_spec() -> DetectorSpec` | Resolve `detector` to a concrete spec. |
| `to_header_dict() -> dict` | Flat, serializable config + detector for headers. |

---

## `pxr_reduce.detectors`

### `class DetectorSpec` (frozen)

```python
DetectorSpec(name, description, pixel_size_mm, bit_depth=16,
             gain_e_per_adu=1.0, read_noise_adu=0.0,
             dark_current_adu_per_s=0.0, bias_adu=0.0,
             full_well_adu=None, extras={})
```

| Member | Description |
|---|---|
| `saturation_adu` (property) | `2**bit_depth - 1`. |
| `has_measured_noise` (property) | False while noise params are placeholders. |
| `pixel_variance_adu(signal_adu, exposure_s) -> ndarray` | Per-pixel variance (ADU²): shot + read + dark. |
| `is_saturated(image, threshold=1.0) -> bool` | Peak within `threshold` of saturation. |
| `to_header_dict() -> dict` | Flat mapping for export headers. |

### Functions

| Function | Description |
|---|---|
| `get_detector(detector: str \| DetectorSpec) -> DetectorSpec` | Resolve a name or pass through a spec. |
| `register_detector(spec, *, overwrite=False) -> None` | Add a spec to the registry. |
| `available_detectors() -> list[str]` | Sorted registered names. |
| `with_noise(detector, *, name=None, gain_e_per_adu=None, read_noise_adu=None, dark_current_adu_per_s=None, bias_adu=None) -> DetectorSpec` | Copy a spec with updated noise params. |

---

## `pxr_reduce.dataset`

### `class ReducedDataset`

Reduced curve + provenance, with export and combine.

```python
ReducedDataset(data: DataFrame, provenance: ReductionProvenance)
```

| Method | Description |
|---|---|
| `from_loader(loader, *, reduced=None, apply_scale=True, drop_duplicates=True, reduction_time=None)` | Build from a processed loader (classmethod). |
| `combine(other) -> ReducedDataset` | Merge with another, preserving both provenances. |
| `combine_all(datasets) -> ReducedDataset` | Merge a list (staticmethod). |
| `save(path, *, plots=True, dry_run=False) -> dict` | Write `.dat` + a `<stem>_plots/` folder. |
| `save_dat(path, *, dry_run=False) -> Path` | Write the tab-delimited `.dat` with header. |
| `save_plots(directory, *, dry_run=False) -> list[Path]` | One I-vs-q PNG per (energy, polarization). |
| `header_lines() -> list[str]` | The commented header lines. |

---

## `pxr_reduce.reduction`

Composable reduction stages. Each takes the counts table and a `ReductionConfig`.

| Function | Description |
|---|---|
| `reduce(df, config, *, apply_scale=True, drop_duplicates=True) -> DataFrame` | Full pipeline (normalize → stitch → scale → finalize). |
| `normalize_scan(df, config) -> DataFrame` | Per-scan normalization to i0. |
| `mark_stitch_points(df, config) -> DataFrame` | Mark stitch boundaries. |
| `compute_scale_factors(df, config) -> DataFrame` | Fit and accumulate stitch scale factors. |
| `apply_scaling(df, config) -> DataFrame` | Divide R by scale, propagate error. |
| `finalize(df, config, drop_duplicates=True) -> DataFrame` | Mask invalid points, select output. |
| `stitch_ratio_model(r, scale) -> ndarray` | `curve_fit` model `y = scale·r`. |

---

## `pxr_reduce.image_ops`

Array-level image processing (fed from the `ImageStore`).

| Function | Description |
|---|---|
| `trim(image, trim_x, trim_y) -> ndarray` | Remove edge pixels. |
| `dezinger(image, config) -> ndarray` | Median-filter/dezinger (or passthrough). |
| `clean_image(raw, config) -> ndarray` | Trim + dezinger a full frame. |
| `build_series_mask(images, config) -> ndarray[bool]` | Mask from the mean image + drift dilation. |
| `mask_bounding_box(mask, pad) -> (slice, slice)` | Padded bounding box of the mask. |
| `crop_window(image, center, size) -> ndarray` | Centered window (clipped to bounds). |
| `locate_beam(image, mask) -> (y, x)` | Brightest pixel within the mask. |
| `roi_slices(beam_spot, config) -> (slice, slice)` | Beam ROI slices. |
| `dark_roi_slices(beam_spot, config) -> (slice, slice)` | Dark ROI slices. |
| `integrate_frame(image, mask, config, detector, exposure_s) -> FrameIntegration` | Locate beam and integrate one frame. |

### `class FrameIntegration` (frozen)

Fields: `beam_spot`, `counts_spot`, `counts_dark`, `net` (a `Value`),
`counts_ratio`, `is_saturated`.

---

## `pxr_reduce.metadata`

| Function | Description |
|---|---|
| `build_metadata_table(records, config) -> DataFrame` | Standardize/rename/round raw header records. |
| `prepare_metadata(df, config) -> (DataFrame, float)` | Full cleanup + energy/theta/q; returns applied theta offset. |
| `clean_monitors(df, config) -> DataFrame` | Sanitize monitors, add exposure offset. |
| `label_scans(df, config) -> DataFrame` | Assign scan indices from `sam_th` jumps. |
| `direct_beam_mask(df) -> Series[bool]` | Flag direct-beam (i0) frames per scan. |
| `determine_sam_th_offset(df) -> float` | Auto theta offset from geometry. |
| `apply_energy_and_theta(df, config) -> (DataFrame, float)` | Apply offsets, compute wavelength + q. |

Constants: `HEADER_NAMES`, `HEADER_RESOLUTIONS`, `SAM_Z_BEAM_MOVE`.

---

## `pxr_reduce.beam_fit`

| Item | Description |
|---|---|
| `class BeamShape` | `centroid_y, centroid_x, sigma_y, sigma_x, amplitude, success`. |
| `estimate_moments(window) -> BeamShape` | Beam center/sigmas from background-subtracted moments. |
| `aggregate_shapes(shapes) -> BeamShape \| None` | Median over successful shapes. |
| `roi_from_shape(shape, n_sigma, *, minimum=3) -> (height, width)` | Rectangular ±Nσ ROI. |

---

## `pxr_reduce.uncertainty`

| Item | Description |
|---|---|
| `class Value` | `(value, sigma)` pair; `.rel` = relative uncertainty. |
| `roi_variance(roi, detector, exposure_s) -> float` | Variance of the summed ROI (ADU²). |
| `net_counts(spot, dark, detector, exposure_s) -> Value` | Background-subtracted counts + uncertainty. |
| `scale(v, factor) -> Value` | Multiply by an exact factor. |
| `ratio(numerator, denominator) -> Value` | Divide, propagating relative errors. |
| `product(a, b) -> Value` | Multiply, propagating relative errors. |
| `apply_scale_factor(r, scale_factor) -> Value` | Apply a stitch scale factor with propagation. |

---

## `pxr_reduce.io.fits_io`

| Item | Description |
|---|---|
| `read_fits(path) -> (dict, ndarray)` | Header + image. |
| `read_fits_header(path) -> dict` | Header only. |
| `read_fits_image(path) -> ndarray` | Image only (float). |
| `class ImageStore` | Lazy, LRU-cached image access by frame index. |

### `class ImageStore`

```python
ImageStore(paths: Mapping[int, Path | str], cache_size: int = 64)
```

| Method | Description |
|---|---|
| `get(index) / [index] -> ndarray` | Image, loading + caching on demand. |
| `indices() -> list[int]` | Sorted frame indices. |
| `path(index) -> Path` | File path for a frame. |
| `iter_images() -> Iterator[(int, ndarray)]` | Stream `(index, image)` pairs. |
| `stack(indices=None) -> ndarray` | 3D stack (bounded subsets only). |
| `clear_cache() -> None` | Drop cached images. |

---

## `pxr_reduce.viewer`

| Function / class | Description |
|---|---|
| `select_indices(loader, **conditions) -> list[int]` | Sorted frame indices matching metadata. |
| `frame_figure(loader, fits_index) -> Figure` | Headless figure: image + overlays + scalars. |
| `class FrameBrowser(loader, indices=None, **conditions)` | Interactive Prev/Next browser; call `.show()`. |

---

## `pxr_reduce.utils`

**`units`**

| Function | Description |
|---|---|
| `energy_to_wavelength(val) -> float` | eV ↔ Å (self-inverse form). |
| `theta_to_q(theta, lam) -> float` | Angle (rad) + wavelength (Å) → q (Å⁻¹). |
| `q_to_theta(q, lam) -> float` | q + wavelength → angle (deg). |

**`name`**

| Function | Description |
|---|---|
| `infer_index_regex(filenames, *, index_group="index", prefix_group=None) -> str` | Infer the filename regex capturing the frame index. |

**`image`**

| Function | Description |
|---|---|
| `dezinger_image(image, med_result=None, threshold=10, size=3) -> ndarray` | Replace hot pixels using a median reference. |

---

## `pxr_reduce.cli`

Typer application (`app`) with two commands:

- `run FOLDER [options]` — reduce a folder to `.dat` + plots. See
  [How-to §1](how-to.md#1-command-line) or `pxr-reduce run --help`.
- `list-detectors` — print registered detector names.
