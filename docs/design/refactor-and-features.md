# pxr-reduce — Design & Refactor Plan

**Status:** Implemented (historical planning document)
**Author:** Thomas Ferron (tjferron@lbl.gov)
**Created:** 2026-07-02
**Scope:** Refactor `PrsoxrLoader` and add viewer, export, combine, CLI, and a
corrected uncertainty model.

> **This is the original planning document; the refactor has since been built and
> extended.** For the current package see the
> [Overview](../index.md), [How-to guide](../how-to.md),
> [Configuration reference](../configuration.md), and
> [API reference](../api-reference.md). Notable additions beyond this plan:
> the **simple median-filter beam tracker** is now the standard `PXRLoader.process`
> (the SNR-gated tracker is the deprecated `process_snr`); **condition-aware stitch
> detection** with `diagnose_stitches`; **significant-figure export rounding**; and
> **TOML-driven batch processing** (`RunConfig`, `discovery`, `batch`, and the
> `batch`/`scan-samples`/`init-config` CLI commands).

This document captures a review of the original package and the plan for the
requested features. It follows ALS Photon Science Computing software standards and
scientific-computing safety practices.

---

## 1. Purpose of the package

Load a series of 2D `.fits` images into a loader, reduce them into 1D
Intensity-vs-`q` (or sample-theta) datasets, while tracking photon energy,
polarization, and metadata. It works today but is slow, memory-heavy, hard to
extend, and mishandles uncertainty.

---

## 2. Review of current state

### 2.1 Architecture

Nearly everything lives in one ~1100-line class, `PrsoxrLoader`
([src/pxr_reduce/loader.py](../../src/pxr_reduce/loader.py)), mixing six
concerns: FITS I/O, metadata cleanup, image processing, beam-finding, reduction
math, and plotting. This monolith is the root cause of the slowness, memory
pressure, awkward uncertainty handling, and difficulty adding features.

### 2.2 Performance / memory root cause

Six full-resolution image arrays are stored **per frame** as object-dtype
columns in a single DataFrame (`raw_image`, `filtered_image`, `zinged_image`,
`reduced_image`, `spot`, `dark`), then processed with ~15 `.apply(axis=1)`
calls. Object columns cannot vectorize, and `apply(axis=1)` is a Python loop.
For large FITS × many frames this is exactly why one DataFrame becomes
unmanageable and slow.

### 2.3 Confirmed bugs (fix regardless of new features)

1. **`counts_err` is not an error.**
   [loader.py:550-556](../../src/pxr_reduce/loader.py#L550-L556) computes
   `counts_err` with the *identical formula* to `counts_refl`. Point-wise
   uncertainty is currently just a duplicate of the signal. (Core of §7.)
2. **Scale-factor error is dropped.**
   [loader.py:868-871](../../src/pxr_reduce/loader.py#L868-L871) — propagation of
   `scale_err` into `R_err` after stitching is commented out, so final `R_err`
   ignores stitch-ratio uncertainty.
3. **`__init__` returns `0`** at
   [loader.py:225](../../src/pxr_reduce/loader.py#L225) and
   [:240](../../src/pxr_reduce/loader.py#L240). Must return `None`; should raise.
4. **`sam_th_offset` never stored.**
   [loader.py:431](../../src/pxr_reduce/loader.py#L431) is a no-op statement; the
   computed offset is lost.
5. **`check_spot` membership test wrong.**
   [loader.py:691](../../src/pxr_reduce/loader.py#L691)
   `fits_index not in self.data["fits_index"]` checks the Series *index*, not its
   values.
6. **`subtract_background` is broken.**
   [loader.py:604](../../src/pxr_reduce/loader.py#L604) references
   `process_vars["bkg_sub"]`, which does not exist → `KeyError`. Dead method.
7. **`save_csv` / `save_hdf5` documented but not implemented**
   ([docstring, loader.py:184](../../src/pxr_reduce/loader.py#L184)).

### 2.4 Standards gaps (ALS + safety)

- `print()` used throughout instead of `logging` (ALS §5).
- Essentially no type annotations (ALS §1).
- `List` / `Optional` from `typing` in
  [name.py:2](../../src/pxr_reduce/utils/name.py#L2) instead of built-in
  generics / `X | None`.
- Imports not grouped/sorted; a local import mid-file.
- No tests anywhere (ALS §9).
- `pyproject.toml` has a placeholder description, no `[project.scripts]`, and no
  test extras beyond ruff.
- `requires-python >=3.13` with `pandas>=3.0` / `numpy>=2.4` is bleeding-edge —
  fine, but `groupby.apply(include_groups=...)` semantics are new; pin deps.

---

## 3. Target architecture

The single highest-leverage change — which unlocks nearly every feature — is
**separating the metadata table from the image data, and splitting the god class
into a pipeline.**

### 3.1 Data model — split in two

- **Metadata / results table** (small: scalars only — motor positions, `q`,
  energy, polarization, counts, `R`, `R_err`, beam_spot, flags). Stays a
  DataFrame, stays fast.
- **Image stack** kept *out* of the DataFrame. Preferred approaches, in order:
  - **Lazy, on-demand loading with an LRU cache.** The loader holds only file
    paths; images are read from FITS when needed (integration, viewing,
    debugging) via a `get_image(index)` accessor. This is the clean answer to
    "don't save everything but be able to rebuild images for debugging."
  - For the reduction pass itself, process images in a **streaming / chunked
    loop**, writing only scalar outputs to the table — never holding all images
    at once.
  - For persistence + fast re-analysis, optionally back the stack with **Zarr or
    HDF5** (chunked, compressed, memory-mapped; the safety-approved alternative
    to pickle; dask-ready for later parallelism).

Expected effect: large speedup, ~6× memory reduction, and "quick rebuild for
validation" becomes a one-liner.

### 3.2 Module split (replacing the monolith)

| Module | Responsibility |
|---|---|
| `io/fits_io.py` | Load FITS, extract header, merge AI `.txt` metadata |
| `utils/name.py` | (exists) naming / index inference |
| `config.py` | `ReductionConfig` typed model (replaces `process_vars` dict + decorator) |
| `metadata.py` | Metadata cleanup, offsets, energy rounding, `q` calculation |
| `image_ops.py` | Dezinger, series mask, beam location, ROI, integration on arrays/stacks |
| `uncertainty.py` | Poisson + detector-noise error model and propagation |
| `reduction.py` | Normalize → stitch → scale, as composable stages |
| `loader.py` | Thin orchestrator holding the table + image accessor |
| `dataset.py` | `ReducedDataset` result container (data + provenance) |
| `export.py` | `.dat` writer, plot PNG export, `combine()` |
| `viewer.py` | Interactive image browser |
| `cli.py` | Typer CLI |

### 3.3 Replace `process_vars` + decorator with a typed config

Retire the property-generating decorator in
[utils/attributes.py](../../src/pxr_reduce/utils/attributes.py) in favor of a
typed `ReductionConfig` (`@dataclass` or pydantic `BaseModel`). The decorator is
clever but defeats IDE autocomplete, type checking, and validation. A config
*model* provides all three **and** serializes directly into the export header
(§5) and CLI arguments (§6). This single change pays off in three feature
requests.

---

## 4. Feature: metadata-driven image viewer

Built on the lazy image accessor. Two tiers:

- **Query API:** e.g. `loader.query(energy=250, sam_th=(0, 5), polarization=...)`
  returns matching indices from the metadata table (trivial once images are out
  of the table).
- **Interactive browser:** cycle through the matched set.
  - Notebook: `ipywidgets` sliders.
  - Notebook-free (for the CLI workflow): `matplotlib` `Slider`/`Button`.
  - Each frame overlays beam position + ROI/dark boxes and shows a side panel of
    scalars (raw counts, reduced intensity, `R`, SNR, saturation flag).
  - Reuse the existing `check_spot` plotting logic, refactored to take
    `(image, metadata_row)` rather than reaching into `self`.

---

## 5. Feature: rich `.dat` export

Tab/space-delimited columns `q R R_err energy polarization sam_th` with a
`#`-commented header. Suggested header fields:

- Sample name; **absolute source path(s)** of the FITS folder and AI file.
- **Collection timestamp** (from FITS headers) vs **reduction timestamp**
  (`datetime.now()`).
- **Software provenance:** package version + git commit hash.
- **Full `ReductionConfig`** used (every parameter — why config-as-model matters).
- Reduction summary: # scans, # points, # dropped (failed stitch / saturated /
  i0), energies & polarizations present, stitch points per scan.
- Column names + **units** line.
- Uncertainty-model description (which noise terms were included).

### 5.1 Combine two loaders

`combine([loader_a, loader_b], ...)` in `export.py` concatenates reduced datasets
and **merges provenance from both** into one header (both source paths, configs,
metadata sets, labeled by polarization/source). A `ReducedDataset` container
makes this a merge of two containers. Warn on schema mismatches (e.g. differing
energy grids).

### 5.2 Plot PNGs alongside the `.dat`

On export, write one I-vs-`q` PNG per (energy, polarization) into a `plots/`
folder co-located with the `.dat`, referenced in the header. A method on
`ReducedDataset`.

---

## 6. Feature: CLI reduction runner

A **Typer** app (ALS §8): `pxr-reduce run FOLDER [--config ...] [--quick]
[--dry-run]`, registered under `[project.scripts]`. Point at a folder, use
defaults or overrides, get `.dat` + plots without a notebook.

- **`--quick` mode:** subsample frames (e.g. every Nth) and/or skip stages.
  Strongest argument for composable reduction stages
  (load → integrate → normalize → [stitch] → [scale]) with toggles —
  "reduce to a datapoint but skip scaling to spot overlap problems fast" becomes
  `--no-scale`.
- **`--dry-run`** (safety): print what would be written without writing.

---

## 7. Improvement: uncertainty model (the science)

Detector: CMOS area detector, raw units = counts (ADU). Correct per-point model:

1. **Per-pixel variance (electrons):**
   `var = (signal_ADU − bias)·gain + read_noise² + dark_current·t`,
   `gain` in e⁻/ADU. In ADU only:
   `var_ADU ≈ (signal − bias)/gain + read_noise_ADU² + dark_rate_ADU·t`.
   The Poisson `√N` shot-noise term is currently **absent**.
2. **ROI sum:** variances add → `σ_spot = √(Σ var_pixel)` over the ROI.
3. **Dark subtraction:** `σ² = σ_spot² + σ_dark²` (scale dark variance by area
   ratio if ROI areas differ).
4. **Normalize by exposure·beam_current:** linear scaling divides value and σ by
   the same factor.
5. **Normalize by i₀:** propagate i₀ std (relative errors add in quadrature).
6. **Stitch scale factor:** propagate `scale_err` into `R_err` — currently
   dropped (bug §2.3.2). The ODR already returns `sd_beta`; wire it through.

**Inputs needed from detector to implement precisely:** gain (e⁻/ADU), read
noise (e⁻ or ADU), dark current rate (ADU/s or e⁻/s/pixel) and whether dark
frames exist, and bias/offset level.

---

## 8. Additional standards-driven improvements

- **Logging:** replace all `print()` with `logging.getLogger(__name__)` (ALS §5);
  keep `tqdm` for progress bars.
- **Typing + docstrings:** full annotations and Google-style docstrings on the
  public API (ALS §1–2); fix `List`/`Optional` → `list` / `| None` in `name.py`.
- **Tests:** pytest suite (ALS §9) — units, naming inference, uncertainty
  propagation, and a small synthetic-FITS reduction fixture. None exist today.
- **`pyproject.toml`:** real description, `[project.scripts]` for the CLI,
  `test`/`dev` extras (pytest, ruff), pinned deps (safety).
- **Serialization safety:** HDF5/Zarr for any cached/persisted intermediate data,
  never pickle.
- **CI + docs:** GitHub Actions for pytest + ruff (ALS §12); MkDocs Material with
  `mkdocstrings` (ALS §11).

---

## 9. Suggested sequencing

1. **Split the data model** (images out of the DataFrame, lazy accessor) —
   unblocks speed, memory, viewer, and debugging-rebuild at once.
2. **Introduce `ReductionConfig`** (typed model) — unblocks CLI + export headers.
3. **Fix the uncertainty model** (needs detector numbers from §7).
4. **Refactor reduction into composable stages** — unblocks `--quick` / `--no-scale`.
5. **Build `ReducedDataset` + export** (`.dat`, headers, combine, PNGs).
6. **Viewer**, then **CLI**.
7. Backfill **tests, logging, docstrings, CI, docs** alongside each step.

---

## 10. Open questions / inputs needed

- Detector specs for §7: gain, read noise, dark current rate, bias, dark-frame
  availability.
- Persistence preference for the image stack: pure lazy-load vs. Zarr/HDF5 cache.
- Config model preference: `@dataclass` vs. pydantic `BaseModel`.
- Viewer target: notebook (`ipywidgets`), notebook-free (`matplotlib` widgets),
  or both.
- `.dat` column set and delimiter conventions expected by downstream analysis
  tools.
