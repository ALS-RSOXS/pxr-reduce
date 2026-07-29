# How-to guide

Practical recipes for operating `pxr-reduce`, from the command line and from
Python. For the meaning of each parameter see the
[Configuration reference](configuration.md); for signatures see the
[API reference](api-reference.md).

---

## 1. Command line

The CLI is the fastest way to reduce a folder without opening a notebook.

### Setup

```bash
# Option A — run from the repo without installing
cd /path/to/pxr-reduce
uv run pxr-reduce run "path/to/data"

# Option B — install once, run from anywhere
uv tool install --editable /path/to/pxr-reduce
pxr-reduce run "path/to/data"
```

Paths with spaces must be quoted: `pxr-reduce run "D:/ALS/2020 Nov/MF114A"`.

### Reduce a folder

```bash
pxr-reduce run "path/to/data"
```

- Globs `*.fits` in the folder, infers the sample name and frame order from the
  filenames, processes and reduces, then writes outputs to **`./results`**
  (relative to your current directory — *not* the data folder).
- Produces `results/<sample>.dat` and a `results/<sample>_plots/` folder with one
  I-vs-q PNG per (energy, polarization).

### Preview before writing

```bash
pxr-reduce run "path/to/data" --dry-run -v
```

`--dry-run` loads, processes, and reduces (so you see the real point count) but
writes nothing. `-v` adds INFO/DEBUG logging (frame counts, beam region size,
fitted ROI, etc.). This is the recommended first command on any new dataset.

### Fast previews

```bash
# Skip stitch scaling (avoids overlap pitfalls), skip plots
pxr-reduce run "path/to/data" --quick --no-plots

# Also skip the median-filter/dezinger step and load every 4th frame
pxr-reduce run "path/to/data" --quick --no-dezinger --subsample 4
```

| Flag | Effect |
|---|---|
| `--quick` | Skip stitch detection/scaling; reflectivity is i0-normalized only. |
| `--no-dezinger` | Skip median-filter/dezinger (much faster, noisier). |
| `--subsample N` | Load every Nth frame. |
| `--no-plots` | Do not export PNGs. |
| `--no-dedup` | Do not average duplicate (theta, energy, polarization) points. |

### Size the ROI from the direct beam

```bash
pxr-reduce run "path/to/data" --fit-roi --roi-n-sigma 3
```

Fits the direct-beam (i0) frames to 2D moments and sizes the beam ROI to
±`roi-n-sigma` of the fitted beam width. See
[ROI from the direct beam](#5-size-the-roi-from-the-direct-beam).

### Choosing the output location

```bash
pxr-reduce run "path/to/data" --results-dir "reduced/run1"   # relative to cwd
pxr-reduce run "path/to/data" -o "D:/analysis/MF114A.dat"    # explicit .dat path
```

### Other useful flags

```bash
pxr-reduce run "path/to/data" \
    --detector cmos_11012 \
    --roi-height 40 --roi-width 40 \
    --energy-offset 0.2 \
    --sam-th-offset -0.01 \
    --pattern "*.fits"
```

ROI flags override the config default only when supplied; otherwise the
`ReductionConfig` defaults are used. See `pxr-reduce run --help` for the full list.

### List detectors

```bash
pxr-reduce list-detectors
```

### Batch many samples from a TOML config

`run` reduces one folder; `batch` reduces many **samples** described by a config.
Each sample lists the scan IDs (the static 5-digit block in the filenames) whose
frames are pooled into one reduction.

```bash
# 1. Discover the scans under a parent folder (prints a ready-to-paste map)
pxr-reduce scan-samples "path/to/beamtime"

# 2. Write a documented starter config and edit [paths]/[samples]
pxr-reduce init-config            # -> ./reduction_config.toml

# 3. Preview, then run
pxr-reduce batch --dry-run        # lists each sample, scan IDs, file count, output
pxr-reduce batch                  # writes results_root/<sample>.dat (+ plots)
pxr-reduce batch --sample B1A1_NEdge_XRR   # just one sample (repeatable)
```

`batch` resolves its config as `--config PATH` → `./reduction_config.toml` →
built-in defaults. The tree under `parent_dir` is searched recursively, so a
scan's frames may sit in their own sub-folder or be mixed together. A sample that
fails is logged and skipped, not fatal to the rest of the batch. See the
[Batch runs (TOML)](configuration.md#batch-runs-toml) section for every config key.

The single-folder `run --config FILE.toml` also accepts a TOML config and uses its
`[reduction]` section (the JSON config of earlier versions is retired).

---

## 2. Python API

Full control lives in the `PXRLoader` class.

```python
from pathlib import Path
from pxr_reduce import PXRLoader, ReductionConfig

files = list(Path("path/to/data").glob("*.fits"))

config = ReductionConfig(
    detector="cmos_11012",
    roi_height=40,
    roi_width=40,
)

loader = PXRLoader(files, config)   # loads metadata; images loaded lazily
loader.process()                    # track beam per scan, integrate every frame
refl = loader.reduce()              # -> DataFrame of the 1D reflectivity curve
```

`reduce()` returns a pandas DataFrame with columns `scan, energy, polarization,
sam_th, q, R, R_err`.

`process()` uses the standard **simple** tracker (median-filter + local-argmax).
Tune it per call:

```python
loader.process(search_radius=45,    # px the beam may drift between frames
               filter_size=5,       # median-filter kernel for finding
               progress=True,       # live per-frame bar + timing summary
               verbose=False)       # per-frame stage timings when profiling
```

The older SNR-gated tracker is the deprecated `loader.process_snr()` (kept for
comparison; emits a `DeprecationWarning`).

### Quick reduction (no scaling)

```python
preview = loader.reduce(apply_scale=False)   # i0-normalized only, no stitching
```

### Diagnose stitching

If a stitch factor looks wrong (or a segment is missing) with no error, inspect
every detected boundary:

```python
loader.diagnose_stitches()   # one row per boundary: trigger, conditions_changed,
                             # num_stitch_points, scale, scale_err, failed
```

`num_stitch_points == 0` or `failed == True` pinpoints a stitch that isn't
working; an empty table means no boundary was detected (check that a watched
condition changes and that `sam_th` steps back into overlap).

### Rebuild any image for debugging

Images are not kept in the table; pull them on demand:

```python
raw = loader.get_image(42)          # raw frame for fits_index 42
clean = loader.get_clean_image(42)  # trimmed + dezingered
```

---

## 3. Examine frames with the viewer

Query frames by metadata and inspect them with beam/ROI overlays.

```python
# Which frames match some metadata?
subset = loader.query(energy=250.0, sam_th=(0.0, 5.0))   # tuple = inclusive range

# Render a single frame (headless Figure; good for scripts/notebooks)
from pxr_reduce.viewer import frame_figure
fig = frame_figure(loader, fits_index=42)
fig.savefig("frame42.png")

# Interactive browser: cycle through a metadata selection with Prev/Next buttons
from pxr_reduce.viewer import FrameBrowser
FrameBrowser(loader, energy=250.0, sam_th=(0.0, 5.0)).show()
```

The viewer shows the cleaned image (log scale), the beam position, the beam and
dark ROI boxes, the integration mask (when one is present, e.g. after
`process_snr`), and a side panel of scalar readouts (raw counts, reduced
intensity, SNR, saturation).

---

## 4. Export results

```python
from pxr_reduce import ReducedDataset

dataset = ReducedDataset.from_loader(loader)

# Write .dat + a sibling <stem>_plots/ folder
dataset.save("results/MF114A.dat")

# Or write the parts separately
dataset.save_dat("results/MF114A.dat")
dataset.save_plots("results/MF114A_plots")

# Preview only
dataset.save("results/MF114A.dat", dry_run=True)
```

The `.dat` file is tab-delimited with a `#`-commented provenance header:
software version + git commit, collection vs. reduction timestamps, the full
config and detector spec, the fitted ROI/beam sigma (if used), energies and
polarizations present, and the uncertainty model. It reads back cleanly with
`pandas.read_csv(path, sep="\t", comment="#")`.

**Significant-figure rounding.** On export, values are rounded to their justified
precision (the in-memory table keeps full precision, so plots are unaffected):
`R` and `R_err` to the PDG significant figures of `R_err` (via the `uncertainties`
package), `q` to the precision implied by propagating the angular step onto it,
and `sam_th` to `angle_decimals` places. Tune with `save(..., angle_decimals=4)`.
Line endings are written cleanly (LF), fixing an earlier double-newline artifact.

### Combine two datasets (e.g. two polarizations)

```python
spol = ReducedDataset.from_loader(loader_spol)
ppol = ReducedDataset.from_loader(loader_ppol)

combined = spol.combine(ppol)          # or ReducedDataset.combine_all([a, b, c])
combined.save("results/MF114A_both.dat")
```

The combined header preserves the provenance of **every** source, so a single
file documents all inputs.

---

## 5. Size the ROI from the direct beam

To avoid accidentally integrating non-specular signal, size the ROI from the
shape of the direct beam (i0) rather than a fixed rectangle:

```python
config = ReductionConfig(roi_from_beam_fit=True, roi_n_sigma=3.0)
loader = PXRLoader(files, config)
loader.process()

print(loader.beam_shape)              # fitted sigma_y, sigma_x, centroid, ...
print(loader.config.roi_height, loader.config.roi_width)  # ROI set from the fit
```

`process()` fits the direct-beam frames with 2D moments, aggregates them (median),
and sets the ROI to ±`roi_n_sigma` of the fitted width. Saturated i0 frames are
skipped; if no usable direct-beam frames are found it falls back to the configured
ROI and warns. The fitted sigma and ROI are recorded in the export header.

---

## 6. Override faulty FITS metadata from scan header files

Some scans were collected with unreliable per-frame metadata in the FITS headers. The
beamline also writes a companion text file per scan holding the authoritative motor
record; point `reduction.header` at the **directory** holding those files to use them
instead:

```toml
[reduction]
# Relative paths resolve against this config file.
header = "headers"
```

Omit the key entirely and nothing changes — the FITS metadata is used exactly as it is
today. From the Python API it is `ReductionConfig(header=Path("headers"))`.

**How frames are matched.** Each row of a header file's `DATA` section ends with the
FITS file it describes, and that filename is the join key. So one directory can hold a
header file per scan, no naming convention is required, and loading a subset of a scan
(`--subsample`) works — unmatched header rows are just counted. If a *loaded* frame has
no header row the reduction **fails** rather than correcting some frames and leaving
others on the faulty metadata.

**What gets overridden.** Only motors the file records as a `Goal`/`Actual` pair —
currently `sam_x`, `sam_y`, `sam_z`, `sam_th`, `det_th`, `energy`, `hos`. Everything
else keeps its FITS value, including `exposure`, the slit apertures, `beam_current`,
`i0`, and `polarization`. The set is derived from the file's own columns, so a header
file that starts recording another motor as a pair is picked up automatically.

**Goal vs Actual.** The nominal `Goal` value becomes the canonical column, so every
correction that keys off intended positions uses it: scan segmentation, the
sample-theta offset, stitch-boundary detection, and stitch overlap matching. `q` is
computed from the `<column>_actual` readback instead. Three columns are kept per
overridden motor:

| column | meaning |
|---|---|
| `sam_th` | `Goal` — drives all corrections, and is the exported angle |
| `sam_th_actual` | readback — `q` is computed from this |
| `sam_th_fits` | the original FITS value, kept so the collection bug stays auditable |

The sample-theta offset is determined from `Goal` and then applied to both readings,
since it corrects the encoder zero. `energy_actual` is deliberately *not* rounded to
`energy_resolution` — that rounding exists to group nominal energies, and applying it
to the readback would erase the very information `q` needs.

A useful side effect: `Goal` values repeat *exactly* across the two passes of an
overlap region, whereas readbacks jitter. Since stitch overlap points are paired by
exact equality on `sam_th`, driving stitching from `Goal` makes overlap matching
markedly more reliable.

The export header records that the metadata did not come from the FITS files, which
header files contributed, and how many frames were overridden.

---

## 7. Tips and troubleshooting

- **Run `--dry-run -v` first.** Confirm the sample name, frame count, beam-region
  size, and reduced point count look right before writing.
- **Check the ROI visually.** Open a frame in the viewer to confirm the beam/dark
  boxes land on the beam and background.
- **`SUSPECT` stitches** are reported by the CLI and in the `.dat` header's *Stitch
  quality* block. The scale was still applied — the flag means it may be wrong, for
  one of two reasons: the overlap points disagree with each other about the scale
  (more than `stitch_max_overlap_rms`), or the fitted scale is far from the value the
  condition change predicts (more than `stitch_max_scale_deviation`). Because
  reflectivity is already normalized by exposure and beam current, a boundary that
  only changes exposure — or a bare angle back-step — must fit ~1.0, so a deviation
  there is a real red flag. A stitch resting on a single overlap angle is always
  suspect: its scale cannot be cross-checked.
- **`FAILED` stitches** could not be fitted at all (usually no overlapping angles).
  The boundary is reported with its reason, later boundaries are still evaluated, and
  every point from the failure to the end of that scan is flagged as having an
  unestablished absolute scale — dropped when `drop_failed_stitch` is set.
- **Use `loader.diagnose_stitches()`** for the full per-boundary table (trigger,
  changed conditions, overlap count, scale, `overlap_rms_rel`, `expected_scale`,
  `suspect`, `quality_note`), and `loader.overlap_report()` for every individual
  candidate frame with the reason it was or wasn't used.
- **`--diagnostics` writes a `stitch/` folder** with a subfolder per scan:

  ```text
  <sample>_diagnostics/stitch/
      stitch_summary.md               every boundary, every dropped point
      dropped_points.md               the same, one table for the whole sample
      scan_00/
          boundary_01.png             the stitch itself
          saturated/frame_00041_roi.png   ROI of a saturation-dropped frame
  ```

  The per-boundary figure has two panels. The top is R vs angle: each overlap
  candidate is marked by fate (used, saturated, below the spot/dark cutoff,
  direct-beam, partner dropped), and the post-change segment is drawn both raw and
  divided by the fitted scale, so you can see whether the two segments overlay. The
  bottom is the fit itself — post-change R against pre-change R, one point per
  matched angle, with the fitted through-origin line and (where predictable) the
  expected line. A point off the line is an angle where the segments disagree; one
  or two points means nothing constrains the fit.

  `stitch_summary.md` embeds each figure and names the source `.fits` file for every
  dropped point, plus the ROI image where one was written. Saturated frames that were
  never overlap candidates are listed there too, without images.
- **Saturation is judged on the integrated beam ROI only** — saturation elsewhere on
  the detector does not affect the measured counts and is not flagged. The per-frame
  pixel counts are in the `n_sat_roi` and `n_sat_dark` columns; a non-zero
  `n_sat_dark` means the background estimate is clipped and `counts_ratio` is
  unreliable. Note the check runs *after* dezingering, so an isolated hot pixel is
  replaced before it can flag a frame; what remains is genuine beam clipping.
- **Slow on a network drive?** Reading full frames over a network is I/O-bound;
  copy the data locally, or ask about lazy section reads.
- **`R_err` looks non-physical.** Detector noise parameters are placeholders until
  measured values are supplied (see [Configuration](configuration.md#detector)).
