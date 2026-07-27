# pxr-reduce

Tools for reducing **polarized X-ray reflectivity (PXR)** data collected with the
CCD/CMOS area detector at the RSoXS beamline (BL 11.0.1.2) at the Advanced Light
Source.

`pxr-reduce` loads a series of 2D `.fits` frames, integrates the specular beam on
each frame into a single intensity point, normalizes and stitches the points into
a 1D reflectivity curve, and exports the result (with full provenance) as a `.dat`
file plus plots. It tracks photon energy, polarization, and instrument metadata
throughout.

## What it does

```
.fits frames ──► load metadata ──► track beam per scan ──► integrate each frame
                                                                  │
        1D R vs q  ◄── stitch & scale ◄── normalize to i0 ◄──────┘
                     │
                     └──► export .dat + I-vs-q plots (with provenance header)
```

- **Loads** a folder of `.fits` frames, inferring the sample name and frame order
  from the filenames.
- **Processes** each frame: cleans it, tracks the beam per scan (median-filter +
  local-argmax, seeded on the direct beam), and integrates a beam ROI minus a dark
  ROI into background-subtracted counts with propagated uncertainty.
- **Reduces** the per-frame counts into reflectivity `R` vs momentum transfer `q`:
  normalizes to the direct beam (i0), detects stitch boundaries (from a `sam_th`
  back-step or a change in exposure/HOS/slits), and fits scale factors to join the
  segments.
- **Exports** the reduced curve as a `.dat` file with an expansive provenance
  header (values rounded to their justified significant figures), plus one I-vs-q
  PNG per (energy, polarization).
- **Batches** many samples from a single TOML config, pooling each sample's scans.
- **Inspects** individual frames with a metadata-driven image viewer.

## Key design points

- **Images stay out of the metadata table.** Frames are loaded lazily on demand
  (`ImageStore`), so memory stays flat regardless of dataset size.
- **Typed configuration.** All reduction parameters live in a single
  `ReductionConfig` dataclass that also serializes into the export header. Batch
  runs are driven by an editable `RunConfig` TOML file.
- **Swappable detectors.** Every detector-specific value (pixel size, bit depth,
  gain, noise) lives in a `DetectorSpec` object in a registry.
- **Data-driven ROI (optional).** The beam ROI can be sized from a 2D-moments fit
  of the direct-beam frames.

## Installation

`pxr-reduce` is a [uv](https://docs.astral.sh/uv/) project.

```bash
# Run without installing (from the repo root)
uv run pxr-reduce run "path/to/data"

# Or install as a global CLI tool (run from anywhere)
uv tool install --editable /path/to/pxr-reduce
pxr-reduce run "path/to/data"
```

See the [How-to guide](how-to.md) for the full CLI and Python workflows.

## Quick start

```bash
# Reduce a folder of .fits files; outputs go to ./results
pxr-reduce run "path/to/data"

# Preview first (writes nothing), with verbose logging
pxr-reduce run "path/to/data" --dry-run -v

# Batch: discover scans, write a config, reduce many samples at once
pxr-reduce scan-samples "path/to/beamtime"   # prints a [samples] map to paste
pxr-reduce init-config                        # -> ./reduction_config.toml (edit it)
pxr-reduce batch                              # -> results/<sample>.dat per sample
```

```python
from pathlib import Path
from pxr_reduce import PXRLoader, ReductionConfig, ReducedDataset

files = list(Path("path/to/data").glob("*.fits"))
loader = PXRLoader(files, ReductionConfig())
loader.process()
refl = loader.reduce()                       # DataFrame: q, R, R_err, energy, ...
ReducedDataset.from_loader(loader).save("results/MySample.dat")
```

## Documentation map

| Page | Contents |
|---|---|
| [How-to guide](how-to.md) | CLI usage, Python workflows, viewing, exporting, combining, quick modes |
| [Configuration reference](configuration.md) | Every `ReductionConfig` parameter, its meaning, default, and units |
| [API reference](api-reference.md) | Public modules, classes, and functions |
| [Tutorial notebook](notebooks/tutorial.ipynb) | Guided walkthrough: load → examine → process → export |
| [Design & refactor plan](design/refactor-and-features.md) | Architecture and roadmap |

## Status

The reduction pipeline is functional end to end. **Detector noise parameters are
currently placeholders** — reflectivity `R` and `q` are correct, but absolute
`R_err` values are not yet physical until measured detector specs are supplied
(the export header flags this with `[PLACEHOLDER noise specs]`).
