# pxr-reduce

Tools for reducing **polarized X-ray reflectivity (PXR)** data collected with the
CCD/CMOS area detector at the RSoXS beamline (BL 11.0.1.2) at the Advanced Light
Source.

`pxr-reduce` loads a series of 2D `.fits` frames, integrates the specular beam on
each frame into a single intensity point, normalizes and stitches the points into
a 1D reflectivity curve (R vs q), and exports the result — with full provenance —
as a `.dat` file plus plots. Photon energy, polarization, and instrument metadata
are tracked throughout.

```
.fits frames ──► load metadata ──► track beam per scan ──► integrate each frame
                                                                  │
        1D R vs q  ◄── stitch & scale ◄── normalize to i0 ◄──────┘
                     │
                     └──► export .dat + I-vs-q plots (with provenance header)
```

## Features

- **Lazy image handling** — frames load on demand, so memory stays flat on large
  datasets; any frame can be rebuilt for debugging.
- **Simple, robust beam tracking** — per-scan median-filter + local-argmax
  tracker (seeded on the direct beam, cropped to the search region for speed).
- **Typed configuration** — all reduction parameters in one `ReductionConfig`,
  serialized into every export header.
- **Batch processing** — reduce many samples from one editable TOML config,
  pooling each sample's scans; discovery maps scan IDs to files automatically.
- **Condition-aware stitching** — boundaries detected from a `sam_th` back-step or
  a change in exposure/HOS/slits, with `diagnose_stitches()` for inspection.
- **Swappable detectors** — detector-specific values live in a `DetectorSpec`
  registry; add new detectors without touching reduction code.
- **Propagated uncertainty** — per-pixel Poisson + read/dark noise carried through
  ROI integration, normalization, and stitch scaling.
- **Data-driven ROI** (optional) — size the beam ROI from a fit of the direct beam.
- **Rich export** — `.dat` with an expansive provenance header, significant-figure
  rounding, per-(energy, polarization) plots, and multi-dataset combining.
- **Metadata-driven viewer** — query frames and inspect them with beam/ROI overlays.
- **CLI** — reduce a folder or a whole batch to `.dat` + plots without a notebook.

## Installation

This is a [uv](https://docs.astral.sh/uv/) project.

```bash
# Run without installing using default reduction parameters (from the repo root)
uv run pxr-reduce run "path/to/data"

# Or install as a global CLI tool (run from anywhere)
uv tool install --editable /path/to/pxr-reduce
pxr-reduce run "path/to/data"
```

## Quick start

**Command line** — outputs go to `./results` (not the data folder):

```bash
pxr-reduce run "path/to/data"            # reduce a folder
pxr-reduce run "path/to/data" --dry-run -v   # preview first, verbose
```

**Changing reduction parameters from the command line.** Any flag overrides the
corresponding `ReductionConfig` default (see the
[Configuration reference](docs/configuration.md)):

```bash
pxr-reduce run "path/to/pxr/data" \
    --detector cmos_11012 \
    --roi-height 30 --roi-width 30 \
    --energy-offset 0.2 \
    --sam-th-offset -0.01 \
    --results-dir "reduced/run1" \
    -v
```

- `--detector cmos_11012` — pick a detector from the registry
- `--roi-height 30 --roi-width 30` — beam integration ROI (pixels)
- `--energy-offset 0.2` — shift the recorded photon energy (eV)
- `--sam-th-offset -0.01` — fixed sample-theta offset (deg)
- `--results-dir "reduced/run1"` — where outputs go (relative to the current directory)
- `-v` — verbose logging

Fast first look — skip stitch scaling and dezingering, load every 4th frame:

```bash
pxr-reduce run "path/to/data" --quick --no-dezinger --subsample 4 --no-plots
```

Size the ROI automatically from the direct beam instead of fixing it:

```bash
pxr-reduce run "path/to/data" --fit-roi --roi-n-sigma 3
```

Run `pxr-reduce run --help` for the full list of options.

**Batch many samples** from one TOML config — each sample pools the frames of its
listed scan IDs (the static 5-digit block in the filenames):

```bash
pxr-reduce scan-samples "path/to/beamtime"   # print a [samples] map to paste
pxr-reduce init-config                        # write ./reduction_config.toml, edit it
pxr-reduce batch --dry-run                    # preview each sample's files + output
pxr-reduce batch                              # -> results/<sample>.dat (+ plots)
```

**Python:**

```python
from pathlib import Path
from pxr_reduce import PXRLoader, ReductionConfig, ReducedDataset

files = list(Path("path/to/data").glob("*.fits"))
loader = PXRLoader(files, ReductionConfig())
loader.process()
refl = loader.reduce()                        # DataFrame: q, R, R_err, energy, ...
ReducedDataset.from_loader(loader).save("results/MySample.dat")
```

## Documentation

| Page | Contents |
|---|---|
| [Overview](docs/index.md) | What the package does and how it fits together |
| [How-to guide](docs/how-to.md) | CLI and Python workflows: run, view, export, combine |
| [Configuration reference](docs/configuration.md) | Every reduction parameter and detector field |
| [API reference](docs/api-reference.md) | Public modules, classes, and functions |
| [Tutorial notebook](docs/notebooks/tutorial.ipynb) | Guided walkthrough: load → examine → reduce → export |
| [Design & refactor plan](docs/design/refactor-and-features.md) | Architecture and roadmap |

## Development

```bash
uv run --group test pytest        # run the test suite
uv run --group dev ruff check src # lint
```

## Status

The reduction pipeline is functional end to end. **Detector noise parameters are
currently placeholders** — reflectivity `R` and `q` are correct, but absolute
`R_err` values are not yet physical until measured detector specs are supplied
(the export header flags this with `[PLACEHOLDER noise specs]`).

## License

See [LICENSE](LICENSE).
