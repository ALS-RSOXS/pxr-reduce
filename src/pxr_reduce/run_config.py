"""Run configuration: TOML schema for batch reductions.

A single editable ``reduction_config.toml`` describes a whole batch run: where the
data lives (``[paths]``), which tracker and export options to use
(``[tracking]``/``[export]``), the reduction parameters (``[reduction]``, backed
by :class:`~pxr_reduce.config.ReductionConfig`), and a ``[samples]`` map of
sample name to the scan IDs that compose it. The schema here provides defaults; a
real run is fully described by the TOML, which can be serialized back into output
headers for reproducibility.

Resolution order for the config file: an explicit ``--config`` path, then
``reduction_config.toml`` in the current working directory, then the built-in
defaults (documented in the bundled ``default_config.toml``).
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field, fields
from importlib.resources import files
from pathlib import Path
from typing import Any

import tomli_w

from pxr_reduce.config import ReductionConfig

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "reduction_config.toml"
BUNDLED_CONFIG = "default_config.toml"


@dataclass
class RunConfig:
    """Configuration for a batch reduction run.

    Attributes:
        parent_dir: Folder searched (recursively) for the FITS scans.
        results_root: Directory into which ``<sample>.dat`` files are written.
        fits_glob: Glob used to find FITS files under ``parent_dir``.
        scan_number_regex: Optional regex overriding scan-ID extraction; must
            expose a ``scan`` capture group. ``None`` uses the width-based rule.
        tracker: Beam tracker to use: ``"simple"`` or ``"base"``.
        search_radius: Local search radius (px) for tracking; ``None`` falls back
            to ``reduction.drift_distance``.
        filter_size: Median-filter kernel for beam-finding; ``None`` falls back to
            ``reduction.filter_size``.
        angle_decimals: Decimals kept for ``sam_th`` on export (also sets the
            angular step propagated onto q).
        plots: Whether to write I-vs-q PNGs alongside each ``.dat``.
        apply_scale: Whether to apply stitch scaling (False = quick reduction).
        drop_duplicates: Average points sharing the duplicate key.
        duplicate_scope: Granularity of that key — "sweep" (default) exports every
            sweep as its own profile, "scan" merges repeat sweeps within a scan,
            "angle" merges everything sharing (sam_th, energy, polarization).
        reduction: The reduction parameters (image processing, stitching, ...).
        samples: Map of sample name to the list of scan IDs composing it.
    """

    # [paths]
    parent_dir: Path = Path("data")
    results_root: Path = Path("results")
    fits_glob: str = "*.fits"
    scan_number_regex: str | None = None

    # [tracking]
    tracker: str = "simple"
    search_radius: int | None = None
    filter_size: int | None = None

    # [export]
    angle_decimals: int = 4
    plots: bool = True
    apply_scale: bool = True
    drop_duplicates: bool = True
    duplicate_scope: str = "sweep"

    # [reduction]
    reduction: ReductionConfig = field(default_factory=ReductionConfig)

    # [samples]
    samples: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.parent_dir = Path(self.parent_dir)
        self.results_root = Path(self.results_root)
        if self.tracker not in ("simple", "base"):
            raise ValueError(
                f"tracker must be 'simple' or 'base', got {self.tracker!r}."
            )


# Top-level (non-reduction) fields grouped into TOML sections.
_SECTIONS: dict[str, tuple[str, ...]] = {
    "paths": (
        "parent_dir",
        "results_root",
        "fits_glob",
        "scan_number_regex",
    ),
    "tracking": ("tracker", "search_radius", "filter_size"),
    "export": (
        "angle_decimals",
        "plots",
        "apply_scale",
        "drop_duplicates",
        "duplicate_scope",
    ),
}
_PATH_FIELDS: frozenset[str] = frozenset({"parent_dir", "results_root"})
# Fields that may be omitted from the TOML, meaning "unset" (None).
_OPTIONAL_FIELDS: frozenset[str] = frozenset(
    {"scan_number_regex", "search_radius", "filter_size"}
)


def run_config_to_dict(cfg: RunConfig) -> dict[str, Any]:
    """Serialize a run configuration to a nested, TOML-ready dict.

    Path fields become strings, ``None`` values are omitted (TOML has no null),
    and the reduction sub-config and sample map are expanded into their tables.

    Args:
        cfg: The configuration to serialize.

    Returns:
        A nested dict keyed by TOML section.
    """
    out: dict[str, Any] = {}
    for section, names in _SECTIONS.items():
        table: dict[str, Any] = {}
        for name in names:
            value = getattr(cfg, name)
            if value is None:
                continue
            table[name] = str(value) if name in _PATH_FIELDS else value
        out[section] = table
    reduction = cfg.reduction.to_dict()
    # TOML has no null and wants lists, not tuples.
    reduction = {
        k: (list(v) if isinstance(v, tuple) else v)
        for k, v in reduction.items()
        if v is not None
    }
    out["reduction"] = reduction
    out["samples"] = {name: list(scans) for name, scans in cfg.samples.items()}
    return out


def run_config_to_toml_str(cfg: RunConfig) -> str:
    """Return the run configuration as a TOML string (e.g. for headers)."""
    return tomli_w.dumps(run_config_to_dict(cfg))


def load_run_config(path: Path | str | None) -> RunConfig:
    """Load a run configuration from a TOML file (or return defaults).

    Args:
        path: Path to the TOML file, or ``None`` to use built-in defaults.

    Returns:
        A configuration with the file's values applied over the defaults.

    Raises:
        FileNotFoundError: If a path is given but does not exist.
    """
    if path is None:
        return RunConfig()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    kwargs: dict[str, Any] = {}

    for section, names in _SECTIONS.items():
        table = raw.get(section, {})
        for key, value in table.items():
            if key not in names:
                logger.warning("Unknown key '%s.%s' in %s; ignoring.", section, key, path)
                continue
            kwargs[key] = Path(value) if key in _PATH_FIELDS else value

    if "reduction" in raw:
        reduction_config = ReductionConfig.from_dict(raw["reduction"])
        # A relative header directory is resolved against the config file, so a config
        # and its header files can be moved together and referenced from any cwd.
        if (
            reduction_config.header is not None
            and not reduction_config.header.is_absolute()
        ):
            reduction_config.header = path.parent / reduction_config.header
        kwargs["reduction"] = reduction_config
    if "samples" in raw:
        kwargs["samples"] = {
            str(name): [int(s) for s in _as_list(scans)]
            for name, scans in raw["samples"].items()
        }

    recognized = set(_SECTIONS) | {"reduction", "samples"}
    for table_name in raw:
        if table_name not in recognized:
            logger.warning("Unknown table '[%s]' in %s; ignoring.", table_name, path)

    return RunConfig(**kwargs)


def _as_list(value: Any) -> list[Any]:
    """Wrap a scalar in a list; pass a list through unchanged."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def resolve_config_path(explicit: Path | str | None) -> Path | None:
    """Determine which config file to use.

    Resolution order: an explicit ``--config`` path, then
    ``reduction_config.toml`` in the current working directory, then ``None``
    (meaning built-in defaults should be used).

    Args:
        explicit: The path passed via ``--config``, or None.

    Returns:
        The path to load, or None to fall back to :class:`RunConfig` defaults.

    Raises:
        FileNotFoundError: If an explicit path was given but does not exist.
    """
    if explicit is not None:
        explicit = Path(explicit)
        if not explicit.exists():
            raise FileNotFoundError(f"--config file not found: {explicit}")
        return explicit
    cwd_config = Path.cwd() / CONFIG_FILENAME
    return cwd_config if cwd_config.exists() else None


def bundled_default_config_text() -> str:
    """Return the documented bundled ``default_config.toml`` template text."""
    return files("pxr_reduce").joinpath(BUNDLED_CONFIG).read_text(encoding="utf-8")


def write_default_config(path: Path | str) -> Path:
    """Write the bundled config template to ``path`` (adds ``.toml`` if missing).

    Args:
        path: Destination for the starter config.

    Returns:
        The path written.

    Raises:
        FileExistsError: If the destination already exists (never overwrites).
    """
    path = Path(path)
    if path.suffix != ".toml":
        path = path.with_suffix(".toml")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing config: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bundled_default_config_text(), encoding="utf-8")
    return path


def validate_run_config(cfg: RunConfig) -> list[str]:
    """Return a list of human-readable problems with the config (empty if OK).

    Checks that ``parent_dir`` exists and that at least one sample is defined.
    ``results_root`` is an output and is created on demand, so it is not checked.

    Args:
        cfg: The run configuration to validate.

    Returns:
        A list of problem descriptions; empty when the config is usable.
    """
    problems: list[str] = []
    if not cfg.parent_dir.is_dir():
        problems.append(f"parent_dir is not a directory: {cfg.parent_dir}")
    if not cfg.samples:
        problems.append("no samples defined in [samples].")
    known = {f.name for f in fields(RunConfig)}
    assert {n for names in _SECTIONS.values() for n in names} <= known
    return problems
