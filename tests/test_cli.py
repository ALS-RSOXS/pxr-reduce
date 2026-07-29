from typer.testing import CliRunner

from pxr_reduce.cli import app

runner = CliRunner()


def test_list_detectors():
    result = runner.invoke(app, ["list-detectors"])
    assert result.exit_code == 0
    assert "default" in result.stdout


def test_init_config_writes_starter(tmp_path):
    dest = tmp_path / "reduction_config.toml"
    result = runner.invoke(app, ["init-config", str(dest)])
    assert result.exit_code == 0, result.stdout
    assert dest.exists()
    # A second write refuses to overwrite.
    assert runner.invoke(app, ["init-config", str(dest)]).exit_code != 0


def test_scan_samples_prints_map(tmp_path):
    folder = tmp_path / "beamtime" / "s1"
    folder.mkdir(parents=True)
    for frame in range(3):
        (folder / f"GlassA_90001-{frame:05d}.fits").write_bytes(b"x")
    result = runner.invoke(app, ["scan-samples", str(tmp_path / "beamtime")])
    assert result.exit_code == 0, result.stdout
    assert "[samples]" in result.stdout
    assert "GlassA = [90001]" in result.stdout


def test_batch_dry_run_lists_samples(synthetic_scan_folder, tmp_path):
    from pxr_reduce.config import ReductionConfig
    from pxr_reduce.run_config import RunConfig, run_config_to_toml_str

    # Reuse the synthetic scan folder as the parent; name its scan via [samples].
    folder = synthetic_scan_folder()
    for frame, old in enumerate(sorted(folder.glob("*.fits"))):
        old.rename(folder / f"GlassA_90001-{frame:05d}.fits")

    cfg = RunConfig(
        parent_dir=folder,
        results_root=tmp_path / "results",
        reduction=ReductionConfig(roi_height=9, roi_width=9),
        samples={"GlassA": [90001]},
    )
    cfg_path = tmp_path / "reduction_config.toml"
    cfg_path.write_text(run_config_to_toml_str(cfg))

    result = runner.invoke(app, ["batch", "--config", str(cfg_path), "--dry-run"])
    assert result.exit_code == 0, result.stdout
    assert "GlassA" in result.stdout
    assert not (tmp_path / "results" / "GlassA.dat").exists()


def test_run_writes_dat_and_plots_to_results_dir(synthetic_scan_folder, tmp_path):
    folder = synthetic_scan_folder()
    results = tmp_path / "results"
    result = runner.invoke(
        app,
        ["run", str(folder), "--results-dir", str(results),
         "--roi-height", "9", "--roi-width", "9"],
    )
    assert result.exit_code == 0, result.stdout
    assert (results / "MF999A.dat").exists()
    assert (results / "MF999A_plots").is_dir()
    # results must NOT land in the data folder
    assert not (folder / "MF999A.dat").exists()


def test_run_defaults_to_cwd_results(fits_writer, frame_builders, tmp_path, monkeypatch):
    # Run from tmp_path as cwd and confirm outputs go to ./results (not the data dir).
    beam_image, frame_header = frame_builders
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    peaks = [10000.0, 10000.0, 4000.0, 2000.0, 900.0, 400.0]
    sam_z = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    sam_th = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    for i, (peak, z, th) in enumerate(zip(peaks, sam_z, sam_th)):
        fits_writer(data / f"MF999A_{i}.fits", beam_image(peak), frame_header(th, z))
    result = runner.invoke(
        app, ["run", str(data), "--roi-height", "9", "--roi-width", "9", "--no-plots"]
    )
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "results" / "MF999A.dat").exists()
    assert not (data / "MF999A.dat").exists()


def test_run_dry_run_writes_nothing(synthetic_scan_folder, tmp_path):
    folder = synthetic_scan_folder()
    results = tmp_path / "results"
    result = runner.invoke(
        app,
        ["run", str(folder), "--results-dir", str(results),
         "--roi-height", "9", "--roi-width", "9", "--dry-run"],
    )
    assert result.exit_code == 0, result.stdout
    assert not (results / "MF999A.dat").exists()
    assert "[dry-run]" in result.stdout


def test_run_quick_mode(synthetic_scan_folder, tmp_path):
    folder = synthetic_scan_folder()
    results = tmp_path / "results"
    result = runner.invoke(
        app,
        ["run", str(folder), "--results-dir", str(results),
         "--roi-height", "9", "--roi-width", "9", "--quick", "--no-plots"],
    )
    assert result.exit_code == 0, result.stdout
    assert (results / "MF999A.dat").exists()


def test_run_no_matching_files_errors(tmp_path):
    result = runner.invoke(app, ["run", str(tmp_path), "--pattern", "*.nope"])
    assert result.exit_code != 0


def test_run_diagnostics_creates_folder(synthetic_scan_folder, tmp_path):
    folder = synthetic_scan_folder()
    results = tmp_path / "results"
    result = runner.invoke(
        app,
        ["run", str(folder), "--results-dir", str(results),
         "--roi-height", "9", "--roi-width", "9", "--no-plots", "--diagnostics"],
    )
    assert result.exit_code == 0, result.stdout
    diag_dirs = list(results.glob("*_diagnostics"))
    assert diag_dirs, "no *_diagnostics folder created"
    assert list(diag_dirs[0].glob("*.png"))


def test_run_with_config_file(synthetic_scan_folder, tmp_path):
    from pxr_reduce.config import ReductionConfig
    from pxr_reduce.run_config import RunConfig, run_config_to_toml_str

    folder = synthetic_scan_folder()
    cfg_path = tmp_path / "tuned.toml"
    cfg = RunConfig(
        reduction=ReductionConfig(roi_height=9, roi_width=9, mask_threshold=90)
    )
    cfg_path.write_text(run_config_to_toml_str(cfg))
    results = tmp_path / "results"
    result = runner.invoke(
        app,
        ["run", str(folder), "--config", str(cfg_path),
         "--results-dir", str(results), "--no-plots"],
    )
    assert result.exit_code == 0, result.stdout
    assert (results / "MF999A.dat").exists()


def test_run_missing_config_file_errors(synthetic_scan_folder, tmp_path):
    folder = synthetic_scan_folder()
    result = runner.invoke(
        app, ["run", str(folder), "--config", str(tmp_path / "nope.json")]
    )
    assert result.exit_code != 0
