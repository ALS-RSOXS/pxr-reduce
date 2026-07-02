from typer.testing import CliRunner

from pxr_reduce.cli import app

runner = CliRunner()


def test_list_detectors():
    result = runner.invoke(app, ["list-detectors"])
    assert result.exit_code == 0
    assert "default" in result.stdout


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
