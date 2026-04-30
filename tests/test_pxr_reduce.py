"""Tests for pxr_reduce."""

from typer.testing import CliRunner

from pxr_reduce import __version__
from pxr_reduce.cli import app

runner = CliRunner()


def test_version() -> None:
    """Test that version is defined."""
    assert __version__ is not None
    assert isinstance(__version__, str)


def test_cli_version() -> None:
    """Test CLI version command."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_cli_hello_default() -> None:
    """Test CLI hello command with default argument."""
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0
    assert "Hello, World!" in result.stdout


def test_cli_hello_with_name() -> None:
    """Test CLI hello command with custom name."""
    result = runner.invoke(app, ["hello", "Test"])
    assert result.exit_code == 0
    assert "Hello, Test!" in result.stdout
