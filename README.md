# pxr-reduce

[CI](https://github.com/als-rsoxs/pxr-reduce/actions/workflows/ci.yml)
[PyPI version](https://badge.fury.io/py/pxr_reduce)
[codecov](https://codecov.io/gh/als-rsoxs/pxr-reduce)
[Python 3.13+](https://www.python.org/downloads/)
[uv](https://github.com/astral-sh/uv)
[Ruff](https://github.com/astral-sh/ruff)
[ty](https://github.com/astral-sh/ty)
[License: MIT](https://github.com/als-rsoxs/pxr-reduce/blob/main/LICENSE)
[Renovate](https://renovateapp.com/)

Reducing pxr data from bl 11.0.1.2

## Features

- Fast and modern Python toolchain using Astral's tools (uv, ruff, ty)
- Type-safe with full type annotations
- Command-line interface built with Typer
- Comprehensive documentation with MkDocs — [View Docs](https://als-rsoxs.github.io/pxr-reduce/)

## Installation

```bash
pip install pxr_reduce
```

Or using uv (recommended):

```bash
uv add pxr_reduce
```

## Quick Start

```python
import pxr_reduce

print(pxr_reduce.__version__)
```

### CLI Usage

```bash
# Show version
pxr_reduce --version

# Say hello
pxr_reduce hello World
```

## Development

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for package management

### Setup

```bash
git clone https://github.com/als-rsoxs/pxr-reduce.git
cd pxr-reduce
make install
```

### Running Tests

```bash
make test

# With coverage
make test-cov

# Across all Python versions
make test-matrix
```

### Code Quality

```bash
# Run all checks (lint, format, type-check)
make verify

# Auto-fix lint and format issues
make fix
```

### Prek

```bash
prek install
prek run --all-files
```

### Documentation

```bash
make docs-serve
```

## Dependency Updates

This project uses [Renovate](https://renovateapp.com/) to keep dependencies up to date automatically. Renovate will open pull requests when new versions of dependencies are available.

To enable it, install the [Renovate GitHub App](https://github.com/apps/renovate) and grant it access to this repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.