# pxr-reduce

Reducing pxr data from bl 11.0.1.2

## Installation

Install using pip:

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

### Command Line Interface

pxr-reduce provides a command-line interface:

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

Clone the repository and install dependencies:

```bash
git clone https://github.com/als-rsoxs/pxr-reduce.git
cd pxr-reduce
uv sync --group dev
```

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run ty check
```

### Prek Hooks

Install prek hooks:

```bash
prek install
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/als-rsoxs/pxr-reduce/blob/main/LICENSE) file for details.
