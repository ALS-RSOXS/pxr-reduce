from collections.abc import Iterable
from pathlib import Path
from typing import Any


def ensure_paths(items: Iterable[str | Path | Any] | str | Path) -> list[Path]:
    """
    Ensure `items` is a list of `pathlib.Path` objects.
    - Accepts a single `str` or `Path`, or an iterable of `str`/`Path`.
    - Converts strings to `Path`.
    - Raises TypeError for unsupported element types.
    """
    # Handle single path passed directly
    if isinstance(items, (str, Path)):
        items = [items]

    try:
        iterator = iter(items)
    except TypeError as e:
        raise TypeError("Input must be a Path/str or an iterable of Path/str") from e

    out = []
    for i, it in enumerate(iterator):
        match it:
            case Path():
                out.append(it)
            case str():
                out.append(Path(it))
            case _:
                raise TypeError(
                    f"""
                    Item at index {i} has unsupported type {type(it)};
                    expected str or pathlib.Path got {type(it)}
                    """
                ) from None
    return out
