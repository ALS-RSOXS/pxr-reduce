from pathlib import Path
from typing import Iterable, List, Union


def ensure_paths(items: Union[Iterable[Union[str, Path]], str, Path]) -> List[Path]:
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
    except TypeError:
        raise TypeError("Input must be a Path/str or an iterable of Path/str")

    out = []
    for i, it in enumerate(iterator):
        if isinstance(it, Path):
            out.append(it)
        elif isinstance(it, str):
            out.append(Path(it))
        else:
            raise TypeError(
                f"Item at index {i} has unsupported type {type(it)}; expected str or pathlib.Path"
            )
    return out
