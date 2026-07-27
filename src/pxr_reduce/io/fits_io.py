"""FITS loading and lazy, memory-bounded image access.

The central design change of the refactor is to keep large 2D images OUT of the
metadata DataFrame. Instead the loader holds an :class:`ImageStore` that maps a
frame index to a file path and reads pixel data on demand, caching only the most
recently used images. This keeps memory flat regardless of dataset size while
making "rebuild an image for debugging/validation" a single call.

FITS layout note: images produced at BL11.0.1.2 store metadata in the primary
header (HDU 0) and the pixel array in HDU 2. These indices are named constants
below so a different layout can be supported by changing them in one place.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# HDU indices for the BL11.0.1.2 FITS layout.
METADATA_HDU = 0
IMAGE_HDU = 2

# Memory-mapping FITS data fails with OSError [Errno 22] on many network/SMB
# shares on Windows, so files are read fully into memory instead.
_MEMMAP = False


def read_fits_header(path: Path | str) -> dict[str, Any]:
    """Read the metadata header of a FITS file into a plain dictionary.

    Args:
        path: Path to the FITS file.

    Returns:
        Dictionary of header keyword/value pairs, with the multi-valued
        ``COMMENT`` card removed.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read (e.g. corrupt or partially written);
            the message includes the offending path.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not a valid file.")
    try:
        with fits.open(path, memmap=_MEMMAP) as hdul:
            header = hdul[METADATA_HDU].header
            return {key: header[key] for key in header if key != "COMMENT"}
    except OSError as exc:
        raise OSError(f"Failed to read FITS header from {path}: {exc}") from exc


def read_fits_image(path: Path | str) -> NDArray[np.floating]:
    """Read the pixel data of a FITS file as a float array.

    Args:
        path: Path to the FITS file.

    Returns:
        The 2D image as a float array.

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the file cannot be read; the message includes the path.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not a valid file.")
    try:
        with fits.open(path, memmap=_MEMMAP) as hdul:
            return np.asarray(hdul[IMAGE_HDU].data, dtype=float)
    except OSError as exc:
        raise OSError(f"Failed to read FITS image from {path}: {exc}") from exc


def read_fits(path: Path | str) -> tuple[dict[str, Any], NDArray[np.floating]]:
    """Read both the header and the image from a FITS file.

    Args:
        path: Path to the FITS file.

    Returns:
        A ``(header, image)`` tuple.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    return read_fits_header(path), read_fits_image(path)


class ImageStore:
    """Lazy, memory-bounded accessor for a set of FITS images by frame index.

    Holds only file paths; images are read on first access and cached in an LRU
    of bounded size. This keeps memory usage independent of the number of frames.

    Args:
        paths: Mapping of frame index to FITS file path.
        cache_size: Maximum number of images held in memory at once.
    """

    def __init__(self, paths: Mapping[int, Path | str], cache_size: int = 64) -> None:
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1.")
        self._paths: dict[int, Path] = {i: Path(p) for i, p in paths.items()}
        self._cache_size = cache_size
        self._cache: OrderedDict[int, NDArray[np.floating]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._paths)

    def __contains__(self, index: int) -> bool:
        return index in self._paths

    def indices(self) -> list[int]:
        """Return frame indices in sorted order."""
        return sorted(self._paths)

    def path(self, index: int) -> Path:
        """Return the file path for a frame index.

        Args:
            index: Frame index.

        Returns:
            The path to the FITS file.

        Raises:
            KeyError: If the index is not present.
        """
        try:
            return self._paths[index]
        except KeyError:
            raise KeyError(f"No image registered for index {index}.") from None

    def get(self, index: int) -> NDArray[np.floating]:
        """Return the image for a frame index, loading and caching on demand.

        Args:
            index: Frame index.

        Returns:
            The 2D image as a float array.

        Raises:
            KeyError: If the index is not present.
        """
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        image = read_fits_image(self.path(index))
        self._cache[index] = image
        self._cache.move_to_end(index)
        if len(self._cache) > self._cache_size:
            # Evict least-recently-used; routine operation, not logged.
            self._cache.popitem(last=False)
        return image

    def __getitem__(self, index: int) -> NDArray[np.floating]:
        return self.get(index)

    def iter_images(self) -> Iterator[tuple[int, NDArray[np.floating]]]:
        """Iterate over ``(index, image)`` pairs in sorted index order.

        Images are streamed one at a time; this does not hold the whole dataset
        in memory.
        """
        for index in self.indices():
            yield index, self.get(index)

    def stack(self, indices: list[int] | None = None) -> NDArray[np.floating]:
        """Return a 3D array of the requested images stacked along axis 0.

        .. warning::
            This materializes every requested image in memory at once. Use only
            for a bounded subset (e.g. a single scan or a viewer selection), not
            the full dataset.

        Args:
            indices: Frame indices to stack; defaults to all, in sorted order.

        Returns:
            Array of shape ``(n, height, width)``.
        """
        if indices is None:
            indices = self.indices()
        return np.stack([self.get(i) for i in indices], axis=0)

    def clear_cache(self) -> None:
        """Drop all cached images, freeing memory."""
        self._cache.clear()
