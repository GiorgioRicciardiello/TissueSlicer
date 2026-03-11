"""
imaging.tiles
=============

Tile-grid utilities for memory-efficient processing of large OME-TIFF images.

When an image is too large to process in memory (e.g., 30-100 GB OME-TIFF),
the standard approach is to partition it into a regular grid of tiles and
process each tile independently, then aggregate or stitch results.

This module provides:

1. :class:`TileSpec` — a descriptor for a tile grid (no data, just geometry).
2. :func:`compute_tile_grid` — produce the list of ``(y, x, h, w)`` rectangles
   that cover an image without overlapping.
3. :func:`iter_channel_tiles` — yield ``(tile_array, y, x, h, w)`` for every
   tile of a channel, loading one tile at a time from an
   :class:`~imaging.reader.OmeTiffReader`.

Patterns derived from MCMICRO pipeline (recyze.py ``base_tiles``) — original
source not modified.

Usage example (colocalization)
-------------------------------
>>> from library.imaging import OmeTiffReader
>>> from library.imaging import iter_channel_tiles
>>> import numpy as np
>>>
>>> with OmeTiffReader("image.ome.tiff") as r:
...     for (t0, y, x, h, w), (t1, *_) in zip(
...         iter_channel_tiles(r, channel_idx=0, tile_size=1024),
...         iter_channel_tiles(r, channel_idx=5, tile_size=1024),
...     ):
...         # Pearson correlation on this tile pair
...         a = t0.astype(np.float32).ravel()
...         b = t1.astype(np.float32).ravel()
...         r_val = np.corrcoef(a, b)[0, 1]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generator

import numpy as np

if TYPE_CHECKING:
    from library.imaging.reader import OmeTiffReader


@dataclass(frozen=True)
class TileSpec:
    """Geometry descriptor for a tile grid over a 2D image.

    No pixel data is stored here — this is purely a geometric specification
    that can be computed once and reused across multiple channels or images
    that share the same spatial dimensions.

    Parameters
    ----------
    tile_size:
        Target square tile side length in pixels.
    image_height:
        Full image height in pixels (at the relevant pyramid level).
    image_width:
        Full image width in pixels (at the relevant pyramid level).

    Notes
    -----
    The last row and column of tiles may be smaller than ``tile_size`` if
    the image dimensions are not exact multiples of ``tile_size``.
    """

    tile_size: int
    image_height: int
    image_width: int

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Number of tiles as ``(n_rows, n_cols)``."""
        n_rows = math.ceil(self.image_height / self.tile_size)
        n_cols = math.ceil(self.image_width / self.tile_size)
        return (n_rows, n_cols)

    @property
    def n_tiles(self) -> int:
        """Total number of tiles."""
        n_rows, n_cols = self.grid_shape
        return n_rows * n_cols

    @property
    def coverage_fraction(self) -> float:
        """Fraction of image area covered (always 1.0 for non-overlapping tiles)."""
        return 1.0


def compute_tile_grid(
    image_height: int,
    image_width: int,
    tile_size: int,
) -> list[tuple[int, int, int, int]]:
    """Compute the list of non-overlapping tiles that cover a 2D image.

    Tiles are generated in row-major order (left to right, top to bottom).
    The last row and column of tiles are clipped to the image boundary.

    Parameters
    ----------
    image_height:
        Full image height in pixels.
    image_width:
        Full image width in pixels.
    tile_size:
        Target square tile side length in pixels.

    Returns
    -------
    list[tuple[int, int, int, int]]
        List of ``(y, x, height, width)`` tuples.  Each tuple describes one
        tile: ``y`` and ``x`` are the top-left corner; ``height`` and ``width``
        are the actual extents (may be smaller than ``tile_size`` at edges).

    Example
    -------
    >>> grid = compute_tile_grid(100, 100, tile_size=64)
    >>> len(grid)
    4
    >>> grid[0]
    (0, 0, 64, 64)
    >>> grid[-1]
    (64, 64, 36, 36)
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if image_height <= 0 or image_width <= 0:
        raise ValueError(
            f"image dimensions must be positive, got ({image_height}, {image_width})"
        )

    tiles: list[tuple[int, int, int, int]] = []
    for y in range(0, image_height, tile_size):
        h = min(tile_size, image_height - y)
        for x in range(0, image_width, tile_size):
            w = min(tile_size, image_width - x)
            tiles.append((y, x, h, w))
    return tiles


def iter_channel_tiles(
    reader: "OmeTiffReader",
    channel_idx: int,
    tile_size: int = 1024,
    level: int = 0,
) -> Generator[tuple[np.ndarray, int, int, int, int], None, None]:
    """Iterate over all tiles of one channel, loading one tile at a time.

    Each tile is read from disk only when the generator advances to it.
    Peak memory per iteration is ``tile_size * tile_size * itemsize`` bytes
    (e.g., ~2 MB for a 1024×1024 uint16 tile).

    Parameters
    ----------
    reader:
        An open :class:`~imaging.reader.OmeTiffReader` instance.
    channel_idx:
        0-based channel index.
    tile_size:
        Target square tile side length in pixels.
    level:
        Pyramid level (0 = full resolution).

    Yields
    ------
    tile : np.ndarray
        2D array of shape ``(h, w)`` with actual tile contents.
    y : int
        Row offset of the tile's top-left corner in the full image.
    x : int
        Column offset of the tile's top-left corner in the full image.
    h : int
        Actual tile height (may be < ``tile_size`` at image boundary).
    w : int
        Actual tile width (may be < ``tile_size`` at image boundary).

    Example
    -------
    >>> from library.imaging import OmeTiffReader
    >>> from library.imaging import iter_channel_tiles
    >>> import numpy as np
    >>>
    >>> with OmeTiffReader("image.ome.tiff") as r:
    ...     thresholds = []
    ...     for tile, y, x, h, w in iter_channel_tiles(r, channel_idx=0, tile_size=2048):
    ...         nonzero = tile[tile > 0]
    ...         if nonzero.size > 0:
    ...             from skimage.filters import threshold_otsu
    ...             thresholds.append(threshold_otsu(tile))
    ...     global_threshold = float(np.median(thresholds))
    """
    _, img_h, img_w = reader.level_shape(level)
    grid = compute_tile_grid(img_h, img_w, tile_size)
    for y, x, h, w in grid:
        tile = reader.get_channel_roi(channel_idx, y=y, x=x, height=h, width=w, level=level)
        yield tile, y, x, h, w


def iter_channel_pair_tiles(
    reader: "OmeTiffReader",
    channel_a: int,
    channel_b: int,
    tile_size: int = 1024,
    level: int = 0,
) -> Generator[tuple[np.ndarray, np.ndarray, int, int, int, int], None, None]:
    """Iterate over matching tiles for a pair of channels simultaneously.

    Loads both channels for the same tile in a single disk read using
    :meth:`~imaging.reader.OmeTiffReader.get_channels_roi` (orthogonal
    selection), which is more efficient than two separate single-channel reads.

    This is the recommended pattern for pairwise colocalization analysis
    (Pearson, Manders, overlap coefficient) over the full image.

    Parameters
    ----------
    reader:
        An open :class:`~imaging.reader.OmeTiffReader` instance.
    channel_a:
        0-based index of the first channel.
    channel_b:
        0-based index of the second channel.
    tile_size:
        Target square tile side length in pixels.
    level:
        Pyramid level.

    Yields
    ------
    tile_a : np.ndarray
        2D tile for ``channel_a``.
    tile_b : np.ndarray
        2D tile for ``channel_b``, same shape as ``tile_a``.
    y : int
        Row offset.
    x : int
        Column offset.
    h : int
        Tile height.
    w : int
        Tile width.

    Example
    -------
    >>> import numpy as np
    >>> from library.imaging import OmeTiffReader
    >>> from library.imaging import iter_channel_pair_tiles
    >>>
    >>> pearson_values = []
    >>> with OmeTiffReader("image.ome.tiff") as r:
    ...     for ta, tb, y, x, h, w in iter_channel_pair_tiles(r, 0, 5, tile_size=2048):
    ...         a = ta.astype(np.float32).ravel()
    ...         b = tb.astype(np.float32).ravel()
    ...         pearson_values.append(np.corrcoef(a, b)[0, 1])
    >>> print(f"Mean Pearson r across tiles: {np.nanmean(pearson_values):.4f}")
    """
    _, img_h, img_w = reader.level_shape(level)
    grid = compute_tile_grid(img_h, img_w, tile_size)
    for y, x, h, w in grid:
        pair = reader.get_channels_roi(
            [channel_a, channel_b], y=y, x=x, height=h, width=w, level=level
        )
        yield pair[0], pair[1], y, x, h, w
