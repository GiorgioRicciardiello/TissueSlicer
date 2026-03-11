"""
imaging.reader
==============

Lazy, memory-efficient reader for large OME-TIFF files.

Design principles
-----------------
- Uses ``tifffile`` + ``zarr`` backend: no full image is ever loaded into RAM.
  A 30-100 GB file is opened in milliseconds; only the slices you request are
  read from disk.
- Supports multi-resolution (pyramid) OME-TIFFs: callers can choose a
  downsampled level for fast overviews or analysis that does not require full
  resolution.
- All public methods return ``np.ndarray`` (concrete data) or ``zarr.Array``
  (lazy proxy).  The docstring explicitly states which.
- Thread-worker counts are set before file open and respected by tifffile's
  internal thread pool.

Patterns derived from MCMICRO pipeline (recyze.py, story.py) — original
source not modified.

Usage example
-------------
>>> from library.imaging import OmeTiffReader
>>> with OmeTiffReader("image.ome.tiff") as r:
...     print(r.shape)          # (20, 27075, 26235)
...     level = r.select_level(min_spatial_dim=512)
...     roi = r.get_channel_roi(0, y=0, x=0, height=2048, width=2048)
...     tiles = list(r.iter_tiles(0, tile_size=1024))
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Union

import numpy as np
import tifffile
import zarr


class OmeTiffReader:
    """Lazy reader for large OME-TIFF files via tifffile + zarr.

    Opening the file does not load any pixel data. Data is only pulled from
    disk when you explicitly call a method that returns pixel values.

    Parameters
    ----------
    path:
        Absolute or relative path to the OME-TIFF file.
    num_workers:
        Number of I/O worker threads passed to ``tifffile``.
        0 (default) lets tifffile auto-scale based on available CPUs.
        Pass an explicit integer to reproduce behaviour across environments.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be opened as a TIFF or has no image series.

    Notes
    -----
    Use as a context manager (``with OmeTiffReader(...) as r:``) to ensure
    the underlying file handle is closed properly.  Direct instantiation is
    also supported; call :meth:`close` explicitly when done.
    """

    def __init__(self, path: Union[str, Path], num_workers: int = 0) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"OME-TIFF not found: {path.resolve()}")

        # Configure tifffile thread pool BEFORE opening the file.
        if num_workers > 0:
            tifffile.TIFF.MAXWORKERS = num_workers
            tifffile.TIFF.MAXIOWORKERS = num_workers * 5
        else:
            cpu_count = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else (os.cpu_count() or 4)
            )
            tifffile.TIFF.MAXWORKERS = cpu_count
            tifffile.TIFF.MAXIOWORKERS = cpu_count * 5

        try:
            # is_ome=False: skip slow OME-XML parse at open time; metadata is
            # read on demand via imaging.metadata functions.
            self._tiff = tifffile.TiffFile(str(path), is_ome=False)
        except Exception as exc:
            raise ValueError(f"Cannot open file as TIFF: {path.name}") from exc

        if not self._tiff.series:
            raise ValueError(f"No image series found in {path.name}")

        self._series = self._tiff.series[0]
        self._path = path

        # Open the base pyramid level (level "0") as a zarr array.
        # This is a lazy proxy — no pixel data is loaded yet.
        self._zarr_store: zarr.Array = zarr.open(self._series.aszarr(), mode="r")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "OmeTiffReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying file handle."""
        self._tiff.close()

    # ------------------------------------------------------------------
    # Introspection properties  (no I/O)
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Absolute path to the source file."""
        return self._path.resolve()

    @property
    def shape(self) -> tuple[int, int, int]:
        """Full-resolution image shape as ``(C, H, W)``."""
        arr = self._zarr_store["0"]
        if arr.ndim == 3:
            return (arr.shape[0], arr.shape[1], arr.shape[2])
        # Single-channel image stored as (H, W)
        return (1, arr.shape[0], arr.shape[1])

    @property
    def num_channels(self) -> int:
        """Number of channels (stains) in the image."""
        return self.shape[0]

    @property
    def dtype(self) -> np.dtype:
        """Pixel data type (e.g. ``uint16``)."""
        return self._zarr_store["0"].dtype

    @property
    def num_levels(self) -> int:
        """Number of pyramid resolution levels."""
        return len(self._series.levels)

    def level_shape(self, level: int) -> tuple[int, int, int]:
        """Shape ``(C, H, W)`` at a given pyramid level.

        Parameters
        ----------
        level:
            0 = full resolution; higher values are progressively downsampled.
        """
        arr = self._zarr_store[str(level)]
        if arr.ndim == 3:
            return (arr.shape[0], arr.shape[1], arr.shape[2])
        return (1, arr.shape[0], arr.shape[1])

    # ------------------------------------------------------------------
    # Pyramid level selection
    # ------------------------------------------------------------------

    def select_level(self, min_spatial_dim: int = 200) -> int:
        """Return the index of the smallest pyramid level whose spatial
        dimensions are each ``>= min_spatial_dim``.

        This pattern (derived from story.py) lets you quickly select an
        appropriate downsampled level for analysis tasks that don't require
        full resolution — e.g., channel statistics, histogram fitting,
        coarse tissue detection — without manually inspecting level shapes.

        Parameters
        ----------
        min_spatial_dim:
            Minimum required size in **both** H and W dimensions.

        Returns
        -------
        int
            Pyramid level index (0 = full resolution).

        Raises
        ------
        ValueError
            If no level satisfies the constraint (image is smaller than
            ``min_spatial_dim`` in at least one dimension).

        Example
        -------
        >>> level = reader.select_level(min_spatial_dim=512)
        >>> shape = reader.level_shape(level)  # (C, H, W), H >= 512, W >= 512
        """
        # Iterate from finest (0) to coarsest; pick the last one still valid.
        # Equivalent to: reversed levels → first level with dims >= target.
        for level_idx in range(self.num_levels - 1, -1, -1):
            _, h, w = self.level_shape(level_idx)
            if h >= min_spatial_dim and w >= min_spatial_dim:
                return level_idx
        raise ValueError(
            f"No pyramid level with both spatial dims >= {min_spatial_dim}. "
            f"Image shape at level 0: {self.level_shape(0)}"
        )

    # ------------------------------------------------------------------
    # Channel access — lazy proxy
    # ------------------------------------------------------------------

    def get_channel(self, channel_idx: int, level: int = 0) -> zarr.Array:
        """Return a **lazy** 2D zarr array for one channel.

        No pixel data is loaded until you slice the returned array.
        Use this when you want to defer loading to a later step or pass
        the array to a library that understands zarr (e.g. dask).

        Parameters
        ----------
        channel_idx:
            0-based channel index. Use :func:`imaging.metadata.get_channel_index`
            to look up by name.
        level:
            Pyramid level (0 = full resolution).

        Returns
        -------
        zarr.Array
            2D lazy array of shape ``(H, W)``.

        Example
        -------
        >>> arr = reader.get_channel(0)         # lazy
        >>> patch = arr[0:1024, 0:1024]         # triggers disk read
        """
        self._validate_channel(channel_idx)
        zarr_level = self._zarr_store[str(level)]
        if zarr_level.ndim == 2:
            return zarr_level
        return zarr_level[channel_idx]  # type: ignore[index]

    # ------------------------------------------------------------------
    # Channel access — concrete numpy arrays (actual I/O)
    # ------------------------------------------------------------------

    def get_channel_roi(
        self,
        channel_idx: int,
        y: int,
        x: int,
        height: int,
        width: int,
        level: int = 0,
    ) -> np.ndarray:
        """Read a spatial region of interest from one channel.

        Only the requested rectangle is read from disk — not the full channel.

        Parameters
        ----------
        channel_idx:
            0-based channel index.
        y, x:
            Top-left corner of the region (row, column).
        height, width:
            Extent of the region in pixels.
        level:
            Pyramid level (0 = full resolution).

        Returns
        -------
        np.ndarray
            2D array of shape ``(height, width)`` and ``self.dtype``.

        Example
        -------
        >>> roi = reader.get_channel_roi(0, y=0, x=0, height=2048, width=2048)
        >>> roi.shape
        (2048, 2048)
        """
        self._validate_channel(channel_idx)
        zarr_level = self._zarr_store[str(level)]
        y_end = min(y + height, zarr_level.shape[-2])
        x_end = min(x + width, zarr_level.shape[-1])
        if zarr_level.ndim == 2:
            return np.asarray(zarr_level[y:y_end, x:x_end])
        return np.asarray(zarr_level[channel_idx, y:y_end, x:x_end])

    def get_channels_roi(
        self,
        channel_indices: list[int],
        y: int,
        x: int,
        height: int,
        width: int,
        level: int = 0,
    ) -> np.ndarray:
        """Read multiple channels over the same spatial region in one call.

        Uses zarr's ``get_orthogonal_selection`` to avoid reading channels
        that are not requested.  Memory footprint = ``len(channel_indices) *
        height * width * itemsize``.

        This is the correct pattern for colocalization: load only the pair
        (or small set) of channels you need for a given ROI.

        Parameters
        ----------
        channel_indices:
            List of 0-based channel indices to read.
        y, x:
            Top-left corner of the region.
        height, width:
            Extent of the region.
        level:
            Pyramid level (0 = full resolution).

        Returns
        -------
        np.ndarray
            3D array of shape ``(len(channel_indices), height, width)``.

        Example
        -------
        >>> # Read channels 0 (Hoechst) and 5 (CD31) over a 2048×2048 tile
        >>> pair = reader.get_channels_roi([0, 5], y=0, x=0, height=2048, width=2048)
        >>> pair.shape
        (2, 2048, 2048)
        """
        for idx in channel_indices:
            self._validate_channel(idx)
        zarr_level = self._zarr_store[str(level)]
        y_end = min(y + height, zarr_level.shape[-2])
        x_end = min(x + width, zarr_level.shape[-1])
        if zarr_level.ndim == 2:
            # Single-channel image: ignore channel_indices, return (1, H, W)
            return np.asarray(zarr_level[y:y_end, x:x_end])[np.newaxis]
        result = zarr_level.get_orthogonal_selection(
            (channel_indices, slice(y, y_end), slice(x, x_end))
        )
        return np.asarray(result)

    # ------------------------------------------------------------------
    # Max projection
    # ------------------------------------------------------------------

    def max_projection(
        self,
        channel_indices: list[int],
        y: int = 0,
        x: int = 0,
        height: int | None = None,
        width: int | None = None,
        level: int = 0,
    ) -> np.ndarray:
        """Pixel-wise maximum across a set of channels over a spatial region.

        Useful for generating overview composite images or for detecting any
        signal presence across a marker group without loading all channels
        simultaneously at full resolution.

        Parameters
        ----------
        channel_indices:
            Channels to reduce.
        y, x:
            Top-left corner of the region.  Default: origin (0, 0).
        height, width:
            Extent.  Default: full image height / width at the given level.
        level:
            Pyramid level.

        Returns
        -------
        np.ndarray
            2D array of shape ``(height, width)``.

        Example
        -------
        >>> # Max projection of channels 1-19 (all markers) at level 2
        >>> proj = reader.max_projection(list(range(1, 20)), level=2)
        """
        _, lh, lw = self.level_shape(level)
        height = height if height is not None else lh
        width = width if width is not None else lw
        multichannel = self.get_channels_roi(channel_indices, y, x, height, width, level)
        return np.max(multichannel, axis=0)

    # ------------------------------------------------------------------
    # Tile iteration
    # ------------------------------------------------------------------

    def iter_tiles(
        self,
        channel_idx: int,
        tile_size: int = 1024,
        level: int = 0,
    ) -> Generator[tuple[np.ndarray, int, int], None, None]:
        """Iterate over non-overlapping tiles of one channel.

        Each iteration yields a concrete numpy tile — only one tile is in
        memory at a time.  Use this for any channel-wise analysis that must
        cover the full image without loading it (thresholding, per-tile
        statistics, colocalization metrics).

        Parameters
        ----------
        channel_idx:
            0-based channel index.
        tile_size:
            Square tile side length in pixels.  The last row and column of
            tiles will be smaller if the image dimensions are not multiples
            of ``tile_size``.
        level:
            Pyramid level.

        Yields
        ------
        tile : np.ndarray
            2D array of shape ``(tile_h, tile_w)`` for the current tile.
        y_offset : int
            Row offset of this tile within the full level.
        x_offset : int
            Column offset of this tile within the full level.

        Example
        -------
        >>> for tile, y, x in reader.iter_tiles(0, tile_size=1024):
        ...     threshold = skimage.filters.threshold_otsu(tile)
        ...     mask_tile = tile > threshold
        ...     # stitch mask_tile into a pre-allocated output array
        """
        self._validate_channel(channel_idx)
        _, h, w = self.level_shape(level)
        for y_off in range(0, h, tile_size):
            for x_off in range(0, w, tile_size):
                tile = self.get_channel_roi(
                    channel_idx,
                    y=y_off,
                    x=x_off,
                    height=tile_size,
                    width=tile_size,
                    level=level,
                )
                yield tile, y_off, x_off

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_channel(self, channel_idx: int) -> None:
        n = self.num_channels
        if not (0 <= channel_idx < n):
            raise IndexError(
                f"channel_idx {channel_idx} out of range for image with "
                f"{n} channel(s)."
            )
