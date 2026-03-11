"""
imaging
=======

Lazy, memory-efficient I/O library for large OME-TIFF microscopy images.

Designed for 30-100 GB Orion multi-stain images where loading the full file
into RAM is not feasible.  Uses tifffile + zarr as the backend.

Install
-------
pip install tifffile zarr ome-types numpy

Import
------
Add the project root to sys.path, then::

    from library.imaging import OmeTiffReader, extract_ome_metadata

Or add library/ to sys.path and import directly::

    from imaging import OmeTiffReader, extract_ome_metadata

Quick start
-----------
    import sys
    sys.path.insert(0, r"C:/Users/.../OrionVascularImg")
    from library.imaging import OmeTiffReader, extract_ome_metadata

    path = r"C:/path/to/image.ome.tiff"
    meta = extract_ome_metadata(path)
    print(meta["channel_names"][0])   # 01_Nucleus_Hoechst
    print(meta["pixel_size_um"])      # {"x_um": 0.325, "y_um": 0.325}

    with OmeTiffReader(path) as r:
        print(r.shape)          # (20, 37041, 65603)
        level = r.select_level(min_spatial_dim=512)
        roi = r.get_channel_roi(0, y=0, x=0, height=2048, width=2048)

    from library.imaging import iter_channel_pair_tiles
    with OmeTiffReader(path) as r:
        for ta, tb, y, x, h, w in iter_channel_pair_tiles(r, 0, 15, tile_size=2048):
            pass  # colocalization per tile
"""

# Use relative imports so this package works whether imported as
# ``library.imaging`` (project root on sys.path) or ``imaging``
# (library/ on sys.path).
from .metadata import (
    extract_ome_metadata,
    get_channel_index,
    get_channel_names,
    get_pixel_size_um,
)
from .reader import OmeTiffReader
from .tiles import (
    TileSpec,
    compute_tile_grid,
    iter_channel_pair_tiles,
    iter_channel_tiles,
)

__all__ = [
    # Reader
    "OmeTiffReader",
    # Metadata
    "extract_ome_metadata",
    "get_channel_names",
    "get_pixel_size_um",
    "get_channel_index",
    # Tiles
    "TileSpec",
    "compute_tile_grid",
    "iter_channel_tiles",
    "iter_channel_pair_tiles",
]
