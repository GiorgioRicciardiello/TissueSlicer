"""
imaging.metadata
================

OME-XML metadata extraction for large OME-TIFF files.

Provides lightweight, standalone functions that parse the OME-XML embedded in
TIFF tag 270 using the ``ome-types`` library.  No pixel data is ever loaded.

Patterns derived from MCMICRO pipeline (recyze.py, story.py) — original
source not modified.

Compatibility
-------------
Tested against ome-types >= 0.5.  Uses ``physical_size_x`` (float) +
``physical_size_x_unit`` (UnitsLength enum) — the pint-Quantity API
(``physical_size_x_quantity``) is NOT available in ome-types 0.6.x.

Usage example
-------------
>>> from library.imaging.metadata import extract_ome_metadata, get_channel_names
>>> meta = extract_ome_metadata("image.ome.tiff")
>>> meta['channel_names']
['01_Nucleus_Hoechst', '02_AF1', ...]
>>> meta['pixel_size_um']
{'x_um': 0.325, 'y_um': 0.325}
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import tifffile
from ome_types import from_tiff, from_xml

# Conversion factors to micrometers for ome_types UnitsLength enum values.
_UM_CONVERSION: dict[str, float] = {
    "NANOMETER": 1e-3,
    "MICROMETER": 1.0,
    "MILLIMETER": 1e3,
    "CENTIMETER": 1e4,
    "METER": 1e6,
    "INCH": 25400.0,
    "ANGSTROM": 1e-4,
}


def _unit_to_um(unit: object) -> float:
    """Return factor to convert *unit* to micrometers. Falls back to 1.0 (µm)."""
    if unit is None:
        return 1.0
    name = getattr(unit, "name", str(unit)).upper()
    return _UM_CONVERSION.get(name, 1.0)


def extract_ome_metadata(path: Union[str, Path]) -> dict:
    """Extract structured metadata from an OME-TIFF file.

    No pixel data is read. Returns both summary fields and the full OME object
    for downstream extraction of channel properties (wavelengths, fluorophores, etc.).

    Parameters
    ----------
    path : str or Path
        Path to the OME-TIFF file.

    Returns
    -------
    dict
        Keys:
        - ``channel_names`` (list[str])
        - ``pixel_size_um`` (dict|None) with keys 'x_um', 'y_um'
        - ``image_shape`` (tuple C,H,W)
        - ``dtype`` (str)
        - ``acquisition_date`` (str|None)
        - ``ome_obj`` (ome_types.OME) — full parsed OME object for metadata reconstruction

    Example
    -------
    >>> meta = extract_ome_metadata("FNEL03.ome.tiff")
    >>> meta['channel_names'][0]
    '01_Nucleus_Hoechst'
    >>> meta['pixel_size_um']
    {'x_um': 0.325, 'y_um': 0.325}
    >>> meta['ome_obj']  # Full OME object with channel metadata
    <OME object>
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    try:
        ome = from_tiff(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to parse OME-XML from {path.name}: {exc}") from exc

    px = ome.images[0].pixels

    channel_names: list[str] = []
    for i, ch in enumerate(px.channels):
        name = ch.name or ""
        channel_names.append(name if name.strip() else f"Channel {i}")

    pixel_size_um: dict | None = None
    try:
        x_val = px.physical_size_x
        y_val = px.physical_size_y
        if x_val is not None:
            x_unit = getattr(px, "physical_size_x_unit", None)
            y_unit = getattr(px, "physical_size_y_unit", None)
            x_um = float(x_val) * _unit_to_um(x_unit)
            y_um = float(y_val or x_val) * _unit_to_um(y_unit or x_unit)
            pixel_size_um = {"x_um": x_um, "y_um": y_um}
    except Exception:
        pass

    image_shape: tuple[int, int, int] = (int(px.size_c), int(px.size_y), int(px.size_x))
    dtype_str = str(px.type.value).lower() if hasattr(px.type, "value") else str(px.type).lower()

    acq_date: str | None = None
    try:
        date_val = ome.images[0].acquisition_date
        if date_val is not None:
            acq_date = str(date_val)
    except Exception:
        pass

    return {
        "channel_names": channel_names,
        "pixel_size_um": pixel_size_um,
        "image_shape": image_shape,
        "dtype": dtype_str,
        "acquisition_date": acq_date,
        "ome_obj": ome,
    }


def get_channel_names(path: Union[str, Path]) -> list[str]:
    """Return ordered channel names from an OME-TIFF.

    Example
    -------
    >>> get_channel_names("image.ome.tiff")
    ['01_Nucleus_Hoechst', '02_AF1', ...]
    """
    return extract_ome_metadata(path)["channel_names"]


def get_pixel_size_um(path: Union[str, Path]) -> tuple[float, float]:
    """Return physical pixel size as ``(x_um, y_um)`` in micrometers.

    Raises ValueError if pixel size is absent from metadata.

    Example
    -------
    >>> x_um, y_um = get_pixel_size_um("image.ome.tiff")
    >>> print(f"{x_um} um x {y_um} um")
    0.325 um x 0.325 um
    """
    ps = extract_ome_metadata(path)["pixel_size_um"]
    if ps is None:
        raise ValueError(
            f"No physical pixel size in OME-XML of {Path(path).name}."
        )
    return float(ps["x_um"]), float(ps["y_um"])


def get_channel_index(path: Union[str, Path], name: str) -> int:
    """Return 0-based index of first channel whose name contains *name* (case-insensitive).

    Raises KeyError if not found.

    Example
    -------
    >>> get_channel_index("image.ome.tiff", "CD31")
    15
    """
    query = name.lower()
    for i, ch_name in enumerate(get_channel_names(path)):
        if query in ch_name.lower():
            return i
    raise KeyError(
        f"No channel matching '{name}'. Available: {get_channel_names(path)}"
    )


def get_channel_metadata_from_tiff(path: Union[str, Path]) -> dict:
    """Read OME-XML directly from TIFF tag 270 (ImageDescription).

    Same as :func:`extract_ome_metadata` but uses the raw tifffile tag
    instead of ``ome_types.from_tiff``.  Matches MCMICRO's story.py pattern.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    try:
        with tifffile.TiffFile(str(path), is_ome=False) as tif:
            xml_str = tif.pages[0].tags[270].value
        ome = from_xml(xml_str)
    except Exception as exc:
        raise ValueError(f"Failed to read tag 270 from {path.name}: {exc}") from exc

    px = ome.images[0].pixels
    channel_names: list[str] = []
    for i, ch in enumerate(px.channels):
        n = ch.name or ""
        channel_names.append(n if n.strip() else f"Channel {i}")

    pixel_size_um: dict | None = None
    try:
        x_val = px.physical_size_x
        y_val = px.physical_size_y
        if x_val is not None:
            x_unit = getattr(px, "physical_size_x_unit", None)
            y_unit = getattr(px, "physical_size_y_unit", None)
            pixel_size_um = {
                "x_um": float(x_val) * _unit_to_um(x_unit),
                "y_um": float(y_val or x_val) * _unit_to_um(y_unit or x_unit),
            }
    except Exception:
        pass

    return {
        "channel_names": channel_names,
        "pixel_size_um": pixel_size_um,
        "image_shape": (int(px.size_c), int(px.size_y), int(px.size_x)),
        "dtype": str(px.type.value).lower() if hasattr(px.type, "value") else str(px.type).lower(),
        "acquisition_date": None,
    }
