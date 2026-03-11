"""
Manifest JSON writer for tissue extraction output.

Generates a ``manifest.json`` at the output directory root that indexes all
extracted regions, their spatial metadata, and output file paths for
downstream batch processing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from library.imaging_analysis.tissue_extractor import TissueRegion

logger = logging.getLogger(__name__)


def write_manifest(
    regions: list[TissueRegion],
    source_file: str,
    image_shape: tuple[int, int, int],
    metadata: dict,
    channel_roles: dict[str, str] | None,
    detection_level: int,
    scale_factors: tuple[float, float],
    seed: int,
    output_dir: str,
) -> Path:
    """Write ``manifest.json`` summarizing the extraction run.

    Parameters
    ----------
    regions : list[TissueRegion]
        All extracted regions (successful and failed).
    source_file : str
        Path to the input OME-TIFF.
    image_shape : tuple[int, int, int]
        Full-resolution (C, H, W) shape.
    metadata : dict
        Output of ``extract_ome_metadata()``.
    channel_roles : dict[str, str] | None
        Optional mapping of channel names to roles (e.g., ``{"Hoechst": "nucleus"}``).
    detection_level : int
        Pyramid level used for tissue detection.
    scale_factors : tuple[float, float]
        (scale_y, scale_x) from detection level to full resolution.
    seed : int
        Random seed used.
    output_dir : str
        Base output directory.

    Returns
    -------
    Path
        Path to the written ``manifest.json``.
    """
    channel_names = metadata.get("channel_names", [])
    pixel_size = metadata.get("pixel_size_um", {"x_um": 0.0, "y_um": 0.0})

    channels_list = []
    for i, name in enumerate(channel_names):
        entry: dict = {"index": i, "name": name}
        if channel_roles and name in channel_roles:
            entry["role"] = channel_roles[name]
        channels_list.append(entry)

    manifest = {
        "version": "1.0",
        "source_file": source_file,
        "extraction_timestamp": datetime.now().isoformat(),
        "image_shape": list(image_shape),
        "pixel_size_um": {"x": pixel_size["x_um"], "y": pixel_size["y_um"]},
        "num_regions": len(regions),
        "detection_level": detection_level,
        "detection_scale_factor": list(scale_factors),
        "seed": seed,
        "channels": channels_list,
        "regions": [_region_to_dict(r) for r in regions],
    }

    output_path = Path(output_dir) / "manifest.json"
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(f"Manifest written: {output_path}")
    return output_path


def _region_to_dict(region: TissueRegion) -> dict:
    """Serialize a TissueRegion to a JSON-compatible dict.

    Parameters
    ----------
    region : TissueRegion
        Region to serialize.

    Returns
    -------
    dict
        JSON-serializable dictionary.
    """
    return {
        "region_id": region.region_id,
        "folder": f"region_{region.region_id:03d}",
        "bbox": {
            "ymin": region.bbox_ymin,
            "xmin": region.bbox_xmin,
            "ymax": region.bbox_ymax,
            "xmax": region.bbox_xmax,
        },
        "padded_bbox": {
            "ymin": region.padded_ymin,
            "xmin": region.padded_xmin,
            "ymax": region.padded_ymax,
            "xmax": region.padded_xmax,
        },
        "area_um2": region.area_um2,
        "centroid_px": {"y": region.centroid_y_px, "x": region.centroid_x_px},
        "origin_um": {"y": region.origin_um_y, "x": region.origin_um_x},
        "files": {
            "hdf5": region.output_hdf5 or None,
            "ometiff": region.output_ometiff or None,
            "thumbnail": region.output_thumbnail or None,
        },
        "success": region.success,
        "error": region.error_message or None,
    }
