"""
Manual tissue region extraction from user-drawn polygons.

Wrapper around the library.imaging_analysis extraction pipeline for
GUI-driven (non-automatic) tissue selection.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable

try:
    from gui.backend.utils.image_loader import ProgressReader
except ImportError:
    from utils.image_loader import ProgressReader

import numpy as np

from library.imaging import extract_ome_metadata, get_pixel_size_um
from library.imaging_analysis.tissue_extractor import TissueRegion
from library.imaging_analysis._writers import save_region
from gui.backend.utils.polygon_ops import (
    polygon_to_mask,
    compute_bbox_from_mask,
    apply_padding,
    force_square,
    scale_bbox_to_full_resolution,
    validate_coordinates,
)

logger = logging.getLogger(__name__)


def extract_region_with_padding(
    loader: Any,
    polygon_coords: List[List[float]],
    selection_name: str,
    padding_px: int = 50,
    force_square: bool = False,
    scale_y: float = 16.0,
    scale_x: float = 16.0,
    output_dir: str = "./extracted_tissues/",
    output_formats: List[str] | None = None,
    verbose: bool = True,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """
    Extract a tissue region defined by user-drawn polygon.

    Pipeline:
    1. Rasterize polygon at downsampled (display) level
    2. Compute tight bounding box
    3. Apply padding and optional square forcing
    4. Scale coordinates to full resolution
    5. Extract all channels at full resolution
    6. Save as HDF5/OME-TIFF with provenance metadata

    Args:
        loader: OmeTiffImageLoader instance (must be at full resolution)
        polygon_coords: List of [x, y] coordinates in display (downsampled) space
        selection_name: Human-readable name for this region
        padding_px: Padding to add around tissue (pixels)
        force_square: If True, expand bbox to square
        scale_y, scale_x: Downsampled → full-resolution scale factors
        output_dir: Directory to save extracted files
        output_formats: List of formats to save ("hdf5", "ometiff")
        verbose: Enable logging
        progress_callback: Optional callable(channel_idx: int) fired after each channel read. Used for real-time progress reporting.

    Returns:
        Dict with keys:
        - output_folder: Path to extracted region folder
        - files: Dict with 'hdf5', 'ometiff', 'thumbnail' paths
        - region: TissueRegion dataclass with metadata

    Raises:
        ValueError: If polygon is invalid or extraction fails
    """
    if output_formats is None:
        output_formats = ["hdf5", "ometiff"]

    logger.info(
        f"Extracting region '{selection_name}': "
        f"{len(polygon_coords)} vertices, padding={padding_px}px, "
        f"force_square={force_square}, scale=({scale_y:.2f}, {scale_x:.2f})"
    )

    # Validate input
    if len(polygon_coords) < 3:
        raise ValueError(f"Polygon must have >= 3 vertices, got {len(polygon_coords)}")

    # --- Step 1: Get downsampled image shape ---
    _, shape_full, _, _ = loader.get_channel_downsampled(0)
    num_channels, img_h_full, img_w_full = shape_full
    _, img_h_display, img_w_display = loader.get_level_shape()

    logger.debug(
        f"Image shape: display={img_h_display}×{img_w_display}, "
        f"full_res={img_h_full}×{img_w_full}"
    )

    # --- Step 2: Convert polygon to mask at display level ---
    mask = polygon_to_mask(
        polygon_coords,
        (img_h_display, img_w_display)
    )

    # Verify mask has area
    area_px_display = np.sum(mask)
    if area_px_display == 0:
        raise ValueError("Polygon does not intersect image (no area in mask)")

    logger.info(f"Polygon rasterized: {area_px_display} pixels at display level")

    # --- Step 3: Compute tight bounding box ---
    bbox_tight = compute_bbox_from_mask(mask)
    if bbox_tight is None:
        raise ValueError("Failed to compute bounding box from mask")

    ymin_t, xmin_t, ymax_t, xmax_t = bbox_tight
    logger.debug(f"Tight bbox (display): ({ymin_t}, {xmin_t}, {ymax_t}, {xmax_t})")

    # --- Step 4: Apply padding ---
    bbox_padded = apply_padding(
        bbox_tight, padding_px, img_h_display, img_w_display
    )
    if bbox_padded is None:
        raise ValueError("Failed to apply padding")

    ymin_p, xmin_p, ymax_p, xmax_p = bbox_padded
    logger.debug(f"Padded bbox (display): ({ymin_p}, {xmin_p}, {ymax_p}, {xmax_p})")

    # --- Step 5: Force square if requested ---
    if force_square:
        bbox_padded = force_square(
            bbox_padded, img_h_display, img_w_display, expand_mode='outer'
        )
        if bbox_padded is None:
            raise ValueError("Failed to force square")

        ymin_p, xmin_p, ymax_p, xmax_p = bbox_padded
        logger.debug(f"Force-square bbox (display): ({ymin_p}, {xmin_p}, {ymax_p}, {xmax_p})")

    # --- Step 6: Scale coordinates to full resolution ---
    bbox_fullres = scale_bbox_to_full_resolution(
        bbox_padded, scale_y, scale_x
    )
    if bbox_fullres is None:
        raise ValueError("Failed to scale coordinates to full resolution")

    ymin_f, xmin_f, ymax_f, xmax_f = bbox_fullres
    logger.info(f"Scaled bbox (full-res): ({ymin_f}, {xmin_f}, {ymax_f}, {xmax_f})")

    # --- Step 7: Validate final coordinates ---
    is_valid = validate_coordinates(
        bbox_fullres, img_h_full, img_w_full, name="extracted_region"
    )
    if not is_valid:
        raise ValueError(
            f"Extracted region out of bounds: bbox={bbox_fullres}, "
            f"image_shape={img_h_full}×{img_w_full}"
        )

    # --- Step 8: Compute region metadata ---
    h_fullres = ymax_f - ymin_f
    w_fullres = xmax_f - xmin_f
    area_px_fullres = h_fullres * w_fullres

    # Get pixel size from metadata
    pixel_size_x_um, pixel_size_y_um = get_pixel_size_um(str(loader.image_path))
    pixel_area_um2 = pixel_size_x_um * pixel_size_y_um
    area_um2 = area_px_fullres * pixel_area_um2

    # Compute centroid from tight bounding box
    ymin_t_f, xmin_t_f, ymax_t_f, xmax_t_f = scale_bbox_to_full_resolution(
        bbox_tight, scale_y, scale_x
    )
    centroid_y = (ymin_t_f + ymax_t_f) / 2.0
    centroid_x = (xmin_t_f + xmax_t_f) / 2.0

    # Physical origin (top-left corner of padded bbox)
    origin_y_um = ymin_f * pixel_size_y_um
    origin_x_um = xmin_f * pixel_size_x_um

    logger.info(
        f"Region metadata: area={area_px_fullres} px = {area_um2:.1f} um², "
        f"centroid=({centroid_y:.1f}, {centroid_x:.1f}) px, "
        f"origin=({origin_x_um:.1f}, {origin_y_um:.1f}) um"
    )

    # --- Step 9: Create TissueRegion object ---
    region = TissueRegion(
        region_id=0,  # Dummy ID, will be set later
        bbox_ymin=int(ymin_t_f),
        bbox_xmin=int(xmin_t_f),
        bbox_ymax=int(ymax_t_f),
        bbox_xmax=int(xmax_t_f),
        padded_ymin=int(ymin_f),
        padded_xmin=int(xmin_f),
        padded_ymax=int(ymax_f),
        padded_xmax=int(xmax_f),
        area_px=int(area_px_fullres),
        area_um2=float(area_um2),
        centroid_y_px=float(centroid_y),
        centroid_x_px=float(centroid_x),
        origin_um_y=float(origin_y_um),
        origin_um_x=float(origin_x_um),
    )

    # --- Step 10: Save region ---
    output_path = Path(output_dir) / selection_name.replace(" ", "_")
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Get metadata for saving
        metadata = extract_ome_metadata(str(loader.image_path))

        # Wrap reader to intercept channel reads for progress reporting
        active_reader = loader.reader
        if progress_callback is not None:
            active_reader = ProgressReader(
                loader.reader,
                num_channels=loader.num_channels,
                progress_callback=progress_callback,
            )

        save_region(
            region,
            active_reader,
            metadata,
            str(output_path),
            output_formats,
            verbose=verbose,
        )

        logger.info(f"Region extraction complete: {selection_name} → {output_path}")

        # --- Step 11: Write coordinates JSON for back-correction ---
        coords_json = {
            "source_image": str(loader.image_path),
            "source_shape_px": {"height": img_h_full, "width": img_w_full},
            "pixel_size_um": {"x": pixel_size_x_um, "y": pixel_size_y_um},
            "region_name": selection_name,
            "region_id": region.region_id,
            "bbox_tight_px": {
                "ymin": region.bbox_ymin,
                "xmin": region.bbox_xmin,
                "ymax": region.bbox_ymax,
                "xmax": region.bbox_xmax,
            },
            "bbox_padded_px": {
                "ymin": region.padded_ymin,
                "xmin": region.padded_xmin,
                "ymax": region.padded_ymax,
                "xmax": region.padded_xmax,
            },
            "back_correction": {
                "description": (
                    "To map region pixel (ry, rx) back to source image coordinates: "
                    "source_y = ry + padded_ymin, source_x = rx + padded_xmin"
                ),
                "padded_ymin": region.padded_ymin,
                "padded_xmin": region.padded_xmin,
            },
            "area_px": int(area_px_fullres),
            "area_um2": float(area_um2),
            "centroid_px": {"y": float(centroid_y), "x": float(centroid_x)},
            "origin_um": {"y": float(origin_y_um), "x": float(origin_x_um)},
            "extraction_timestamp": datetime.now().isoformat(),
            "num_pyramid_levels": region.num_pyramid_levels,
        }

        # Write to the region subfolder (same level as region_000.h5)
        region_subdir = Path(region.output_hdf5).parent if region.output_hdf5 else output_path
        coords_file = region_subdir / f"region_{region.region_id:03d}_coordinates.json"
        with open(coords_file, "w") as f:
            json.dump(coords_json, f, indent=2)
        logger.info(f"Coordinates saved: {coords_file}")

        # Collect output files
        files = {}
        if "hdf5" in output_formats and region.output_hdf5:
            files['hdf5'] = region.output_hdf5
        if "ometiff" in output_formats and region.output_ometiff:
            files['ometiff'] = region.output_ometiff
        if region.output_thumbnail:
            files['thumbnail'] = region.output_thumbnail
        files['coordinates'] = str(coords_file)

        return {
            'output_folder': str(output_path),
            'files': files,
            'region': region,
            'area_um2': area_um2,
            'area_px': area_px_fullres,
        }

    except Exception as e:
        logger.error(f"Error saving region {selection_name}: {e}", exc_info=True)
        raise
