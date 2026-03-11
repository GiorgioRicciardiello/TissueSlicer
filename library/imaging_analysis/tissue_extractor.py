"""
Tissue Region Extraction from Large OME-TIFF Slides

Separates multiple tissue samples from a silicon assay slide into individual
HDF5 + OME-TIFF files with full provenance metadata for downstream quantification.

Optimized for large images (30-100+ GB): tissue detection runs on a
pyramid-downsampled level (~40 MB RAM) instead of full resolution (~12 GB),
and region saving streams one channel at a time (~200 MB peak).

Usage
-----
>>> from library.imaging_analysis import extract_tissue_regions
>>> regions = extract_tissue_regions(
...     "large_slide.ome.tiff",
...     output_dir="./extracted_tissues/",
...     n_workers=7,
... )
>>> for region in regions:
...     if region.success:
...         print(f"Region {region.region_id}: {region.area_um2:.0f} um2")
"""

from __future__ import annotations

import gc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from library.imaging import (
    OmeTiffReader,
    extract_ome_metadata,
    get_pixel_size_um,
)
from library.imaging_analysis._detection import (
    compute_connected_components,
    extract_region_specs,
    generate_tissue_mask_pyramid,
    select_detection_level,
)
from library.imaging_analysis._manifest import write_manifest
from library.imaging_analysis._writers import save_region

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structure
# ============================================================================


@dataclass
class TissueRegion:
    """Descriptor for an extracted tissue region.

    Stores spatial coordinates (pixel and physical), area metrics, and output
    file paths. All pixel coordinates are in the full-image space of the input
    OME-TIFF.

    Attributes
    ----------
    region_id : int
        Zero-indexed identifier, assigned after spatial sorting.
    bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax : int
        Tight bounding box of the tissue region (no padding).
    padded_ymin, padded_xmin, padded_ymax, padded_xmax : int
        Extraction bounding box (tight bbox + padding, clamped to image bounds).
    area_px : int
        Number of pixels in the tissue region.
    area_um2 : float
        Physical area in micrometers squared.
    centroid_y_px, centroid_x_px : float
        Center of mass of the region in pixel coordinates.
    origin_um_y, origin_um_x : float
        Physical coordinate (um) of the top-left corner of the padded bbox.
    output_hdf5 : str
        Absolute path to the saved HDF5 file.
    output_ometiff : str
        Absolute path to the saved OME-TIFF file.
    output_thumbnail : str
        Absolute path to the QC thumbnail image.
    num_pyramid_levels : int
        Number of pyramid levels preserved in output (= source pyramid levels).
    success : bool
        True if extraction and saving were successful.
    error_message : str
        Empty string on success; error description on failure.
    """

    region_id: int
    bbox_ymin: int
    bbox_xmin: int
    bbox_ymax: int
    bbox_xmax: int
    padded_ymin: int
    padded_xmin: int
    padded_ymax: int
    padded_xmax: int
    area_px: int
    area_um2: float
    centroid_y_px: float
    centroid_x_px: float
    origin_um_y: float
    origin_um_x: float
    output_hdf5: str = ""
    output_ometiff: str = ""
    output_thumbnail: str = ""
    num_pyramid_levels: int = 0
    success: bool = False
    error_message: str = ""


# ============================================================================
# Input Validation
# ============================================================================


def _validate_inputs(
    nucleus_channel: int,
    min_tissue_area_um2: float,
    padding_px: int,
    n_workers: int,
    output_formats: list[str],
    detection_min_dim: int,
) -> None:
    """Validate extraction parameters at the system boundary.

    Raises
    ------
    ValueError
        If any parameter is out of valid range.
    """
    if min_tissue_area_um2 <= 0:
        raise ValueError(f"min_tissue_area_um2 must be > 0, got {min_tissue_area_um2}")
    if padding_px < 0:
        raise ValueError(f"padding_px must be >= 0, got {padding_px}")
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if detection_min_dim < 100:
        raise ValueError(f"detection_min_dim must be >= 100, got {detection_min_dim}")

    valid_formats = {"hdf5", "ometiff"}
    invalid = set(output_formats) - valid_formats
    if invalid:
        raise ValueError(f"Invalid output formats: {invalid}. Valid: {valid_formats}")


# ============================================================================
# Public API
# ============================================================================


def extract_tissue_regions(
    image_path: str,
    output_dir: str,
    nucleus_channel: int = 0,
    min_tissue_area_um2: float = 5000.0,
    padding_px: int = 50,
    tile_size: int = 4096,
    n_workers: int = 7,
    output_formats: list[str] | None = None,
    detection_min_dim: int = 2000,
    seed: int = 42,
    verbose: bool = True,
) -> list[TissueRegion]:
    """Extract individual tissue regions from a large multi-tissue OME-TIFF slide.

    Detects separate tissue samples using connected component analysis on
    a pyramid-downsampled nucleus channel, then extracts each tissue as
    separate HDF5 and/or OME-TIFF files with full provenance metadata.

    Parameters
    ----------
    image_path : str
        Path to the input OME-TIFF file (30-100+ GB supported).
    output_dir : str
        Directory to save extracted region files. Created if it doesn't exist.
    nucleus_channel : int, optional
        Channel index for tissue detection (default: 0, typically Hoechst/DAPI).
    min_tissue_area_um2 : float, optional
        Minimum tissue region size in um2 (default: 5000.0). Regions smaller
        than this are discarded as artifacts.
    padding_px : int, optional
        Border size (pixels) to add around each tissue region (default: 50).
    tile_size : int, optional
        Tile size for fallback full-resolution processing (default: 4096).
        Only used when no pyramid levels are available.
    n_workers : int, optional
        Number of parallel workers for saving regions (default: 7).
    output_formats : list[str] | None, optional
        Formats to save: ``["hdf5", "ometiff"]`` or subsets. Default: both.
    detection_min_dim : int, optional
        Minimum spatial dimension for pyramid level selection (default: 2000).
        The smallest pyramid level where both H and W >= this value is used
        for tissue detection.
    seed : int, optional
        Random seed for reproducibility (default: 42).
    verbose : bool, optional
        Enable logging.

    Returns
    -------
    list[TissueRegion]
        Extracted tissue regions with spatial metadata and output file paths.

    Output Structure
    ----------------
    ::

        output_dir/
            manifest.json              # slide-level index
            region_000/
                region_000.h5          # all channels, HDF5
                region_000.ome.tiff    # all channels, OME-TIFF
                region_000_thumbnail.png
            region_001/
                ...
    """
    if output_formats is None:
        output_formats = ["hdf5", "ometiff"]

    _validate_inputs(
        nucleus_channel, min_tissue_area_um2, padding_px,
        n_workers, output_formats, detection_min_dim,
    )

    image_path_obj = Path(image_path)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    if verbose:
        logger.info("=" * 70)
        logger.info(f"Tissue Extraction: {image_path_obj.name}")
        logger.info(f"Output: {output_dir_obj}")
        logger.info(
            f"Parameters: min_area={min_tissue_area_um2} um2, "
            f"padding={padding_px} px, detection_min_dim={detection_min_dim}"
        )
        logger.info("=" * 70)

    with OmeTiffReader(str(image_path)) as reader:
        metadata = extract_ome_metadata(str(image_path))
        pixel_size_x_um, pixel_size_y_um = get_pixel_size_um(str(image_path))
        _, img_h, img_w = reader.shape

        if verbose:
            logger.info(f"Image shape: {reader.shape} ({reader.dtype})")
            logger.info(f"Pixel size: {pixel_size_x_um:.4f} x {pixel_size_y_um:.4f} um")
            logger.info(f"Pyramid levels: {reader.num_levels}")

        # --- Step 1: Select detection level ---
        det_level, scale_y, scale_x = select_detection_level(
            reader, min_dim=detection_min_dim
        )

        # --- Step 2: Generate tissue mask at detection level ---
        mask = generate_tissue_mask_pyramid(
            reader, nucleus_channel, level=det_level, verbose=verbose
        )

        # --- Step 3: Connected components on small mask ---
        # Convert min area from full-res um2 to detection-level pixels.
        pixel_area_um2 = pixel_size_x_um * pixel_size_y_um
        min_area_full_px = int(np.ceil(min_tissue_area_um2 / pixel_area_um2))
        min_area_det_px = max(1, int(round(min_area_full_px / (scale_y * scale_x))))

        labeled, n_regions = compute_connected_components(
            mask, min_area_det_px, verbose=verbose
        )

        # --- Step 4: Extract region specs (upscale to full res) ---
        region_dicts = extract_region_specs(
            labeled, n_regions,
            scale_y, scale_x,
            pixel_size_x_um, pixel_size_y_um,
            padding_px, img_h, img_w,
            verbose=verbose,
        )

        # Convert dicts to TissueRegion dataclass instances.
        regions = [TissueRegion(**rd) for rd in region_dicts]

        # Release detection arrays.
        del mask, labeled
        gc.collect()

        if verbose:
            logger.info("Detection arrays freed")

        # --- Step 5: Save regions ---
        if len(regions) == 0:
            if verbose:
                logger.warning("No tissue regions found!")
            # Write manifest even with zero regions.
            write_manifest(
                regions, str(image_path_obj), reader.shape, metadata,
                None, det_level, (scale_y, scale_x), seed, str(output_dir_obj),
            )
            return regions

        if n_workers <= 1:
            for region in regions:
                save_region(
                    region, reader, metadata,
                    str(output_dir_obj), output_formats, verbose,
                )
        else:
            futures = {}
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                for region in regions:
                    future = executor.submit(
                        save_region,
                        region, reader, metadata,
                        str(output_dir_obj), output_formats, verbose,
                    )
                    futures[future] = region

                for future in as_completed(futures):
                    region = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        region.success = False
                        region.error_message = str(e)
                        logger.error(f"Region {region.region_id} failed: {e}")

    # --- Step 6: Write manifest ---
    write_manifest(
        regions, str(image_path_obj), (reader.num_channels, img_h, img_w),
        metadata, None, det_level, (scale_y, scale_x), seed, str(output_dir_obj),
    )

    # Summary.
    successful = [r for r in regions if r.success]
    if verbose:
        logger.info("=" * 70)
        logger.info(
            f"Extraction complete: {len(successful)}/{len(regions)} regions saved"
        )
        for r in successful:
            logger.info(
                f"  Region {r.region_id:02d}: {r.area_um2 / 1e6:.2f} mm2 "
                f"@ ({r.origin_um_x:.0f}, {r.origin_um_y:.0f}) um"
            )
        logger.info("=" * 70)

    return regions
