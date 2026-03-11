"""
Tissue detection on pyramid-downsampled images.

Performs tissue mask generation, connected component labeling, and region
specification extraction on a low-resolution pyramid level, then scales
bounding boxes back to full resolution. This reduces peak memory from
~12 GB (full-res labels) to ~40 MB (downsampled labels).

All functions are private — called by :func:`tissue_extractor.extract_tissue_regions`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage
from scipy.ndimage import label as ndimage_label
from skimage.filters import threshold_otsu
from skimage.morphology import closing, disk, opening

if TYPE_CHECKING:
    from library.imaging.reader import OmeTiffReader

from library.imaging import iter_channel_tiles

logger = logging.getLogger(__name__)

# Extra pixels added to upscaled bounding boxes to absorb rounding errors.
_SCALE_PADDING_PX: int = 10


@dataclass(frozen=True)
class DetectionResult:
    """Immutable result of pyramid-level tissue detection.

    Attributes
    ----------
    level : int
        Pyramid level used for detection (0 = full resolution).
    scale_y : float
        Vertical scale factor (full_res_H / detection_H).
    scale_x : float
        Horizontal scale factor (full_res_W / detection_W).
    mask : np.ndarray
        Binary uint8 mask at the detection level.
    labeled : np.ndarray
        Integer label array at the detection level.
    num_regions : int
        Number of tissue regions found after area filtering.
    """

    level: int
    scale_y: float
    scale_x: float
    mask: np.ndarray
    labeled: np.ndarray
    num_regions: int


def select_detection_level(
    reader: OmeTiffReader,
    min_dim: int = 2000,
) -> tuple[int, float, float]:
    """Select the smallest pyramid level suitable for tissue detection.

    Parameters
    ----------
    reader : OmeTiffReader
        Open lazy reader.
    min_dim : int
        Minimum spatial dimension (both H and W must be >= min_dim).

    Returns
    -------
    level : int
        Pyramid level index.
    scale_y : float
        full_res_H / level_H.
    scale_x : float
        full_res_W / level_W.
    """
    _, full_h, full_w = reader.shape

    if reader.num_levels <= 1:
        logger.warning(
            "No pyramid levels available — falling back to full-resolution "
            "detection (higher memory usage)."
        )
        return 0, 1.0, 1.0

    try:
        level = reader.select_level(min_spatial_dim=min_dim)
    except ValueError:
        logger.warning(
            f"No pyramid level with dims >= {min_dim}; using level 0."
        )
        return 0, 1.0, 1.0

    _, level_h, level_w = reader.level_shape(level)
    scale_y = full_h / level_h
    scale_x = full_w / level_w

    logger.info(
        f"Detection level {level}: {level_h}×{level_w} "
        f"(scale {scale_y:.1f}× / {scale_x:.1f}×)"
    )

    return level, scale_y, scale_x


def generate_tissue_mask_pyramid(
    reader: OmeTiffReader,
    nucleus_channel: int,
    level: int,
    verbose: bool = True,
) -> np.ndarray:
    """Generate binary tissue mask from a downsampled pyramid level.

    Reads the entire nucleus channel at the given pyramid level (typically
    ~4000×2300, ~19 MB for uint16), applies global Otsu thresholding, and
    cleans up with morphological opening + closing.

    Parameters
    ----------
    reader : OmeTiffReader
        Open lazy reader.
    nucleus_channel : int
        Channel index for tissue detection (typically 0 = Hoechst).
    level : int
        Pyramid level to use. If 0, falls back to tile-based processing.
    verbose : bool
        Log progress.

    Returns
    -------
    np.ndarray
        Binary uint8 mask at the detection level resolution.
    """
    if level == 0:
        return _generate_tissue_mask_tiled(reader, nucleus_channel, verbose=verbose)

    _, level_h, level_w = reader.level_shape(level)

    if verbose:
        logger.info(
            f"Generating tissue mask on pyramid level {level} "
            f"({level_h}×{level_w})"
        )

    # Read entire channel at the downsampled level — fits in memory.
    img = reader.get_channel_roi(
        nucleus_channel, y=0, x=0, height=level_h, width=level_w, level=level
    )

    # Global Otsu threshold.
    threshold = threshold_otsu(img)
    mask = (img > threshold).astype(np.uint8)

    if verbose:
        coverage = np.sum(mask) / mask.size * 100
        logger.info(f"Tissue mask: {coverage:.1f}% coverage (threshold={threshold})")

    # Morphological cleanup — scale kernel radius to resolution.
    # At full-res, disk(5) covers ~1.6 µm at 0.325 µm/px.
    # At a downsampled level, scale kernel radius proportionally.
    _, full_h, _ = reader.shape
    downsample_factor = full_h / level_h
    kernel_radius = max(1, int(round(5 / downsample_factor)))
    kernel = disk(kernel_radius)

    mask = opening(mask, footprint=kernel).astype(np.uint8)
    mask = closing(mask, footprint=kernel).astype(np.uint8)

    if verbose:
        coverage = np.sum(mask) / mask.size * 100
        logger.info(f"After morphology: {coverage:.1f}% (kernel radius={kernel_radius})")

    return mask


def _generate_tissue_mask_tiled(
    reader: OmeTiffReader,
    nucleus_channel: int,
    tile_size: int = 4096,
    verbose: bool = True,
) -> np.ndarray:
    """Fallback: tile-based mask generation at full resolution.

    Used when no pyramid levels are available. Processes the nucleus channel
    in non-overlapping tiles to limit memory to one tile at a time.

    Parameters
    ----------
    reader : OmeTiffReader
        Open lazy reader.
    nucleus_channel : int
        Channel index for tissue detection.
    tile_size : int
        Tile side length in pixels.
    verbose : bool
        Log progress.

    Returns
    -------
    np.ndarray
        Binary uint8 mask at full resolution. WARNING: 2.4 GB for 65k×37k.
    """
    if verbose:
        logger.info(
            f"Fallback: tile-based mask at full resolution "
            f"(tile_size={tile_size})"
        )

    _, h, w = reader.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for tile, y_off, x_off, th, tw in iter_channel_tiles(
        reader, nucleus_channel, tile_size=tile_size, level=0
    ):
        threshold = threshold_otsu(tile)
        tile_mask = (tile > threshold).astype(np.uint8)
        y_end = min(y_off + th, h)
        x_end = min(x_off + tw, w)
        mask[y_off:y_end, x_off:x_end] = tile_mask[: y_end - y_off, : x_end - x_off]

    kernel = disk(5)
    mask = opening(mask, footprint=kernel).astype(np.uint8)
    mask = closing(mask, footprint=kernel).astype(np.uint8)

    if verbose:
        coverage = np.sum(mask) / mask.size * 100
        logger.info(f"Tiled mask: {coverage:.1f}% coverage")

    return mask


def compute_connected_components(
    mask: np.ndarray,
    min_area_px: int,
    verbose: bool = True,
) -> tuple[np.ndarray, int]:
    """Label connected components and filter by minimum area.

    Operates on the (small) detection-level mask. At 4100×2315, the labels
    array is ~38 MB int64 instead of ~9.7 GB at full resolution.

    Parameters
    ----------
    mask : np.ndarray
        Binary uint8 mask (detection level resolution).
    min_area_px : int
        Minimum region area in pixels at the detection level.
    verbose : bool
        Log progress.

    Returns
    -------
    labeled : np.ndarray
        Integer label array (0 = background, 1..N = tissue regions).
    num_labels : int
        Number of distinct regions after filtering.
    """
    if verbose:
        logger.info("Computing connected components...")

    labeled, num_features = ndimage_label(mask)

    if verbose:
        logger.info(f"Found {num_features} initial regions")

    # Filter by area.
    areas = np.bincount(labeled.ravel())
    valid_labels = np.where(areas >= min_area_px)[0]
    valid_labels = valid_labels[valid_labels > 0]

    if verbose:
        logger.info(f"Filtering: {len(valid_labels)} regions >= {min_area_px} px")

    # Zero out small regions.
    labeled[~np.isin(labeled, valid_labels)] = 0

    # Re-label sequentially.
    labeled, final_num = ndimage_label(labeled > 0)

    if verbose:
        logger.info(f"Final: {final_num} regions after filtering")

    return labeled, final_num


def extract_region_specs(
    labeled: np.ndarray,
    num_labels: int,
    scale_y: float,
    scale_x: float,
    pixel_size_x_um: float,
    pixel_size_y_um: float,
    padding_px: int,
    img_h: int,
    img_w: int,
    verbose: bool = True,
) -> list[dict]:
    """Extract bounding boxes from the labeled array and upscale to full resolution.

    Parameters
    ----------
    labeled : np.ndarray
        Label array at detection level.
    num_labels : int
        Number of regions.
    scale_y, scale_x : float
        Scale factors from detection level to full resolution.
    pixel_size_x_um, pixel_size_y_um : float
        Physical pixel size at full resolution.
    padding_px : int
        Border padding in full-resolution pixels.
    img_h, img_w : int
        Full-resolution image dimensions.
    verbose : bool
        Log progress.

    Returns
    -------
    list[dict]
        Region specifications sorted spatially (top-to-bottom, left-to-right).
        Each dict contains keys matching TissueRegion fields.
    """
    if verbose:
        logger.info("Extracting region specifications...")

    regions: list[dict] = []
    find_objs = ndimage.find_objects(labeled)

    for label_idx in range(1, num_labels + 1):
        slices = find_objs[label_idx - 1]
        if slices is None:
            continue

        # Bounding box at detection level.
        det_ymin, det_xmin = slices[0].start, slices[1].start
        det_ymax, det_xmax = slices[0].stop, slices[1].stop

        # Area at detection level → upscale to full-res pixel count.
        det_area_px = int(np.sum(labeled == label_idx))
        area_px = int(round(det_area_px * scale_y * scale_x))
        area_um2 = float(area_px * pixel_size_x_um * pixel_size_y_um)

        # Centroid at detection level → upscale.
        y_coords, x_coords = np.where(labeled == label_idx)
        centroid_y_det = float(np.mean(y_coords))
        centroid_x_det = float(np.mean(x_coords))
        centroid_y_px = centroid_y_det * scale_y
        centroid_x_px = centroid_x_det * scale_x

        # Upscale bounding box with rounding-error padding.
        bbox_ymin = max(0, int(det_ymin * scale_y) - _SCALE_PADDING_PX)
        bbox_xmin = max(0, int(det_xmin * scale_x) - _SCALE_PADDING_PX)
        bbox_ymax = min(img_h, int(det_ymax * scale_y) + _SCALE_PADDING_PX)
        bbox_xmax = min(img_w, int(det_xmax * scale_x) + _SCALE_PADDING_PX)

        # Padded bounding box (clamped to image bounds).
        padded_ymin = max(0, bbox_ymin - padding_px)
        padded_xmin = max(0, bbox_xmin - padding_px)
        padded_ymax = min(img_h, bbox_ymax + padding_px)
        padded_xmax = min(img_w, bbox_xmax + padding_px)

        # Physical origin of padded bbox.
        origin_um_y = padded_ymin * pixel_size_y_um
        origin_um_x = padded_xmin * pixel_size_x_um

        regions.append(
            {
                "region_id": -1,
                "bbox_ymin": bbox_ymin,
                "bbox_xmin": bbox_xmin,
                "bbox_ymax": bbox_ymax,
                "bbox_xmax": bbox_xmax,
                "padded_ymin": padded_ymin,
                "padded_xmin": padded_xmin,
                "padded_ymax": padded_ymax,
                "padded_xmax": padded_xmax,
                "area_px": area_px,
                "area_um2": area_um2,
                "centroid_y_px": centroid_y_px,
                "centroid_x_px": centroid_x_px,
                "origin_um_y": origin_um_y,
                "origin_um_x": origin_um_x,
            }
        )

    # Sort spatially: top-to-bottom, then left-to-right.
    regions.sort(key=lambda r: (r["centroid_y_px"], r["centroid_x_px"]))

    # Assign sequential IDs.
    for i, region in enumerate(regions):
        region["region_id"] = i

    if verbose:
        logger.info(f"Extracted {len(regions)} region specifications")

    return regions
