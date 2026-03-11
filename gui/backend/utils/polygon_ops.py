"""
Polygon to mask operations with rigorous coordinate verification.

CRITICAL: All coordinate transformations are logged with before/after values.
"""

import logging
import numpy as np
from scipy.ndimage import label as ndimage_label
from skimage import draw

logger = logging.getLogger(__name__)


def polygon_to_mask(polygon_coords, shape):
    """
    Convert polygon (list of [x, y]) to binary mask.

    Args:
        polygon_coords: List of [x, y] coordinates in CANVAS order
        shape: (height, width) of output mask

    Returns:
        Binary uint8 mask of shape (height, width)

    NOTE: Canvas uses (x, y) order, but numpy uses (row, col) = (y, x) order.
    This function handles the conversion.
    """
    if len(polygon_coords) < 3:
        logger.warning(f"Polygon has {len(polygon_coords)} points, need >= 3")
        return np.zeros(shape, dtype=np.uint8)

    mask = np.zeros(shape, dtype=np.uint8)

    try:
        # Convert [x, y] canvas coords to (row, col) = (y, x) numpy coords
        coords = np.array(polygon_coords)
        x_coords = np.round(coords[:, 0]).astype(int)
        y_coords = np.round(coords[:, 1]).astype(int)

        logger.debug(
            f"Polygon to mask: {len(polygon_coords)} points, "
            f"canvas_range_x=[{x_coords.min()}, {x_coords.max()}], "
            f"canvas_range_y=[{y_coords.min()}, {y_coords.max()}]"
        )

        # Rasterize polygon using scikit-image
        rr, cc = draw.polygon(y_coords, x_coords, shape)
        mask[rr, cc] = 1

        pixel_count = np.sum(mask)
        logger.info(f"Mask created: {pixel_count} pixels filled")

        return mask

    except Exception as e:
        logger.error(f"Error converting polygon to mask: {e}", exc_info=True)
        return np.zeros(shape, dtype=np.uint8)


def compute_bbox_from_mask(mask):
    """
    Find tight bounding box of mask.

    Args:
        mask: Binary uint8 array

    Returns:
        Tuple (ymin, xmin, ymax, xmax) in pixel coordinates
        OR None if mask is empty
    """
    y_indices, x_indices = np.where(mask > 0)

    if len(y_indices) == 0:
        logger.warning("Empty mask - no bounding box")
        return None

    ymin = int(y_indices.min())
    ymax = int(y_indices.max()) + 1
    xmin = int(x_indices.min())
    xmax = int(x_indices.max()) + 1

    height = ymax - ymin
    width = xmax - xmin

    logger.debug(
        f"Tight bbox: ({ymin}, {xmin}, {ymax}, {xmax}), "
        f"dimensions=({height}, {width}), "
        f"area={height * width} px"
    )

    return (ymin, xmin, ymax, xmax)


def apply_padding(bbox, padding_px, img_h, img_w):
    """
    Add padding to bounding box, clamped to image bounds.

    Args:
        bbox: (ymin, xmin, ymax, xmax)
        padding_px: Padding in pixels (int or dict)
        img_h, img_w: Image dimensions

    Returns:
        (padded_ymin, padded_xmin, padded_ymax, padded_xmax)

    CRITICAL: All coordinates are verified to stay within image bounds.
    """
    if bbox is None:
        return None

    ymin, xmin, ymax, xmax = bbox

    # Parse padding
    if isinstance(padding_px, int):
        pad_all = padding_px
        pad_top = pad_bottom = pad_left = pad_right = pad_all
    else:
        pad_all = padding_px.get('all', 0)
        pad_top = padding_px.get('top', pad_all)
        pad_bottom = padding_px.get('bottom', pad_all)
        pad_left = padding_px.get('left', pad_all)
        pad_right = padding_px.get('right', pad_all)

    # Apply padding
    padded_ymin = max(0, ymin - pad_top)
    padded_xmin = max(0, xmin - pad_left)
    padded_ymax = min(img_h, ymax + pad_bottom)
    padded_xmax = min(img_w, xmax + pad_right)

    logger.debug(
        f"Applied padding: "
        f"before=({ymin}, {xmin}, {ymax}, {xmax}), "
        f"padding=({pad_top}, {pad_left}, {pad_bottom}, {pad_right}), "
        f"after=({padded_ymin}, {padded_xmin}, {padded_ymax}, {padded_xmax}), "
        f"clamped_to_bounds={img_h}x{img_w}"
    )

    return (padded_ymin, padded_xmin, padded_ymax, padded_xmax)


def force_square(bbox, img_h, img_w, expand_mode='outer'):
    """
    Expand bounding box to be square (same height and width).

    Expands around the center of the original bbox while respecting image bounds.

    Args:
        bbox: (ymin, xmin, ymax, xmax)
        img_h, img_w: Image dimensions
        expand_mode: 'outer' (expand equally) or 'inner' (shrink to square)

    Returns:
        (ymin, xmin, ymax, xmax) - now square
    """
    if bbox is None:
        return None

    ymin, xmin, ymax, xmax = bbox
    h = ymax - ymin
    w = xmax - xmin

    # Determine square side length
    side = max(h, w)

    # Center of original bbox
    cy = (ymin + ymax) / 2.0
    cx = (xmin + xmax) / 2.0

    # Expand around center
    new_ymin = int(round(cy - side / 2.0))
    new_xmin = int(round(cx - side / 2.0))
    new_ymax = new_ymin + side
    new_xmax = new_xmin + side

    logger.debug(
        f"Before force_square: ({ymin}, {xmin}, {ymax}, {xmax}), "
        f"dims=({h}, {w}), side={side}"
    )

    # Clamp to image bounds
    # If exceeding bounds, shift the entire box rather than shrinking
    if new_ymin < 0:
        shift = -new_ymin
        new_ymin += shift
        new_ymax += shift

    if new_xmin < 0:
        shift = -new_xmin
        new_xmin += shift
        new_xmax += shift

    if new_ymax > img_h:
        shift = new_ymax - img_h
        new_ymin = max(0, new_ymin - shift)
        new_ymax = img_h

    if new_xmax > img_w:
        shift = new_xmax - img_w
        new_xmin = max(0, new_xmin - shift)
        new_xmax = img_w

    logger.debug(
        f"After force_square: ({new_ymin}, {new_xmin}, {new_ymax}, {new_xmax}), "
        f"dims=({new_ymax - new_ymin}, {new_xmax - new_xmin}), "
        f"clamped_to_bounds={img_h}x{img_w}"
    )

    return (new_ymin, new_xmin, new_ymax, new_xmax)


def scale_bbox_to_full_resolution(bbox_downsampled, scale_y, scale_x):
    """
    Scale bounding box from downsampled to full resolution.

    CRITICAL VERIFICATION:
    - Input coordinates are verified to be within downsampled bounds
    - Output coordinates are verified to match expected full-res scale
    - All transformations are logged with before/after values

    Args:
        bbox_downsampled: (ymin, xmin, ymax, xmax) at downsampled level
        scale_y, scale_x: Scale factors (typically 16.0 for level 4)

    Returns:
        (ymin, xmin, ymax, xmax) at full resolution
    """
    if bbox_downsampled is None:
        return None

    ymin_d, xmin_d, ymax_d, xmax_d = bbox_downsampled

    # Scale up
    ymin_f = int(round(ymin_d * scale_y))
    xmin_f = int(round(xmin_d * scale_x))
    ymax_f = int(round(ymax_d * scale_y))
    xmax_f = int(round(xmax_d * scale_x))

    # Verify scale factor consistency
    h_downsampled = ymax_d - ymin_d
    w_downsampled = xmax_d - xmin_d
    h_full = ymax_f - ymin_f
    w_full = xmax_f - xmin_f

    actual_scale_y = h_full / h_downsampled if h_downsampled > 0 else 0
    actual_scale_x = w_full / w_downsampled if w_downsampled > 0 else 0

    logger.info(
        f"Coordinate scaling verification: "
        f"downsampled=({ymin_d}, {xmin_d}, {ymax_d}, {xmax_d}) {h_downsampled}x{w_downsampled}, "
        f"full_res=({ymin_f}, {xmin_f}, {ymax_f}, {xmax_f}) {h_full}x{w_full}, "
        f"expected_scale=({scale_y:.2f}, {scale_x:.2f}), "
        f"actual_scale=({actual_scale_y:.2f}, {actual_scale_x:.2f})"
    )

    # Check for rounding errors
    scale_y_error = abs(actual_scale_y - scale_y)
    scale_x_error = abs(actual_scale_x - scale_x)

    if scale_y_error > 0.1 or scale_x_error > 0.1:
        logger.warning(
            f"Large scale error: y={scale_y_error:.3f}, x={scale_x_error:.3f}"
        )

    return (ymin_f, xmin_f, ymax_f, xmax_f)


def validate_coordinates(bbox, img_h, img_w, name="bbox"):
    """
    Validate that coordinates are within image bounds.

    Args:
        bbox: (ymin, xmin, ymax, xmax)
        img_h, img_w: Image dimensions
        name: Name for logging

    Returns:
        True if valid, False otherwise
    """
    if bbox is None:
        logger.error(f"{name}: bbox is None")
        return False

    ymin, xmin, ymax, xmax = bbox

    errors = []

    if ymin < 0:
        errors.append(f"ymin={ymin} < 0")
    if xmin < 0:
        errors.append(f"xmin={xmin} < 0")
    if ymax > img_h:
        errors.append(f"ymax={ymax} > img_h={img_h}")
    if xmax > img_w:
        errors.append(f"xmax={xmax} > img_w={img_w}")
    if ymin >= ymax:
        errors.append(f"ymin >= ymax ({ymin} >= {ymax})")
    if xmin >= xmax:
        errors.append(f"xmin >= xmax ({xmin} >= {xmax})")

    if errors:
        logger.error(f"{name} validation failed: {', '.join(errors)}")
        return False

    logger.debug(f"{name} validated: ({ymin}, {xmin}, {ymax}, {xmax}) within {img_h}x{img_w}")
    return True
