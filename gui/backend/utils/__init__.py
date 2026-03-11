"""Backend utility modules for tissue selection GUI."""

from .image_loader import OmeTiffImageLoader
from .polygon_ops import (
    polygon_to_mask,
    compute_bbox_from_mask,
    apply_padding,
    force_square,
    scale_bbox_to_full_resolution,
    validate_coordinates,
)
from .coordinate_mapper import CoordinateMapper
from .session_manager import SessionManager
from .extraction import extract_region_with_padding

__all__ = [
    "OmeTiffImageLoader",
    "polygon_to_mask",
    "compute_bbox_from_mask",
    "apply_padding",
    "force_square",
    "scale_bbox_to_full_resolution",
    "validate_coordinates",
    "CoordinateMapper",
    "SessionManager",
    "extract_region_with_padding",
]
