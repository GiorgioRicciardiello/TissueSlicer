"""
Coordinate system mapping between downsampled (display) and full resolution.

Provides utilities for transforming coordinates while maintaining verification.
"""

import logging
from typing import Tuple, List, Union

logger = logging.getLogger(__name__)


class CoordinateMapper:
    """
    Map coordinates between downsampled (display level) and full resolution.

    All transformations include verification and logging.
    """

    def __init__(self, scale_y: float, scale_x: float):
        """
        Initialize coordinate mapper with scale factors.

        Args:
            scale_y: Full_height / display_height
            scale_x: Full_width / display_width
        """
        self.scale_y = scale_y
        self.scale_x = scale_x
        logger.debug(f"CoordinateMapper initialized: scale=({scale_y:.2f}, {scale_x:.2f})")

    def downsampled_to_fullres_bbox(
        self,
        bbox_downsampled: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Scale a bounding box from downsampled to full resolution.

        Args:
            bbox_downsampled: (ymin, xmin, ymax, xmax) at downsampled level

        Returns:
            (ymin, xmin, ymax, xmax) at full resolution
        """
        ymin_d, xmin_d, ymax_d, xmax_d = bbox_downsampled

        ymin_f = int(round(ymin_d * self.scale_y))
        xmin_f = int(round(xmin_d * self.scale_x))
        ymax_f = int(round(ymax_d * self.scale_y))
        xmax_f = int(round(xmax_d * self.scale_x))

        # Verify scale consistency
        h_d = ymax_d - ymin_d
        w_d = xmax_d - xmin_d
        h_f = ymax_f - ymin_f
        w_f = xmax_f - xmin_f

        actual_scale_y = h_f / h_d if h_d > 0 else 0
        actual_scale_x = w_f / w_d if w_d > 0 else 0

        scale_error_y = abs(actual_scale_y - self.scale_y)
        scale_error_x = abs(actual_scale_x - self.scale_x)

        logger.debug(
            f"BBox scaling: downsampled=({ymin_d}, {xmin_d}, {ymax_d}, {xmax_d}) "
            f"→ fullres=({ymin_f}, {xmin_f}, {ymax_f}, {xmax_f}), "
            f"scale_error=(y:{scale_error_y:.3f}, x:{scale_error_x:.3f})"
        )

        if scale_error_y > 0.1 or scale_error_x > 0.1:
            logger.warning(
                f"Large scale error detected: y_error={scale_error_y:.3f}, "
                f"x_error={scale_error_x:.3f}"
            )

        return (ymin_f, xmin_f, ymax_f, xmax_f)

    def downsampled_to_fullres_points(
        self,
        points_downsampled: List[Tuple[float, float]]
    ) -> List[Tuple[int, int]]:
        """
        Scale a list of (y, x) points from downsampled to full resolution.

        Args:
            points_downsampled: List of (y, x) tuples at downsampled level

        Returns:
            List of (y, x) tuples at full resolution
        """
        points_fullres = []
        for y_d, x_d in points_downsampled:
            y_f = int(round(y_d * self.scale_y))
            x_f = int(round(x_d * self.scale_x))
            points_fullres.append((y_f, x_f))

        logger.debug(
            f"Points scaling: {len(points_downsampled)} points "
            f"from downsampled → fullres"
        )

        return points_fullres

    def fullres_to_downsampled_bbox(
        self,
        bbox_fullres: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Scale a bounding box from full resolution to downsampled.

        Args:
            bbox_fullres: (ymin, xmin, ymax, xmax) at full resolution

        Returns:
            (ymin, xmin, ymax, xmax) at downsampled level
        """
        ymin_f, xmin_f, ymax_f, xmax_f = bbox_fullres

        ymin_d = int(round(ymin_f / self.scale_y))
        xmin_d = int(round(xmin_f / self.scale_x))
        ymax_d = int(round(ymax_f / self.scale_y))
        xmax_d = int(round(xmax_f / self.scale_x))

        logger.debug(
            f"BBox scaling (reverse): fullres=({ymin_f}, {xmin_f}, {ymax_f}, {xmax_f}) "
            f"→ downsampled=({ymin_d}, {xmin_d}, {ymax_d}, {xmax_d})"
        )

        return (ymin_d, xmin_d, ymax_d, xmax_d)

    def fullres_to_downsampled_points(
        self,
        points_fullres: List[Tuple[float, float]]
    ) -> List[Tuple[int, int]]:
        """
        Scale a list of (y, x) points from full resolution to downsampled.

        Args:
            points_fullres: List of (y, x) tuples at full resolution

        Returns:
            List of (y, x) tuples at downsampled level
        """
        points_downsampled = []
        for y_f, x_f in points_fullres:
            y_d = int(round(y_f / self.scale_y))
            x_d = int(round(x_f / self.scale_x))
            points_downsampled.append((y_d, x_d))

        logger.debug(
            f"Points scaling (reverse): {len(points_fullres)} points "
            f"from fullres → downsampled"
        )

        return points_downsampled

    def verify_scale_factors(
        self,
        downsampled_h: int,
        downsampled_w: int,
        fullres_h: int,
        fullres_w: int
    ) -> bool:
        """
        Verify that scale factors are consistent with image dimensions.

        Args:
            downsampled_h, downsampled_w: Display level dimensions
            fullres_h, fullres_w: Full resolution dimensions

        Returns:
            True if scale factors are valid, False otherwise
        """
        if downsampled_h <= 0 or downsampled_w <= 0:
            logger.error(f"Invalid downsampled dims: {downsampled_h}×{downsampled_w}")
            return False

        expected_scale_y = fullres_h / downsampled_h
        expected_scale_x = fullres_w / downsampled_w

        error_y = abs(expected_scale_y - self.scale_y)
        error_x = abs(expected_scale_x - self.scale_x)

        is_valid = error_y < 0.1 and error_x < 0.1

        logger.info(
            f"Scale verification: "
            f"expected=({expected_scale_y:.2f}, {expected_scale_x:.2f}), "
            f"actual=({self.scale_y:.2f}, {self.scale_x:.2f}), "
            f"error=({error_y:.3f}, {error_x:.3f}), valid={is_valid}"
        )

        return is_valid
