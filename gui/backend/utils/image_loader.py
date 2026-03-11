"""
Image loader for OME-TIFF with pyramid level support.
Coordinates are always logged for verification.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from library.imaging import OmeTiffReader, extract_ome_metadata, get_pixel_size_um

logger = logging.getLogger(__name__)


def compute_channel_calibration(channel_data: np.ndarray) -> dict:
    """
    Compute contrast/brightness calibration for a single channel using percentile stretching.

    Uses 2nd and 98th percentiles (industry standard in microscopy) to ignore outliers
    while preserving dynamic range.

    Args:
        channel_data: uint16 numpy array (any shape)

    Returns:
        dict with keys:
        - 'p2': 2nd percentile (suggested min display value)
        - 'p98': 98th percentile (suggested max display value)
        - 'min': actual minimum value in data
        - 'max': actual maximum value in data
        - 'mean': mean value (for reference)
    """
    if channel_data.size == 0:
        # Return neutral calibration for empty channels
        return {'p2': 0, 'p98': 255, 'min': 0, 'max': 255, 'mean': 128}

    p2, p98 = np.percentile(channel_data, (2, 98))
    min_val = float(np.min(channel_data))
    max_val = float(np.max(channel_data))
    mean_val = float(np.mean(channel_data))

    calibration = {
        'p2': float(p2),
        'p98': float(p98),
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
    }

    logger.debug(
        f"Channel calibration: p2={p2:.1f}, p98={p98:.1f}, "
        f"min={min_val:.1f}, max={max_val:.1f}, mean={mean_val:.1f}"
    )

    return calibration


class OmeTiffImageLoader:
    """
    Lazy loader for large OME-TIFF files with pyramid support.

    Loads only the requested pyramid level to minimize memory usage.
    """

    def __init__(self, image_path):
        """
        Initialize loader.

        Args:
            image_path: Path to OME-TIFF file
        """
        self.image_path = Path(image_path)
        self.reader = OmeTiffReader(str(self.image_path))

        # Metadata
        self.metadata = extract_ome_metadata(str(self.image_path))
        self.pixel_size_x_um, self.pixel_size_y_um = get_pixel_size_um(str(self.image_path))

        # Full resolution shape
        self.num_channels, self.height_full, self.width_full = self.reader.shape

        logger.info(
            f"Loaded {self.image_path.name}: "
            f"shape={self.reader.shape}, "
            f"pixel_size=({self.pixel_size_x_um:.4f}, {self.pixel_size_y_um:.4f}) um, "
            f"pyramid_levels={self.reader.num_levels}"
        )

        # Detect display level (pyramid level 4 or closest)
        self.display_level = self._select_display_level()

        logger.info(f"Display level: {self.display_level}")

        # Cache for channel calibrations (computed on-demand)
        self._calibration_cache = {}
        # Cache for display-level uint16 arrays (avoids repeated disk reads in compositing)
        self._display_cache: dict[int, np.ndarray] = {}

    def _select_display_level(self, target_dim=2000):
        """
        Select the smallest pyramid level where both dimensions >= target_dim.

        For FNEL03 88GB image:
        - Full res: 65603 x 37041
        - Level 4: ~4100 x 2315 (16x downsampled) ← ideal for web display
        """
        try:
            level = self.reader.select_level(min_spatial_dim=target_dim)
            logger.debug(f"Selected pyramid level {level} for target_dim={target_dim}")
            return level
        except ValueError:
            logger.warning(f"No pyramid level with dim >= {target_dim}, using level 0")
            return 0

    def get_channel_downsampled(self, channel_idx):
        """
        Read channel at display level (downsampled).

        Returns:
            img: uint16 array at display level resolution
            shape_full: Full resolution shape (C, H, W)
            scale_y: Full_H / display_H
            scale_x: Full_W / display_W
        """
        if channel_idx < 0 or channel_idx >= self.num_channels:
            raise IndexError(f"Channel {channel_idx} out of range [0, {self.num_channels-1}]")

        # Get display level shape
        _, h_display, w_display = self.reader.level_shape(self.display_level)

        # Scale factors
        scale_y = self.height_full / h_display
        scale_x = self.width_full / w_display

        logger.debug(
            f"Getting channel {channel_idx} at level {self.display_level}: "
            f"display_shape=({h_display}, {w_display}), "
            f"scale=({scale_y:.2f}, {scale_x:.2f})"
        )

        # Read channel at display level
        img = self.reader.get_channel_roi(
            channel_idx,
            y=0, x=0,
            height=h_display,
            width=w_display,
            level=self.display_level
        )

        return img, (self.num_channels, self.height_full, self.width_full), scale_y, scale_x

    def get_composite_display(self, channel_configs: list) -> np.ndarray:
        """
        Generate an additive composite RGB image from multiple channels.

        For each enabled config: read uint16 display-level array (cached),
        apply linear stretch [display_min, display_max] → [0, 1], multiply by
        RGB tint color, and additively accumulate into a float32 BGR buffer.
        Returns uint8 BGR ndarray (H, W, 3) ready for cv2.imencode.

        Args:
            channel_configs: list of dicts with keys:
                - 'index' (int): channel index 0–19
                - 'enabled' (bool): skip if False
                - 'color' (list[int, int, int]): RGB tint color, 0–255 each
                - 'display_min' (float): uint16 black point, clipped to [0, 65535]
                - 'display_max' (float): uint16 white point, clipped to [0, 65535]

        Returns:
            uint8 ndarray of shape (H, W, 3) in BGR channel order (OpenCV convention),
            ready for cv2.imencode('.png', ...).
        """
        # Get display level shape
        _, h_display, w_display = self.reader.level_shape(self.display_level)
        composite = np.zeros((h_display, w_display, 3), dtype=np.float32)  # BGR

        for cfg in channel_configs:
            if not cfg.get('enabled', False):
                continue

            idx = int(cfg['index'])
            color = cfg.get('color', [255, 255, 255])
            lo = float(max(0.0, cfg.get('display_min', 0)))
            hi = float(min(65535.0, cfg.get('display_max', 4095)))
            hi = max(lo + 1.0, hi)  # Guard against division by zero

            # Read with cache (avoids repeated disk I/O after first composite)
            if idx not in self._display_cache:
                raw, _, _, _ = self.get_channel_downsampled(idx)
                self._display_cache[idx] = raw
            raw = self._display_cache[idx]

            # Stretch uint16 → float32 [0, 1]
            stretched = (raw.astype(np.float32) - lo) / (hi - lo)
            np.clip(stretched, 0.0, 1.0, out=stretched)

            # Apply RGB tint (additive blend into BGR accumulator)
            r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
            composite[:, :, 0] += stretched * b  # OpenCV BGR order
            composite[:, :, 1] += stretched * g
            composite[:, :, 2] += stretched * r

        np.clip(composite, 0.0, 1.0, out=composite)
        logger.debug(
            f"Composite generated: shape={composite.shape}, "
            f"channels={len([c for c in channel_configs if c.get('enabled')])}"
        )
        return (composite * 255.0).astype(np.uint8)

    def get_level_shape(self, level=None):
        """
        Get shape of a pyramid level.

        Returns:
            (num_channels, height, width) at specified level
        """
        if level is None:
            level = self.display_level

        return self.reader.level_shape(level)

    def get_channel_roi_full_res(self, channel_idx, y, x, height, width):
        """
        Read region of interest from full resolution.

        Args:
            channel_idx: Channel to read
            y, x: Top-left corner
            height, width: ROI extent

        Returns:
            uint16 array of shape (height, width)
        """
        logger.debug(
            f"Reading channel {channel_idx} at full resolution: "
            f"roi=({y}, {x}, {height}, {width})"
        )

        return self.reader.get_channel_roi(
            channel_idx,
            y=y, x=x,
            height=height, width=width,
            level=0  # Full resolution
        )

    def get_all_channels_roi_full_res(self, y, x, height, width):
        """
        Read all channels for a region at full resolution.

        Returns:
            dict: {channel_idx: array} for all 20 channels
        """
        logger.info(
            f"Reading all {self.num_channels} channels at full resolution: "
            f"roi=({y}, {x}, {height}, {width})"
        )

        channels = {}
        for ch_idx in range(self.num_channels):
            channels[ch_idx] = self.get_channel_roi_full_res(ch_idx, y, x, height, width)

        return channels

    def get_all_channels_roi_full_res_parallel(
        self,
        y: int,
        x: int,
        height: int,
        width: int,
        n_workers: int = 4,
        progress_callback: callable = None,
    ) -> dict:
        """
        Read all channels for a region at full resolution using a thread pool.

        Each channel is an independent random-access read; parallelising across
        channels exploits I/O concurrency on NVMe/SSD storage.

        Args:
            y, x: Top-left corner of the ROI in full-resolution pixels.
            height, width: Extent of the ROI.
            n_workers: Number of parallel reader threads (default 4).
                Tune down to 2 for spinning HDD; up to 8 for NVMe RAID.
            progress_callback: Optional callable(channel_idx: int) invoked each
                time a channel finishes reading.  Used to stream progress to the
                job-state dict.

        Returns:
            dict mapping channel_idx (int) → uint16 ndarray of shape (height, width),
            ordered by channel index.
        """
        logger.info(
            f"Parallel read of {self.num_channels} channels "
            f"roi=({y}, {x}, {height}, {width}) n_workers={n_workers}"
        )

        # Each thread needs its own reader to avoid seek-position races.
        # OmeTiffReader is cheap to open (metadata is small).
        lock = threading.Lock()  # protects progress_callback if needed

        def _read_channel(ch_idx: int) -> tuple:
            # Open a private reader for this thread
            try:
                from library.imaging import OmeTiffReader
            except ImportError:
                raise RuntimeError("OmeTiffReader not available")
            reader = OmeTiffReader(str(self.image_path))
            try:
                arr = reader.get_channel_roi(ch_idx, y=y, x=x, height=height, width=width, level=0)
            finally:
                reader.close()
            if progress_callback is not None:
                with lock:
                    progress_callback(ch_idx)
            return ch_idx, arr

        channels: dict = {}
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_read_channel, ch): ch for ch in range(self.num_channels)}
            for fut in as_completed(futures):
                ch_idx, arr = fut.result()
                channels[ch_idx] = arr

        # Return sorted by channel index for deterministic downstream processing
        return dict(sorted(channels.items()))

    def get_channel_names(self):
        """Get list of channel names."""
        return self.metadata.get('channel_names', [])

    def get_channel_calibration(self, channel_idx: int) -> dict:
        """
        Get calibration (percentile stretching values) for a channel.

        Computes calibration from the display-level image to be fast.
        Results are cached.

        Args:
            channel_idx: Channel index (0 to num_channels-1)

        Returns:
            dict with keys: p2, p98, min, max, mean
        """
        if channel_idx in self._calibration_cache:
            return self._calibration_cache[channel_idx]

        # Load channel at display level for calibration
        img, _, _, _ = self.get_channel_downsampled(channel_idx)

        # Compute calibration
        calibration = compute_channel_calibration(img)
        self._calibration_cache[channel_idx] = calibration

        logger.info(
            f"Computed calibration for channel {channel_idx}: "
            f"p2={calibration['p2']:.1f}, p98={calibration['p98']:.1f}"
        )

        return calibration

    def close(self):
        """Close the reader."""
        self.reader.close()

    def __del__(self):
        """Ensure reader is closed."""
        try:
            self.close()
        except:
            pass


class ProgressReader:
    """
    Thin proxy around OmeTiffReader that intercepts `get_channel_roi` calls
    to report per-channel progress to an ExtractionJob.

    Used so that `save_region()` (from the library, which we don't own) can
    drive progress updates without modification.

    Thread-safety: progress_callback is called under a lock supplied by caller.
    """

    def __init__(self, reader, num_channels: int, progress_callback: callable):
        """
        Args:
            reader: The real OmeTiffReader instance.
            num_channels: Total number of channels (for progress denominator).
            progress_callback: Callable(channel_idx: int) called after each
                channel read completes.
        """
        self._reader = reader
        self._num_channels = num_channels
        self._progress_callback = progress_callback
        self._lock = threading.Lock()

    def get_channel_roi(self, channel_idx, y, x, height, width, level=0):
        """Delegate to real reader, then fire progress callback."""
        result = self._reader.get_channel_roi(channel_idx, y=y, x=x,
                                               height=height, width=width,
                                               level=level)
        with self._lock:
            self._progress_callback(channel_idx)
        return result

    # Proxy every other attribute access to the wrapped reader
    def __getattr__(self, name):
        return getattr(self._reader, name)
