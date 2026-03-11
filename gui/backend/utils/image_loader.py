"""
Image loader for OME-TIFF with pyramid level support.
Coordinates are always logged for verification.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from library.imaging import OmeTiffReader, extract_ome_metadata, get_pixel_size_um

logger = logging.getLogger(__name__)


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
