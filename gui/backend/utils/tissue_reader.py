"""
Utility for reading extracted tissue regions from HDF5 files.

Provides convenient access to channel data, metadata, and provenance
from tissue regions extracted by the GUI.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def read_tissue_region(hdf5_path: str) -> Dict:
    """
    Open and read a tissue region HDF5 file.

    Args:
        hdf5_path: Absolute or relative path to region_000.h5

    Returns:
        Dict with keys:
        - 'channels': {ch_idx: uint16 ndarray} for all 20 channels
        - 'channel_names': List[str] of channel names
        - 'shape': (height, width) of extracted region
        - 'dtype': numpy dtype (uint16)
        - 'metadata': Dict with additional metadata if present
        - 'provenance': Dict with extraction info if present

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a valid tissue region HDF5
        ImportError: If h5py is not installed
    """
    import h5py

    h5_path = Path(hdf5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    logger.info(f"Reading tissue region: {h5_path}")

    with h5py.File(str(h5_path), 'r') as f:
        # Validate structure
        if 'channels' not in f:
            raise ValueError(
                f"Invalid tissue region file: missing /channels group"
            )

        # Read all channels
        channels = {}
        channel_names = []
        shape = None

        for ch_idx in range(20):
            ch_key = f'ch_{ch_idx:02d}'
            if ch_key not in f['channels']:
                logger.warning(f"Missing channel {ch_idx}")
                continue

            ch_data = f['channels'][ch_key][()]
            channels[ch_idx] = ch_data
            shape = ch_data.shape

        # Read channel names
        if 'metadata' in f and 'channel_names' in f['metadata']:
            ch_names_arr = f['metadata']['channel_names'][()]
            # Convert bytes to strings if needed
            channel_names = [
                name.decode('utf-8') if isinstance(name, bytes) else name
                for name in ch_names_arr
            ]
        else:
            # Fallback: generate default names
            channel_names = [f'Channel {i}' for i in range(20)]

        # Read metadata
        metadata = {}
        if 'metadata' in f:
            for key in f['metadata'].attrs:
                metadata[key] = f['metadata'].attrs[key]
            for key in f['metadata'].keys():
                if key != 'channel_names':
                    val = f['metadata'][key][()]
                    metadata[key] = val.item() if isinstance(val, np.ndarray) else val

        # Read provenance
        provenance = {}
        if 'provenance' in f:
            for key in f['provenance'].attrs:
                provenance[key] = f['provenance'].attrs[key]
            for key in f['provenance'].keys():
                val = f['provenance'][key][()]
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                provenance[key] = val.item() if isinstance(val, np.ndarray) else val

        logger.info(
            f"Loaded tissue region: shape={shape}, "
            f"channels={len(channels)}, dtype={channels[0].dtype if channels else 'unknown'}"
        )

        return {
            'channels': channels,
            'channel_names': channel_names,
            'shape': shape,
            'dtype': channels[0].dtype if channels else np.uint16,
            'num_channels': len(channels),
            'metadata': metadata,
            'provenance': provenance,
        }


def get_channel(hdf5_path: str, channel_idx: int) -> np.ndarray:
    """
    Convenience function to read a single channel from a tissue region.

    Args:
        hdf5_path: Path to region_000.h5
        channel_idx: Channel index (0-19)

    Returns:
        uint16 ndarray of shape (height, width)

    Raises:
        FileNotFoundError, ValueError, ImportError (see read_tissue_region)
        IndexError: If channel_idx is out of range
    """
    region = read_tissue_region(hdf5_path)
    if channel_idx not in region['channels']:
        raise IndexError(
            f"Channel {channel_idx} not found. Available: {list(region['channels'].keys())}"
        )
    return region['channels'][channel_idx]


def get_channel_by_name(hdf5_path: str, channel_name: str) -> Tuple[np.ndarray, int]:
    """
    Read a channel by name (e.g., 'Hoechst', 'CD31', 'IBA1').

    Args:
        hdf5_path: Path to region_000.h5
        channel_name: Substring of channel name (case-insensitive)
                      E.g., 'hoechst', 'cd31', 'iba1'

    Returns:
        (data, channel_idx) where data is uint16 ndarray, channel_idx is int (0-19)

    Raises:
        ValueError: If channel name not found or is ambiguous
    """
    region = read_tissue_region(hdf5_path)
    channel_names = region['channel_names']

    # Find matching channels (case-insensitive substring match)
    matches = [
        (i, name) for i, name in enumerate(channel_names)
        if channel_name.lower() in name.lower()
    ]

    if not matches:
        raise ValueError(
            f"Channel '{channel_name}' not found. Available channels:\n" +
            '\n'.join(f"  {i}: {name}" for i, name in enumerate(channel_names))
        )

    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous channel name '{channel_name}'. Matches:\n" +
            '\n'.join(f"  {i}: {name}" for i, name in matches) +
            "\nPlease be more specific."
        )

    ch_idx, ch_name = matches[0]
    return region['channels'][ch_idx], ch_idx


def get_region_info(hdf5_path: str) -> Dict:
    """
    Get metadata summary of a tissue region without loading channel data.

    Useful for quickly checking region dimensions, channel names, etc.
    without the I/O cost of reading all 20 channels.

    Args:
        hdf5_path: Path to region_000.h5

    Returns:
        Dict with keys:
        - 'shape': (height, width)
        - 'num_channels': int
        - 'dtype': numpy dtype
        - 'channel_names': List[str]
        - 'file_size_gb': float
    """
    import h5py

    h5_path = Path(hdf5_path)
    file_size_gb = h5_path.stat().st_size / 1e9

    with h5py.File(str(h5_path), 'r') as f:
        if 'channels' not in f or 'ch_00' not in f['channels']:
            raise ValueError("Not a valid tissue region HDF5 file")

        ch_00 = f['channels']['ch_00']
        shape = ch_00.shape
        dtype = ch_00.dtype

        channel_names = []
        if 'metadata' in f and 'channel_names' in f['metadata']:
            ch_names_arr = f['metadata']['channel_names'][()]
            channel_names = [
                name.decode('utf-8') if isinstance(name, bytes) else name
                for name in ch_names_arr
            ]

        return {
            'shape': shape,
            'num_channels': 20,
            'dtype': dtype,
            'channel_names': channel_names,
            'file_size_gb': round(file_size_gb, 2),
        }


def get_region_pyramid_info(hdf5_path: str) -> Dict:
    """Get pyramid level information from an extracted tissue region HDF5.

    Returns metadata about available pyramid levels without loading pixel data.

    Parameters
    ----------
    hdf5_path : str
        Path to the extracted region HDF5 file.

    Returns
    -------
    Dict
        Keys:
        - 'num_levels': int — number of pyramid levels
        - 'level_shapes': List[Tuple[int, int]] — (height, width) at each level
        - 'scale_factors_y': List[float] — scale factor per level (height_level / height_full)
        - 'scale_factors_x': List[float] — scale factor per level (width_level / width_full)
    """
    import h5py

    h5_path = Path(hdf5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(str(h5_path), 'r') as f:
        if 'pyramid_metadata' not in f:
            # No pyramid data — single resolution only
            ch_00 = f['channels']['ch_00']
            h, w = ch_00.shape
            return {
                'num_levels': 1,
                'level_shapes': [(h, w)],
                'scale_factors_y': [1.0],
                'scale_factors_x': [1.0],
            }

        pyr_meta = f['pyramid_metadata']
        num_levels = pyr_meta.attrs['num_levels']

        scale_y = pyr_meta['scale_y_per_level'][()]
        scale_x = pyr_meta['scale_x_per_level'][()]

        # Compute level shapes from scale factors
        full_h, full_w = f['metadata'].attrs['region_height_px'], f['metadata'].attrs['region_width_px']
        level_shapes = [
            (int(round(full_h * scale_y[i])), int(round(full_w * scale_x[i])))
            for i in range(num_levels)
        ]

        return {
            'num_levels': num_levels,
            'level_shapes': level_shapes,
            'scale_factors_y': list(scale_y),
            'scale_factors_x': list(scale_x),
        }


def get_channel_at_level(hdf5_path: str, channel_idx: int, level: int = 0) -> np.ndarray:
    """Read a single channel at a specific pyramid level.

    Parameters
    ----------
    hdf5_path : str
        Path to the extracted region HDF5 file.
    channel_idx : int
        Channel index (0-19).
    level : int, optional
        Pyramid level (0 = full resolution, default). Higher numbers are downsampled.

    Returns
    -------
    np.ndarray
        Channel data at the requested level (uint16).

    Raises
    ------
    FileNotFoundError
        If file doesn't exist.
    ValueError
        If file is not a valid tissue region HDF5.
    IndexError
        If channel_idx or level are out of range.
    """
    import h5py

    h5_path = Path(hdf5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(str(h5_path), 'r') as f:
        if 'channels' not in f:
            raise ValueError(f"Invalid tissue region file: missing /channels group")

        if level == 0:
            # Full resolution from channels/ group (backward compat)
            ch_key = f'ch_{channel_idx:02d}'
            if ch_key not in f['channels']:
                raise IndexError(f"Channel {channel_idx} not found at level 0")
            return f['channels'][ch_key][()]

        # Downsampled level from pyramid/ group
        if 'pyramid' not in f:
            raise IndexError(f"No pyramid data; level {level} not available (only level 0)")

        level_key = f'level_{level}'
        if level_key not in f['pyramid']:
            raise IndexError(
                f"Pyramid level {level} not found. Available levels: "
                f"{list(f['pyramid'].keys())}"
            )

        ch_key = f'ch_{channel_idx:02d}'
        if ch_key not in f['pyramid'][level_key]:
            raise IndexError(f"Channel {channel_idx} not found at level {level}")

        return f['pyramid'][level_key][ch_key][()]


def get_channel_metadata(hdf5_path: str, channel_idx: int) -> Dict:
    """Extract full channel metadata (wavelengths, fluorophore, etc.) from provenance.

    Parses the stored OME-XML to reconstruct channel properties from the original image.

    Parameters
    ----------
    hdf5_path : str
        Path to the extracted region HDF5 file.
    channel_idx : int
        Channel index (0-19).

    Returns
    -------
    Dict
        Channel properties with keys:
        - 'name': str
        - 'excitation_wavelength': float (nm) or None
        - 'emission_wavelength': float (nm) or None
        - 'fluorophore': str or None
        ... and any other metadata available in OME-XML
    """
    import h5py

    h5_path = Path(hdf5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(str(h5_path), 'r') as f:
        if 'channels' not in f:
            raise ValueError(f"Invalid tissue region file: missing /channels group")

        # Get channel name from attributes
        ch_key = f'ch_{channel_idx:02d}'
        if ch_key not in f['channels']:
            raise IndexError(f"Channel {channel_idx} not found")

        metadata = {}
        if 'channel_name' in f['channels'][ch_key].attrs:
            metadata['name'] = f['channels'][ch_key].attrs['channel_name']
        else:
            metadata['name'] = f"Channel {channel_idx}"

        # Try to extract from stored OME-XML
        if 'provenance' in f and 'ome_xml_full' in f['provenance'].attrs:
            try:
                from ome_types import from_xml

                ome_xml_str = f['provenance'].attrs['ome_xml_full']
                ome = from_xml(ome_xml_str)
                px = ome.images[0].pixels

                if channel_idx < len(px.channels):
                    ch = px.channels[channel_idx]
                    if ch.excitation_wavelength is not None:
                        metadata['excitation_wavelength'] = float(ch.excitation_wavelength)
                    if ch.emission_wavelength is not None:
                        metadata['emission_wavelength'] = float(ch.emission_wavelength)
                    if ch.fluorophore is not None:
                        metadata['fluorophore'] = str(ch.fluorophore)
            except Exception as e:
                logger.debug(f"Could not parse OME-XML from provenance: {e}")

        return metadata


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python tissue_reader.py <path_to_region.h5>")
        print("\nExample:")
        print("  python tissue_reader.py extracted_tissues/cortex/region_000.h5")
        sys.exit(1)

    hdf5_file = sys.argv[1]

    try:
        # Print region info
        info = get_region_info(hdf5_file)
        print(f"\n=== Tissue Region ===")
        print(f"Shape: {info['shape']}")
        print(f"Channels: {info['num_channels']}")
        print(f"Dtype: {info['dtype']}")
        print(f"File size: {info['file_size_gb']} GB")
        print(f"\nChannel names:")
        for i, name in enumerate(info['channel_names']):
            print(f"  [{i:2d}] {name}")

        # Pyramid info if available
        try:
            pyr_info = get_region_pyramid_info(hdf5_file)
            print(f"\n=== Pyramid Info ===")
            print(f"Levels: {pyr_info['num_levels']}")
            for i, shape in enumerate(pyr_info['level_shapes']):
                print(f"  Level {i}: {shape[1]}x{shape[0]} px (scale: {pyr_info['scale_factors_y'][i]:.3f})")
        except Exception as e:
            print(f"No pyramid data: {e}")

        # Example: read a channel by name
        print(f"\n=== Reading Hoechst channel ===")
        hoechst, ch_idx = get_channel_by_name(hdf5_file, 'hoechst')
        print(f"Channel {ch_idx}: {hoechst.dtype} {hoechst.shape}")
        print(f"  Min: {hoechst.min()}, Max: {hoechst.max()}, Mean: {hoechst.mean():.1f}")

        # Channel metadata
        ch_meta = get_channel_metadata(hdf5_file, ch_idx)
        print(f"\nMetadata:")
        for key, val in ch_meta.items():
            print(f"  {key}: {val}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
