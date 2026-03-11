"""
Region writers: HDF5, streaming OME-TIFF, and QC thumbnails.

Each writer reads one channel at a time from the lazy reader, keeping peak
memory at ~200 MB per channel crop regardless of region size.

All functions are private — called by :func:`tissue_extractor.extract_tissue_regions`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from xml.sax.saxutils import quoteattr

import h5py
import numpy as np
import tifffile

if TYPE_CHECKING:
    from library.imaging_analysis.tissue_extractor import TissueRegion
    from library.imaging.reader import OmeTiffReader

logger = logging.getLogger(__name__)

# Try to import LZ4 compression for faster HDF5 writes.
try:
    import hdf5plugin  # noqa: F401

    _HAS_LZ4 = True
except ImportError:
    _HAS_LZ4 = False


def _compute_level_roi(
    padded_ymin: int,
    padded_xmin: int,
    padded_ymax: int,
    padded_xmax: int,
    full_h: int,
    full_w: int,
    level_h: int,
    level_w: int,
) -> tuple[int, int, int, int]:
    """Convert full-resolution bbox to pyramid level coordinates.

    Parameters
    ----------
    padded_ymin, padded_xmin, padded_ymax, padded_xmax : int
        Bounding box in full-resolution (level 0) pixels.
    full_h, full_w : int
        Full-resolution image height and width.
    level_h, level_w : int
        Pyramid level height and width.

    Returns
    -------
    y_level, x_level, h_level, w_level : tuple[int, int, int, int]
        ROI coordinates and dimensions in pyramid level pixel space.
    """
    scale_y = level_h / full_h
    scale_x = level_w / full_w

    y_level = int(np.floor(padded_ymin * scale_y))
    x_level = int(np.floor(padded_xmin * scale_x))
    h_level = max(1, int(np.round((padded_ymax - padded_ymin) * scale_y)))
    w_level = max(1, int(np.round((padded_xmax - padded_xmin) * scale_x)))

    # Clamp to level bounds
    y_level = min(y_level, level_h - 1)
    x_level = min(x_level, level_w - 1)
    h_level = min(h_level, level_h - y_level)
    w_level = min(w_level, level_w - x_level)

    return y_level, x_level, h_level, w_level


def save_region(
    region: TissueRegion,
    reader: OmeTiffReader,
    metadata: dict,
    output_dir: str,
    output_formats: list[str],
    verbose: bool = True,
) -> TissueRegion:
    """Save a single tissue region in all requested formats.

    Writes formats sequentially within one region to avoid doubling memory
    from overlapping reads. This function is the unit of work for parallel
    execution (one region per thread).

    Parameters
    ----------
    region : TissueRegion
        Region specification with bounding box coordinates.
    reader : OmeTiffReader
        Open lazy reader for the source image.
    metadata : dict
        Output of ``extract_ome_metadata()`` with channel_names, pixel_size_um.
    output_dir : str
        Base output directory. A subfolder ``region_{id:03d}/`` is created.
    output_formats : list[str]
        Formats to save: ``["hdf5", "ometiff"]`` or subsets.
    verbose : bool
        Log progress.

    Returns
    -------
    TissueRegion
        Updated region with output file paths and success flag.
    """
    region_dir = Path(output_dir) / f"region_{region.region_id:03d}"
    region_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    try:
        if "hdf5" in output_formats:
            _save_region_hdf5(region, reader, metadata, region_dir, verbose)
    except Exception as e:
        errors.append(f"HDF5: {e}")
        logger.error(f"Region {region.region_id} HDF5 failed: {e}", exc_info=True)

    try:
        if "ometiff" in output_formats:
            _save_region_ometiff(region, reader, metadata, region_dir, verbose)
    except Exception as e:
        errors.append(f"OME-TIFF: {e}")
        logger.error(f"Region {region.region_id} OME-TIFF failed: {e}", exc_info=True)

    try:
        # Thumbnail (always generated for QC).
        _save_region_thumbnail(region, reader, region_dir, verbose=verbose)
    except Exception as e:
        errors.append(f"Thumbnail: {e}")
        logger.warning(f"Region {region.region_id} thumbnail failed: {e}")

    if errors:
        region.success = False
        region.error_message = "; ".join(errors)
    else:
        region.success = True

    return region


def _save_region_hdf5(
    region: TissueRegion,
    reader: OmeTiffReader,
    metadata: dict,
    region_dir: Path,
    verbose: bool = True,
) -> None:
    """Save region as HDF5 with pyramid levels and channel-by-channel streaming.

    Writes all pyramid levels from the source image. Full-resolution data is stored
    in both `channels/` (backward compat) and `pyramid/level_0/` (for consistency).
    Lower-resolution levels are stored in `pyramid/level_1/`, etc.

    Parameters
    ----------
    region : TissueRegion
        Region specification.
    reader : OmeTiffReader
        Open lazy reader.
    metadata : dict
        Channel names, pixel sizes, and full OME object.
    region_dir : Path
        Output folder for this region.
    verbose : bool
        Log progress.
    """
    output_file = region_dir / f"region_{region.region_id:03d}.h5"
    h = region.padded_ymax - region.padded_ymin
    w = region.padded_xmax - region.padded_xmin
    _, full_h, full_w = reader.shape

    compression_kwargs: dict
    if _HAS_LZ4:
        compression_kwargs = {"compression": hdf5plugin.LZ4()}  # type: ignore[name-defined]
        logger.debug("Using LZ4 compression for HDF5")
    else:
        compression_kwargs = {"compression": "gzip", "compression_opts": 4}

    with h5py.File(output_file, "w") as f:
        # --- Full-resolution channels (level 0, backward compat) ---
        channels_grp = f.create_group("channels")

        for ch_idx in range(reader.num_channels):
            crop = reader.get_channel_roi(
                ch_idx,
                y=region.padded_ymin,
                x=region.padded_xmin,
                height=h,
                width=w,
                level=0,
            )

            dset = channels_grp.create_dataset(
                f"ch_{ch_idx:02d}",
                data=crop,
                chunks=(min(256, crop.shape[0]), min(256, crop.shape[1])),
                **compression_kwargs,
            )
            dset.attrs["channel_name"] = metadata["channel_names"][ch_idx]
            dset.attrs["channel_index"] = ch_idx

        # --- Pyramid levels (new structure, levels 1+ only) ---
        # Level 0 data is already in channels/ above; start from level 1 to avoid
        # re-reading full-resolution data from disk for every channel.
        if reader.num_levels > 1:
            pyramid_grp = f.create_group("pyramid")

            # Pre-compute scale factors for all levels (metadata-only, no I/O).
            scale_factors_y = []
            scale_factors_x = []
            for level in range(reader.num_levels):
                _, level_h, level_w = reader.level_shape(level)
                scale_factors_y.append(level_h / full_h)
                scale_factors_x.append(level_w / full_w)

            for level in range(1, reader.num_levels):
                _, level_h, level_w = reader.level_shape(level)
                y_l, x_l, h_l, w_l = _compute_level_roi(
                    region.padded_ymin, region.padded_xmin,
                    region.padded_ymax, region.padded_xmax,
                    full_h, full_w, level_h, level_w,
                )

                level_grp = pyramid_grp.create_group(f"level_{level}")

                for ch_idx in range(reader.num_channels):
                    crop = reader.get_channel_roi(
                        ch_idx,
                        y=y_l,
                        x=x_l,
                        height=h_l,
                        width=w_l,
                        level=level,
                    )

                    dset = level_grp.create_dataset(
                        f"ch_{ch_idx:02d}",
                        data=crop,
                        chunks=(min(256, crop.shape[0]), min(256, crop.shape[1])),
                        **compression_kwargs,
                    )
                    dset.attrs["channel_name"] = metadata["channel_names"][ch_idx]
                    dset.attrs["channel_index"] = ch_idx

            # Actual inter-level downsample factor derived from scale factors.
            # Computed from levels 0→1 (most reliable); 2.0 is the typical but
            # not guaranteed value for ORION pyramid stacks.
            if len(scale_factors_y) >= 2 and scale_factors_y[1] > 0:
                actual_ds = round(1.0 / scale_factors_y[1], 6)
            else:
                actual_ds = 2.0  # safe fallback

            # Pyramid metadata
            pyr_meta_grp = f.create_group("pyramid_metadata")
            pyr_meta_grp.attrs["num_levels"] = reader.num_levels
            pyr_meta_grp.attrs["downsample_factor"] = actual_ds
            pyr_meta_grp.create_dataset("scale_y_per_level", data=scale_factors_y)
            pyr_meta_grp.create_dataset("scale_x_per_level", data=scale_factors_x)

        # Metadata group.
        meta_grp = f.create_group("metadata")
        meta_grp.attrs["num_channels"] = reader.num_channels
        meta_grp.attrs["region_height_px"] = h
        meta_grp.attrs["region_width_px"] = w
        meta_grp.attrs["pixel_size_x_um"] = metadata["pixel_size_um"]["x_um"]
        meta_grp.attrs["pixel_size_y_um"] = metadata["pixel_size_um"]["y_um"]

        # Channel name list for convenient access.
        ch_names = metadata["channel_names"][: reader.num_channels]
        meta_grp.create_dataset(
            "channel_names",
            data=[n.encode("utf-8") for n in ch_names],
        )

        # Provenance group.
        prov_grp = f.create_group("provenance")
        prov_grp.attrs["source_file"] = str(reader.path)
        prov_grp.attrs["extraction_timestamp"] = datetime.now().isoformat()
        prov_grp.attrs["region_id"] = region.region_id
        prov_grp.attrs["bbox_ymin_px"] = region.bbox_ymin
        prov_grp.attrs["bbox_xmin_px"] = region.bbox_xmin
        prov_grp.attrs["bbox_ymax_px"] = region.bbox_ymax
        prov_grp.attrs["bbox_xmax_px"] = region.bbox_xmax
        prov_grp.attrs["padded_ymin_px"] = region.padded_ymin
        prov_grp.attrs["padded_xmin_px"] = region.padded_xmin
        prov_grp.attrs["padded_ymax_px"] = region.padded_ymax
        prov_grp.attrs["padded_xmax_px"] = region.padded_xmax
        prov_grp.attrs["area_px"] = region.area_px
        prov_grp.attrs["area_um2"] = region.area_um2
        prov_grp.attrs["centroid_y_px"] = region.centroid_y_px
        prov_grp.attrs["centroid_x_px"] = region.centroid_x_px
        prov_grp.attrs["origin_um_y"] = region.origin_um_y
        prov_grp.attrs["origin_um_x"] = region.origin_um_x
        prov_grp.attrs["num_pyramid_levels"] = reader.num_levels

        # Store full OME-XML for metadata reconstruction
        if "ome_obj" in metadata and metadata["ome_obj"] is not None:
            try:
                from ome_types import to_xml
                ome_xml_str = to_xml(metadata["ome_obj"])
                prov_grp.attrs["ome_xml_full"] = ome_xml_str
            except Exception as e:
                logger.warning(f"Could not serialize OME object to XML: {e}")

    region.output_hdf5 = str(output_file)
    region.num_pyramid_levels = reader.num_levels

    if verbose:
        size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(
            f"Region {region.region_id} HDF5: {size_mb:.1f} MB "
            f"({reader.num_levels} pyramid levels)"
        )


def _save_region_ometiff(
    region: TissueRegion,
    reader: OmeTiffReader,
    metadata: dict,
    region_dir: Path,
    verbose: bool = True,
) -> None:
    """Save region as OME-TIFF with pyramid levels and streaming page-by-page writing.

    Writes one channel at a time using ``tifffile.TiffWriter``, keeping peak
    memory at one channel crop (~200 MB) instead of the full (C, H, W) stack.
    If source has multiple pyramid levels, writes them as TIFF SubIFDs.

    Parameters
    ----------
    region : TissueRegion
        Region specification.
    reader : OmeTiffReader
        Open lazy reader.
    metadata : dict
        Channel names, pixel sizes, and full OME object.
    region_dir : Path
        Output folder for this region.
    verbose : bool
        Log progress.
    """
    output_file = region_dir / f"region_{region.region_id:03d}.ome.tiff"
    h = region.padded_ymax - region.padded_ymin
    w = region.padded_xmax - region.padded_xmin
    _, full_h, full_w = reader.shape

    ome_xml = _create_ome_xml(
        num_channels=reader.num_channels,
        height=h,
        width=w,
        dtype_str=str(reader.dtype),
        channel_names=metadata["channel_names"],
        pixel_size_x_um=metadata["pixel_size_um"]["x_um"],
        pixel_size_y_um=metadata["pixel_size_um"]["y_um"],
        origin_x_um=region.origin_um_x,
        origin_y_um=region.origin_um_y,
        num_pyramid_levels=reader.num_levels,
        ome_obj=metadata.get("ome_obj"),
    )

    # TIFF spec: tile dimensions must be multiples of 16.
    # Use 256×256 always — tifffile pads automatically when image is smaller.
    _TILE = (256, 256)
    _num_sublevels = reader.num_levels - 1  # pyramid levels below full-res

    with tifffile.TiffWriter(output_file, bigtiff=True) as tw:
        # Write all channels; for each channel write full-res then all pyramid levels.
        # subifds= on the base write reserves SubIFD slots; subsequent subfiletype=1
        # writes fill those slots in order.
        # Note: do NOT pass metadata= alongside description=; they both target TIFF
        # tag 270 (ImageDescription) and will conflict in older tifffile versions.
        for ch_idx in range(reader.num_channels):
            # --- Level 0: full resolution ---
            crop = reader.get_channel_roi(
                ch_idx,
                y=region.padded_ymin,
                x=region.padded_xmin,
                height=h,
                width=w,
                level=0,
            )
            # subifds= must be passed on EVERY channel's full-resolution IFD.
            # Each channel declares N SubIFD slots, then immediately writes all N
            # pyramid levels (subfiletype=1) before moving to the next channel.
            # tifffile correctly associates each channel's subfiletype=1 writes
            # with that channel's parent IFD because the slots are filled in the
            # same loop iteration. Passing subifds only on ch_idx==0 leaves all
            # other channels without pyramid pointers → orphaned reduced-res pages
            # that QuPath/Bio-Formats cannot link to their parent channel.
            tw.write(
                crop,
                photometric="minisblack",
                compression="deflate",
                tile=_TILE,
                subifds=_num_sublevels,
                description=ome_xml if ch_idx == 0 else None,
                metadata=None,
            )

            # --- Levels 1+: reduced resolution SubIFDs ---
            for level in range(1, reader.num_levels):
                _, level_h, level_w = reader.level_shape(level)
                y_l, x_l, h_l, w_l = _compute_level_roi(
                    region.padded_ymin, region.padded_xmin,
                    region.padded_ymax, region.padded_xmax,
                    full_h, full_w, level_h, level_w,
                )
                crop = reader.get_channel_roi(
                    ch_idx,
                    y=y_l,
                    x=x_l,
                    height=h_l,
                    width=w_l,
                    level=level,
                )
                tw.write(
                    crop,
                    photometric="minisblack",
                    compression="deflate",
                    tile=_TILE,
                    subfiletype=1,
                    metadata=None,
                )

    region.output_ometiff = str(output_file)
    region.num_pyramid_levels = reader.num_levels

    if verbose:
        size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(
            f"Region {region.region_id} OME-TIFF: {size_mb:.1f} MB "
            f"({reader.num_levels} pyramid levels)"
        )


def _save_region_thumbnail(
    region: TissueRegion,
    reader: OmeTiffReader,
    region_dir: Path,
    thumbnail_width: int = 512,
    verbose: bool = True,
) -> None:
    """Save a downsampled Hoechst thumbnail PNG for quick QC.

    Parameters
    ----------
    region : TissueRegion
        Region specification.
    reader : OmeTiffReader
        Open lazy reader.
    region_dir : Path
        Output folder for this region.
    thumbnail_width : int
        Target width in pixels for the thumbnail.
    verbose : bool
        Log progress.
    """
    h = region.padded_ymax - region.padded_ymin
    w = region.padded_xmax - region.padded_xmin

    # Try to read from a pyramid level that covers the region.
    # Fall back to full res + downsample if needed.
    hoechst = reader.get_channel_roi(
        0,
        y=region.padded_ymin,
        x=region.padded_xmin,
        height=h,
        width=w,
    )

    # Downsample to thumbnail size.
    factor = max(1, w // thumbnail_width)
    if factor > 1:
        thumb = hoechst[::factor, ::factor]
    else:
        thumb = hoechst

    # Contrast stretch (2nd–98th percentile).
    p2, p98 = np.percentile(thumb, (2, 98))
    if p98 > p2:
        thumb_f = (thumb.astype(np.float32) - p2) / (p98 - p2)
        thumb_f = np.clip(thumb_f, 0.0, 1.0)
        thumb_u8 = (thumb_f * 255).astype(np.uint8)
    else:
        thumb_u8 = np.zeros_like(thumb, dtype=np.uint8)

    output_file = region_dir / f"region_{region.region_id:03d}_thumbnail.png"

    tifffile.imwrite(str(output_file).replace(".png", ".tif"), thumb_u8)

    # Use a simple approach: save as 8-bit TIFF (universal compatibility).
    # If PIL is available, save as PNG for smaller files.
    try:
        from PIL import Image

        img = Image.fromarray(thumb_u8, mode="L")
        img.save(output_file)
        # Remove the tif fallback if png succeeded.
        tif_path = Path(str(output_file).replace(".png", ".tif"))
        if tif_path.exists():
            tif_path.unlink()
    except ImportError:
        # Rename .tif to .png path for consistency in manifest.
        output_file = Path(str(output_file).replace(".png", ".tif"))

    region.output_thumbnail = str(output_file)

    if verbose:
        logger.info(
            f"Region {region.region_id} thumbnail: "
            f"{thumb_u8.shape[1]}×{thumb_u8.shape[0]} px"
        )


def _create_ome_xml(
    num_channels: int,
    height: int,
    width: int,
    dtype_str: str,
    channel_names: list[str],
    pixel_size_x_um: float,
    pixel_size_y_um: float,
    origin_x_um: float,
    origin_y_um: float,
    num_pyramid_levels: int = 1,
    ome_obj: Optional[object] = None,
) -> str:
    """Create OME-XML description for a multi-channel extracted region.

    Includes full channel metadata (wavelengths, fluorophores, etc.) if ome_obj is provided.

    Parameters
    ----------
    num_channels : int
        Number of channels.
    height, width : int
        Spatial dimensions in pixels (full resolution, level 0).
    dtype_str : str
        Numpy dtype string (e.g. 'uint16').
    channel_names : list[str]
        Channel name strings.
    pixel_size_x_um, pixel_size_y_um : float
        Physical pixel size (micrometers).
    origin_x_um, origin_y_um : float
        Physical origin of the extracted region (micrometers).
    num_pyramid_levels : int
        Number of pyramid levels in output (for informational comment).
    ome_obj : ome_types.OME, optional
        Full OME object with channel metadata. If provided, full metadata is embedded.

    Returns
    -------
    str
        Valid OME-XML string for embedding in TIFF tag 270.
        Uses ASCII "um" for PhysicalSizeXUnit / PhysicalSizeYUnit because
        tifffile encodes tag 270 as ASCII (not UTF-8), and the Unicode µ
        symbol (U+00B5) causes a UnicodeEncodeError.
    """
    # Use valid OME unit: 'nm' (nanometer)
    # Source files use 'µm' (Unicode), but tifffile requires ASCII for tag 270.
    # 'nm' is a valid OME-2016-06 enum value that is ASCII-safe.
    # Conversion: 0.325 µm = 325 nm (multiply by 1000).
    unit = "nm"
    pixel_size_x_nm = pixel_size_x_um * 1000.0
    pixel_size_y_nm = pixel_size_y_um * 1000.0

    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 '
        'http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">',
        f'  <Image ID="Image:0" Name="Extracted Region">',
        f'    <!-- Multi-resolution pyramid: {num_pyramid_levels} levels via TIFF SubIFDs -->',
        f'    <Pixels ID="Pixels:0" DimensionOrder="XYCZT" '
        f'Type="{dtype_str}" '
        f'SizeX="{width}" SizeY="{height}" SizeZ="1" '
        f'SizeC="{num_channels}" SizeT="1" '
        f'SamplesPerPixel="1" '
        f'PhysicalSizeX="{pixel_size_x_nm}" PhysicalSizeXUnit="{unit}" '
        f'PhysicalSizeY="{pixel_size_y_nm}" PhysicalSizeYUnit="{unit}">',
        f'      <!-- Physical origin: ({origin_x_um:.4f}, {origin_y_um:.4f}) um (stored as {pixel_size_x_nm} {unit} above) -->',
    ]

    # Extract channel metadata from ome_obj if available
    channel_metadata = {}
    if ome_obj is not None:
        try:
            px = ome_obj.images[0].pixels
            for ch_idx, ch in enumerate(px.channels):
                metadata = {"name": ch.name or channel_names[ch_idx]}
                if ch.excitation_wavelength is not None:
                    metadata["excitation_wavelength"] = float(ch.excitation_wavelength)
                if ch.emission_wavelength is not None:
                    metadata["emission_wavelength"] = float(ch.emission_wavelength)
                if ch.fluor is not None:
                    metadata["fluorophore"] = str(ch.fluor)
                channel_metadata[ch_idx] = metadata
        except Exception as e:
            logger.debug(f"Could not extract channel metadata from OME object: {e}")

    # Write channel elements with metadata
    for ch_idx in range(num_channels):
        name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Channel_{ch_idx}"
        ch_meta = channel_metadata.get(ch_idx, {})

        # Build channel element with metadata.
        # In OME-XML 2016-06: Fluor is an attribute of <Channel>, not a child element.
        # Excitation/EmissionWavelength are also attributes (in nm).
        # Use quoteattr() for all string values to escape &, <, >, " in channel names
        # (e.g. names containing '&' from instrument export files would break XML).
        ch_attrs = f'ID="Channel:{ch_idx}" Name={quoteattr(name)}'
        if "fluorophore" in ch_meta:
            ch_attrs += f' Fluor={quoteattr(str(ch_meta["fluorophore"]))}'
        if "excitation_wavelength" in ch_meta:
            ch_attrs += f' ExcitationWavelength="{ch_meta["excitation_wavelength"]}"'
        if "emission_wavelength" in ch_meta:
            ch_attrs += f' EmissionWavelength="{ch_meta["emission_wavelength"]}"'

        xml_lines.append(f'      <Channel {ch_attrs}/>')

    xml_lines.extend([
        "    </Pixels>",
        "  </Image>",
        "</OME>",
    ])

    return "\n".join(xml_lines)
