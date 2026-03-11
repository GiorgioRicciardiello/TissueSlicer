"""
test_pyramid_fix.py
====================
Extract the region from the coordinates JSON, then validate that the
output OME-TIFF has properly linked SubIFD pyramid chains for ALL channels.

Usage:
    python test_pyramid_fix.py

Outputs to: extracted_tissues/test_pyramid_fix/
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import tifffile
import zarr

# Add project root to path so library/ is importable.
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from library.imaging import OmeTiffReader, extract_ome_metadata, get_pixel_size_um
from library.imaging_analysis.tissue_extractor import TissueRegion
from library.imaging_analysis._writers import save_region

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — read from the existing coordinates JSON
# ---------------------------------------------------------------------------

COORDS_JSON = Path(
    "extracted_tissues/test/region_000/region_000_coordinates.json"
)
OUTPUT_DIR = Path("extracted_tissues/test_pyramid_fix")
IMAGE_PATH = Path(
    r"C:\Users\riccig01\Documents\vascbrain\OrionImages\data\raw"
    r"\FNEL03_CAD001_001\FNEL03_2026_V1_001703"
    r"\FNEL03_CAD001_001_FNEL03_2026_V1_001703.ome.tiff"
)


# ---------------------------------------------------------------------------
# Step 1: Extract
# ---------------------------------------------------------------------------

def run_extraction() -> Path:
    """Extract region using padded bbox from coordinates JSON."""
    with open(COORDS_JSON) as f:
        coords = json.load(f)

    bp = coords["bbox_padded_px"]
    bt = coords["bbox_tight_px"]
    ou = coords["origin_um"]
    ps = coords["pixel_size_um"]

    region = TissueRegion(
        region_id=0,
        bbox_ymin=bt["ymin"],
        bbox_xmin=bt["xmin"],
        bbox_ymax=bt["ymax"],
        bbox_xmax=bt["xmax"],
        padded_ymin=bp["ymin"],
        padded_xmin=bp["xmin"],
        padded_ymax=bp["ymax"],
        padded_xmax=bp["xmax"],
        area_px=coords["area_px"],
        area_um2=coords["area_um2"],
        centroid_y_px=coords["centroid_px"]["y"],
        centroid_x_px=coords["centroid_px"]["x"],
        origin_um_y=ou["y"],
        origin_um_x=ou["x"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STEP 1: Extraction")
    logger.info(f"  Image  : {IMAGE_PATH.name}")
    logger.info(
        f"  ROI    : y=[{bp['ymin']}, {bp['ymax']}]  "
        f"x=[{bp['xmin']}, {bp['xmax']}]"
    )
    logger.info(f"  Output : {OUTPUT_DIR}")
    logger.info("=" * 60)

    with OmeTiffReader(str(IMAGE_PATH)) as reader:
        metadata = extract_ome_metadata(str(IMAGE_PATH))
        logger.info(
            f"  Source pyramid levels: {reader.num_levels}  "
            f"channels: {reader.num_channels}"
        )
        save_region(
            region,
            reader,
            metadata,
            str(OUTPUT_DIR),
            output_formats=["ometiff"],
            verbose=True,
        )

    ometiff_path = OUTPUT_DIR / "region_000" / "region_000.ome.tiff"
    assert ometiff_path.exists(), f"Output file missing: {ometiff_path}"
    size_mb = ometiff_path.stat().st_size / (1024 ** 2)
    logger.info(f"  Written: {ometiff_path}  ({size_mb:.1f} MB)")
    return ometiff_path


# ---------------------------------------------------------------------------
# Step 2: Validate
# ---------------------------------------------------------------------------

def validate_pyramid(ometiff_path: Path) -> bool:
    """
    Inspect the IFD structure of the output OME-TIFF and verify that
    every channel at the full-resolution level has a SubIFD chain.

    Returns True if all checks pass.
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Pyramid Validation")
    logger.info(f"  File: {ometiff_path}")
    logger.info("=" * 60)

    passed = True

    with tifffile.TiffFile(str(ometiff_path)) as tf:
        series = tf.series[0]
        num_levels = len(series.levels)
        num_channels = len(series.levels[0].pages)
        expected_subifds = num_levels - 1

        logger.info(f"  Detected pyramid levels : {num_levels}")
        logger.info(f"  Detected channels       : {num_channels}")
        logger.info(f"  Expected SubIFDs / chan : {expected_subifds}")

        # --- Check 1: correct number of pyramid levels ---
        with open(COORDS_JSON) as f:
            coords = json.load(f)
        expected_levels = coords["num_pyramid_levels"]
        if num_levels != expected_levels:
            logger.error(
                f"  [FAIL] Level count mismatch: "
                f"got {num_levels}, expected {expected_levels}"
            )
            passed = False
        else:
            logger.info(f"  [PASS] Level count = {num_levels}")

        # --- Check 2: every full-res channel IFD has SubIFD pointers ---
        full_res_pages = series.levels[0].pages
        missing_subifds: list[int] = []
        wrong_count: list[tuple[int, int]] = []

        for ch_idx, page in enumerate(full_res_pages):
            subs = page.subifds  # None or tuple of byte offsets
            if subs is None or len(subs) == 0:
                missing_subifds.append(ch_idx)
            elif len(subs) != expected_subifds:
                wrong_count.append((ch_idx, len(subs)))

        if missing_subifds:
            logger.error(
                f"  [FAIL] Channels with NO SubIFDs "
                f"(cannot zoom in QuPath): {missing_subifds}"
            )
            passed = False
        else:
            logger.info(
                f"  [PASS] All {num_channels} channels have SubIFD pointers"
            )

        if wrong_count:
            for ch, cnt in wrong_count:
                logger.error(
                    f"  [FAIL] Channel {ch}: {cnt} SubIFDs, "
                    f"expected {expected_subifds}"
                )
            passed = False
        else:
            logger.info(
                f"  [PASS] All channels have exactly {expected_subifds} SubIFDs"
            )

        # --- Check 3: SubIFD pages are readable and have decreasing dimensions ---
        logger.info("  Checking SubIFD page readability and dimensions ...")
        prev_h, prev_w = None, None
        dim_errors: list[str] = []

        for level_idx, level in enumerate(series.levels):
            level_pages = level.pages
            page0 = level_pages[0]
            h, w = page0.shape[-2], page0.shape[-1]

            if level_idx == 0:
                prev_h, prev_w = h, w
                logger.info(f"    Level {level_idx}: {w}×{h} px (full-res)")
            else:
                if h >= prev_h or w >= prev_w:
                    dim_errors.append(
                        f"Level {level_idx} ({w}×{h}) is not smaller "
                        f"than level {level_idx-1} ({prev_w}×{prev_h})"
                    )
                else:
                    ds_y = prev_h / h
                    ds_x = prev_w / w
                    logger.info(
                        f"    Level {level_idx}: {w}×{h} px  "
                        f"(~{ds_y:.1f}× downsample)"
                    )
                prev_h, prev_w = h, w

        if dim_errors:
            for err in dim_errors:
                logger.error(f"  [FAIL] {err}")
            passed = False
        else:
            logger.info("  [PASS] Pyramid dimensions decrease monotonically")

        # --- Check 4: spot-read a tile from each level to confirm data integrity ---
        logger.info("  Spot-reading channel 0 tile from each level ...")
        read_errors: list[str] = []

        zarr_store = None
        try:
            zarr_store = zarr.open(series.aszarr(), mode="r")
        except Exception as e:
            logger.warning(f"  Could not open zarr store: {e}")

        if zarr_store is not None:
            for level_idx in range(num_levels):
                try:
                    arr = zarr_store[str(level_idx)]
                    # Read a small patch from channel 0
                    if arr.ndim == 3:
                        patch = np.asarray(arr[0, :min(64, arr.shape[1]), :min(64, arr.shape[2])])
                    else:
                        patch = np.asarray(arr[:min(64, arr.shape[0]), :min(64, arr.shape[1])])
                    if patch.size == 0:
                        read_errors.append(f"Level {level_idx}: empty patch")
                    elif np.all(patch == 0):
                        logger.warning(
                            f"    Level {level_idx}: patch is all zeros "
                            f"(may be normal for sparse tissue)"
                        )
                    else:
                        logger.info(
                            f"    Level {level_idx}: ok  "
                            f"shape={patch.shape}  "
                            f"range=[{patch.min()}, {patch.max()}]"
                        )
                except Exception as e:
                    read_errors.append(f"Level {level_idx}: {e}")

            if read_errors:
                for err in read_errors:
                    logger.error(f"  [FAIL] Read error — {err}")
                passed = False
            else:
                logger.info("  [PASS] All levels readable")

    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        ometiff_path = run_extraction()
        ok = validate_pyramid(ometiff_path)
    except Exception as exc:
        logger.exception(f"Script failed: {exc}")
        sys.exit(1)

    logger.info("=" * 60)
    if ok:
        logger.info("RESULT: ALL CHECKS PASSED — pyramid is QuPath-compatible")
    else:
        logger.error("RESULT: SOME CHECKS FAILED — see errors above")
    logger.info("=" * 60)

    sys.exit(0 if ok else 1)
