"""
Diagnostic script to verify channel preservation in extracted OME-TIFF files.

Usage:
    python diagnose_channels.py <extracted_ometiff_path>

Example:
    python diagnose_channels.py extracted_tissues/test_pyramid_fix/region_000/region_000.ome.tiff
"""

from pathlib import Path
import sys
import json
import tifffile
from ome_types import from_tiff, from_xml

def diagnose_ome_tiff(filepath: str):
    """Analyze channel metadata in an OME-TIFF file."""

    filepath = Path(filepath)
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        sys.exit(1)

    print("=" * 70)
    print(f"Diagnosing OME-TIFF: {filepath.name}")
    print("=" * 70)

    try:
        # Method 1: Parse via ome_types (recommended)
        print("\n[1] Parsing via ome_types.from_tiff()...")
        ome = from_tiff(str(filepath))
        px = ome.images[0].pixels

        num_channels = px.size_c
        height = px.size_y
        width = px.size_x
        dtype = px.type

        print(f"    ✓ Image dimensions: {width}×{height} px")
        print(f"    ✓ Data type: {dtype}")
        print(f"    ✓ Number of channels: {num_channels}")
        print(f"    ✓ Pixel size: {px.physical_size_x} {px.physical_size_x_unit if px.physical_size_x_unit else 'um'} "
              f"× {px.physical_size_y} {px.physical_size_y_unit if px.physical_size_y_unit else 'um'}")

        # Extract channel names and metadata
        print("\n[2] Channel Information:")
        print("-" * 70)
        for i, ch in enumerate(px.channels):
            name = ch.name or f"Channel {i}"
            exc_wl = ch.excitation_wavelength
            em_wl = ch.emission_wavelength
            fluor = ch.fluor

            status = "✓"
            print(f"  {status} Channel {i:2d}: {name}")
            if exc_wl:
                print(f"              Excitation: {exc_wl} nm")
            if em_wl:
                print(f"              Emission:   {em_wl} nm")
            if fluor:
                print(f"              Fluorophore: {fluor}")

        print("-" * 70)

    except Exception as e:
        print(f"    ❌ Error parsing with ome_types: {e}")
        ome = None

    # Method 2: Check TIFF structure with tifffile
    print("\n[3] TIFF Structure Analysis (tifffile)...")
    try:
        with tifffile.TiffFile(str(filepath)) as tif:
            print(f"    ✓ Total IFDs: {len(tif.pages)}")

            # Check for series (multi-resolution pyramid)
            if hasattr(tif, 'series') and tif.series:
                series = tif.series[0]
                print(f"    ✓ Series detected: {len(series.levels)} pyramid levels")

                # Check full-resolution level for SubIFDs
                full_res_pages = series.levels[0].pages
                print(f"    ✓ Full-resolution pages: {len(full_res_pages)}")

                for page_idx, page in enumerate(full_res_pages[:3]):  # Show first 3
                    subifds = page.subifds
                    print(f"      Page {page_idx}: {len(subifds) if subifds else 0} SubIFDs")
                    if page_idx == 2:
                        print(f"      ... (showing first 3 pages)")
            else:
                print(f"    ⚠️  No series detected - may not be a proper pyramid")

            # Check for OME-XML tag (tag 270)
            if 270 in tif.pages[0].tags:
                xml_str = tif.pages[0].tags[270].value
                print(f"    ✓ OME-XML found (tag 270): {len(xml_str)} bytes")
            else:
                print(f"    ❌ OME-XML NOT found (tag 270 missing)")

    except Exception as e:
        print(f"    ❌ Error reading TIFF structure: {e}")

    # Method 3: Raw OME-XML extraction
    print("\n[4] Raw OME-XML Extraction...")
    try:
        with tifffile.TiffFile(str(filepath), is_ome=False) as tif:
            if 270 in tif.pages[0].tags:
                xml_str = tif.pages[0].tags[270].value

                # Parse and show first few lines
                lines = xml_str.split('\n')
                print("    First 10 lines of OME-XML:")
                for line in lines[:10]:
                    print(f"      {line}")

                # Count <Channel> elements
                channel_count = xml_str.count('<Channel')
                print(f"\n    ✓ Found {channel_count} <Channel> elements in OME-XML")

                # Try to parse for channel names
                try:
                    ome_from_xml = from_xml(xml_str)
                    px_from_xml = ome_from_xml.images[0].pixels

                    print(f"    ✓ Parsed {len(px_from_xml.channels)} channels from XML:")
                    for i, ch in enumerate(px_from_xml.channels):
                        name = ch.name or f"Channel {i}"
                        print(f"       [{i}] {name}")

                except Exception as e:
                    print(f"    ❌ Error parsing XML: {e}")
            else:
                print("    ❌ Tag 270 not found")

    except Exception as e:
        print(f"    ❌ Error: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    if ome and num_channels > 1:
        print(f"✓ SUCCESS: File contains {num_channels} channels with metadata")
        print(f"           Channel names are: {[ch.name or f'Channel {i}' for i, ch in enumerate(px.channels)]}")
        print("\nThis file should display all channels in QuPath.")
        print("If QuPath only shows 1 channel, the issue may be:")
        print("  - QuPath version compatibility with multi-channel pyramid TIFF")
        print("  - QuPath configuration/settings")
        print("  - Opening method (try: Automate → Project → Add images)")
    elif ome and num_channels == 1:
        print(f"⚠️  WARNING: File only has 1 channel")
        print("             Check if extraction filtered channels")
    else:
        print(f"❌ ERROR: Could not parse OME metadata")

    print("=" * 70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_channels.py <extracted_ometiff_path>")
        print("\nExample:")
        print("  python diagnose_channels.py extracted_tissues/test_pyramid_fix/region_000/region_000.ome.tiff")
        sys.exit(1)

    filepath = sys.argv[1]
    diagnose_ome_tiff(filepath)
