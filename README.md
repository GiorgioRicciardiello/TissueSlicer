# TissueSlicer 🔬

**Interactive web GUI for extracting tissue regions from large multiplex fluorescence microscope images.**

## The Problem

Orion multiplex fluorescence microscopy produces massive 30–100 GB OME-TIFF images with 20+ channels and full pyramid levels. Researchers need to manually identify and extract regions of interest (tissue regions, structures, landmarks) for downstream analysis, but:

- **Too large to open in desktop software** — files exceed available RAM
- **Hard to see details** — downsampled preview images lack proper contrast calibration
- **Tedious extraction** — manual region selection and coordinate mapping is error-prone
- **Need multi-channel view** — tissue features span multiple channels with different colors

## The Solution

TissueSlicer provides an **interactive web interface** to:

1. **View large images efficiently** — displays pyramid-level previews at ~4K resolution
2. **See tissue details clearly** — auto-calibrated brightness/contrast per channel
3. **Select regions intuitively** — draw polygons or rectangles over the preview
4. **View multiple channels at once** — QuPath-style multi-channel composite with per-channel colors
5. **Extract full-resolution data** — backend reads raw uint16 data at full resolution with pyramid levels preserved

## What It Does

### Input
- **OME-TIFF file path**: Path to a large multiplex fluorescence image (30–100 GB)
  - Example: `C:\path\to\FNEL03_CAD001_001.ome.tiff`
  - Format: 20 channels, uint16 pixel values (0–4095), up to 65,000 × 37,000 pixels

### Workflow
1. **Load image** → Web GUI displays downsampled preview (~4100×2315 pixels)
2. **Select channels** → Enable/disable channels, adjust colors, auto-calibrate brightness
3. **Draw region** → Click vertices to draw a polygon over the tissue region of interest
4. **Adjust contrast** → Per-channel min/max intensity sliders for clear visibility
5. **Save selection** → Store polygon with metadata
6. **Batch extract** → Backend reads all channels at full resolution, applies pyramid levels, writes output

### Output
For each extracted region:
```
<output_dir>/<region_name>/
├── region_000.h5              # HDF5: all 20 channels, 8 pyramid levels, uint16
├── region_000.ome.tiff        # OME-TIFF: same data with OME-XML metadata
├── region_000_thumbnail.png   # QC thumbnail (Hoechst channel)
└── region_000_coordinates.json # Back-correction mapping to source image
```

## Quick Start

```bash
# Install dependencies
conda env create -f environment.yml

# Run the GUI
python run_gui.py
```

Then open **http://localhost:5000** in your browser.

---

## Demo
See demo video in `static_git/TissueSlicerShorter720.mp4`

<div align="center">
  <video width="720" height="480" controls>
    <source src="static_git/TissueSlicerShorter720.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

---

## Key Features

### 🎨 Multi-Channel Visualization
- Select any subset of 20 channels simultaneously
- Each channel gets a distinct color (cyan, yellow, magenta, red, green, etc.)
- Additive blending shows tissue structure across channels
- Per-channel brightness/contrast control

### ⚡ Fast & Memory-Efficient
- Displays pyramid-level previews (~4K resolution) instead of full images
- Display-level cache avoids repeated disk reads
- Backend reads only requested ROI at full resolution
- Peak memory: ~200 MB per channel (streaming, never full stack in RAM)

### 🛠️ Drawing Tools
- **Polygon tool**: Click vertices to draw arbitrary shapes
- **Rectangle tool**: Click-drag for axis-aligned regions
- **Keyboard shortcuts**: Esc (undo), Enter (close), Delete (undo last vertex)
- **Real-time preview**: Mask overlay shows extraction boundaries with padding

### 📊 Auto-Calibration
- **Percentile stretching** (2nd–98th percentile): Industry-standard for scientific imaging
- **Per-channel calibration**: Each channel's display range optimized automatically
- **Interactive sliders**: Fine-tune contrast on the fly, 300 ms debounce for responsiveness

### 💾 Batch Processing
- Extract all selected regions in parallel
- Progress tracking with per-region and overall completion status
- Pyramid levels preserved (8 levels: 16×, 8×, 4×, 2×, 1×, 0.5×, 0.25×, 0.125×)
- Coordinates back-corrected to source image

---

## Architecture

```
Frontend (Browser)          Backend (Flask)             Storage
├── Canvas drawing          ├── Image loader            ├── OME-TIFF input
├── Multi-channel UI        ├── Polygon ops             ├── HDF5 output
├── Brightness/contrast     ├── Coordinate mapping      └── OME-TIFF output
└── Region management       └── Async extraction
```

---

## Tech Stack

- **Frontend**: Vanilla JavaScript, Canvas API, HTML5
- **Backend**: Flask, NumPy, OpenCV, tifffile, zarr
- **Image Format**: OME-TIFF with Zarr pyramid levels
- **Output**: HDF5 + OME-TIFF with full metadata preservation

---

## Requirements

- **Python 3.8+**
- **System RAM**: 8 GB minimum (16+ GB recommended)
- **Storage**: ~2× input file size (for HDF5 + OME-TIFF output)

---

## Documentation

- **[INSTALL.md](INSTALL.md)** — Detailed installation for Windows/macOS/Linux
- **[VISUALIZATION_PLAN.md](VISUALIZATION_PLAN.md)** — Multi-channel visualization architecture
- **[CLAUDE.md](CLAUDE.md)** — Development guidelines and constraints

---

## Example Workflow

```
1. Load image
   Input: /data/raw/FNEL03_CAD001_001.ome.tiff

2. Enable channels
   - Channel 0 (Hoechst, cyan) — nuclear marker
   - Channel 3 (Ki-67, red) — proliferation marker
   - Channel 16 (CD31, green) — endothelial marker

3. Auto-calibrate all channels
   p2/p98 percentiles computed per channel

4. Draw polygon over tissue region
   5 clicks to define region of interest

5. Save selection → Extract
   Backend reads full-resolution data for all 3 enabled channels
   Writes HDF5 + OME-TIFF with pyramid levels

6. Output
   region_000/region_000.h5 (20 channels, 8 pyramid levels, uint16)
```

---

## License

TissueSlicer is developed at Mount Sinai School of Medicine for research purposes.

---

**Built with ❤️ for spatial biology research**
