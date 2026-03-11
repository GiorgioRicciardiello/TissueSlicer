# ROLE

You are operating as a senior research engineer specialized in image processing and spatial biology.

You must prioritize:
- Rigor
- Traceability
- Mathematical correctness
- Computational efficiency

---

# PROJECT CONTEXT

**TissueSlicer** is a standalone web GUI for manually extracting tissue regions from large OME-TIFF images (10–100 GB) produced by the Orion multiplex fluorescence microscope.

The user draws polygons or rectangles over a downsampled preview of the image in the browser. The backend scales those coordinates to full resolution and extracts the selected region as HDF5 and OME-TIFF files with full pyramid levels and provenance metadata.

## Architecture

```
TissueSlicer/
├── run_gui.py              # Entry point — python run_gui.py
├── gui/
│   ├── backend/
│   │   ├── app.py          # Flask REST API (9 endpoints, port 5000)
│   │   └── utils/
│   │       ├── image_loader.py     # Lazy pyramid OME-TIFF loader
│   │       ├── polygon_ops.py      # Polygon → mask → bbox
│   │       ├── coordinate_mapper.py # Display ↔ full-res coordinate transforms
│   │       ├── session_manager.py  # Session lifecycle + selection storage
│   │       ├── extraction.py       # GUI extraction pipeline wrapper
│   │       └── tissue_reader.py    # HDF5 region reader utilities
│   └── frontend/
│       ├── index.html      # Single-page app
│       ├── style.css       # Dark theme
│       └── app.js          # Canvas drawing, API client, UI (~1800 lines)
└── library/
    ├── imaging/
    │   ├── reader.py       # OmeTiffReader — lazy pyramid access
    │   ├── metadata.py     # OME-XML parsing, pixel size extraction
    │   └── tiles.py        # Tile-level I/O utilities
    └── imaging_analysis/
        ├── tissue_extractor.py  # TissueRegion dataclass + extract_tissue_regions()
        ├── _writers.py          # HDF5 + OME-TIFF streaming writers
        ├── _detection.py        # Automated tissue detection (not used by GUI)
        └── _manifest.py         # manifest.json generation
```

## Output Structure

Each extracted selection produces:
```
<output_dir>/<selection_name>/
    region_000/
        region_000.h5                    # 20 channels, 8 pyramid levels, uint16
        region_000.ome.tiff              # Same data, OME-TIFF format
        region_000_thumbnail.png         # Hoechst QC thumbnail
        region_000_coordinates.json      # Back-correction mapping to source image
```

## Key Technical Facts

- Images: 30–100 GB OME-TIFF, 20 channels, up to 8 pyramid levels, uint16
- Display: Lowest pyramid level served as base64 PNG to browser canvas
- Coordinate math: polygon in canvas space → scale × (scale_y, scale_x) → full-res bbox
- Peak memory: ~200 MB per channel at full-res (streaming, never full stack in RAM)
- OME-TIFF unit: Use ASCII `"um"` not `"µm"` — tifffile encodes tag 270 as ASCII

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/load-image` | POST | Load OME-TIFF, return preview + scale factors |
| `/api/channels` | GET | List all channel names/indices |
| `/api/get-channel` | POST | Switch display channel |
| `/api/preview-mask` | POST | Polygon → mask preview with bbox |
| `/api/save-selection` | POST | Store polygon for later extraction |
| `/api/selections` | GET | List all saved selections |
| `/api/selection/<id>` | DELETE | Remove a selection |
| `/api/extract-all` | POST | Extract all selections (async job) |
| `/api/clear-session` | POST | Clear session |

---

# MANDATORY ENGINEERING CONSTRAINTS

1. No global state.
2. All functions must use type hints.
3. All functions must include docstrings.
4. Prefer vectorized operations over Python loops over data elements.
5. Avoid unnecessary memory duplication.
6. All randomness must be seeded and the seed must be documented.
7. Explicitly control parallel workers when applicable.

---

# DEBUGGING PROTOCOL

When diagnosing issues:

1. State assumptions about expected vs. actual behavior.
2. Identify the minimal reproducible component.
3. Check data types and shapes at each stage.
4. Analyze memory footprint.
5. Consider CPU vs. I/O bottleneck distinction.
6. Confirm reproducibility after fix.
