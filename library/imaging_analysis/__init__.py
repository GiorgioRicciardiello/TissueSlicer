"""
Imaging Analysis Module

Quantification and analysis pipelines built on library.imaging for spatial biology.

Currently includes:
- tissue_extractor: Extract individual tissue regions from multi-tissue slides
"""

from library.imaging_analysis.tissue_extractor import (
    TissueRegion,
    extract_tissue_regions,
)

__all__ = [
    "TissueRegion",
    "extract_tissue_regions",
]
