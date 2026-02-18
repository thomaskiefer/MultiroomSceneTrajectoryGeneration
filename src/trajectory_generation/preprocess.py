"""Compatibility layer for preprocessing helpers.

This module keeps the historic import path stable while implementation lives in
`geojson_converter.py` and `hl3d_preprocess.py`.
"""

from .geojson_converter import (
    convert_connectivity_geojson_file,
    convert_connectivity_geojson_payload,
)
from .hl3d_preprocess import (
    Hl3dDebugArtifacts,
    Hl3dPreprocessContext,
    build_hl3d_matterport_connectivity_geojson,
    export_hl3d_matterport_debug_artifacts,
    preprocess_hl3d_matterport_to_structural_json,
)
from .structural_primitives_converter import (
    convert_structural_primitives_file,
    convert_structural_primitives_payload,
)

__all__ = [
    "Hl3dDebugArtifacts",
    "Hl3dPreprocessContext",
    "build_hl3d_matterport_connectivity_geojson",
    "convert_connectivity_geojson_file",
    "convert_connectivity_geojson_payload",
    "convert_structural_primitives_file",
    "convert_structural_primitives_payload",
    "export_hl3d_matterport_debug_artifacts",
    "preprocess_hl3d_matterport_to_structural_json",
]
