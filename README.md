# Multiroom Scene Trajectory Generation

Config-driven multi-room camera trajectory generation and preprocessing.

This repository is a standalone extraction of the trajectory-generation stack so it can be shared and reused independently from HouseLayout3D/GVS code.

## Workflow Choice
- `structural_json` (recommended): portable, shareable workflow from one scene JSON.
- `houselayout3d_matterport`: project-internal workflow using HouseLayout3D geometry and Matterport `.house` annotations.

## Quick Start

### 1) Install

Option A (`uv`, recommended):

```bash
uv sync --extra full
```

Developer install (lint + typecheck tooling):

```bash
uv sync --extra full --group dev
```

Option B (`venv` + `pip`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[full]"
```

Requirements:
- Python `>=3.10`
- `shapely` is only required when using `rooms[].polygon_xy` (included by `.[geometry]`).

### 1b) Developer bootstrap (ruff + mypy)

Use the bootstrap script to set up a full dev environment:

```bash
# installs project deps + dev tools (ruff/mypy/jsonschema)
scripts/dev_setup.sh

# install + run lint/type/test checks
scripts/dev_setup.sh --check
```

### 2) Run the bundled example

```bash
uv run mrstg \
  --workflow structural_json \
  --scene-input-json examples/structural/demo_apartment.json \
  --output-dir outputs/demo \
  --log-level INFO

# Validate schema only (no generation)
uv run mrstg --validate-schema --scene-input-json examples/structural/demo_apartment.json
```

Note: `demo_apartment.json` intentionally contains a disconnected `balcony` room to demonstrate disconnected-component warnings.

With `pip`/`venv` install:

```bash
mrstg \
  --workflow structural_json \
  --scene-input-json examples/structural/demo_apartment.json \
  --output-dir outputs/demo \
  --log-level INFO

# Validate schema only (no generation)
mrstg --validate-schema --scene-input-json examples/structural/demo_apartment.json
```

### 2b) Convert a floorplan connectivity GeoJSON into structural JSON

```bash
python scripts/convert_connectivity_geojson_to_structural_json.py \
  --geojson /path/to/<scene>_connectivity.geojson \
  --output outputs/preprocess/<scene>_structural_scene.json \
  --scene <scene_id>
```

Repository includes a real converted sample:
- `examples/structural/matterport_2t7WUuJeko7.json`

### 2c) Convert neutral structural primitives into structural JSON

Use this when you already have floor footprints, room boxes/polygons, and optional openings.

```bash
python scripts/convert_structural_primitives_to_scene_json.py \
  --input examples/structural/structural_template.json \
  --output outputs/preprocess/structural_template_scene.json
```

Template input example:
- `examples/structural/structural_template.json`

### 2d) Dataset-specific preprocessing (HouseLayout3D + Matterport -> GeoJSON -> structural JSON)

```bash
mrstg preprocess \
  --dataset-root /path/to/houselayout3d/data \
  --scene 2t7WUuJeko7 \
  --project-root /path/to/houselayout3d \
  --geojson-output outputs/preprocess/2t7WUuJeko7_connectivity.geojson \
  --structural-output outputs/preprocess/2t7WUuJeko7_structural_scene.json \
  --log-level INFO

# Also emit intermediate debug artifacts (per-floor geojson + plots with
# rooms, doors/windows, and connectivity overlays), plus diagnostics json/csv
mrstg preprocess \
  --dataset-root /path/to/houselayout3d/data \
  --scene 2t7WUuJeko7 \
  --project-root /path/to/houselayout3d \
  --emit-debug-artifacts

# By default, debug connectivity plots use the same room-center logic as
# trajectory generation; use --use-raw-room-centers to disable this.

# Optional: export only connectivity GeoJSON
mrstg preprocess \
  --dataset-root /path/to/houselayout3d/data \
  --scene 2t7WUuJeko7 \
  --project-root /path/to/houselayout3d \
  --geojson-only

# Legacy wrapper script still works (delegates to `mrstg preprocess`)
python scripts/preprocess_hl3d_matterport_to_structural_json.py ...
```

### 2e) Trajectory visualization (image + video)

For quick validation without dataset preprocessing, first generate the bundled sample artifacts:

```bash
uv run python scripts/regenerate_samples.py

uv run mrstg-viz \
  --geojson samples/matterport_2t7WUuJeko7/matterport_2t7WUuJeko7_connectivity.geojson \
  --trajectory samples/matterport_2t7WUuJeko7/matterport_2t7WUuJeko7_floor_0_trajectory.json \
  --output-dir samples/matterport_2t7WUuJeko7/visualizations \
  --fps 30 \
  --speed 1.0 \
  --image
```

If you already ran `mrstg preprocess` and have matching preprocess outputs:

```bash
mrstg-viz \
  --geojson outputs/preprocess/2t7WUuJeko7_connectivity.geojson \
  --trajectory outputs/trajectory_generation/2t7WUuJeko7_floor_0_trajectory.json \
  --output-dir outputs/visualizations \
  --fps 30 \
  --speed 1.0
```

### 2f) Regenerate canonical sample artifacts

```bash
uv run python scripts/regenerate_samples.py
```

Outputs are written to:
- `samples/demo_apartment/`
- `samples/matterport_2t7WUuJeko7/`
- `samples/structural_template/`

Debug connectivity legend semantics:
- `Observed door passage (actual opening)`: matched to a detected door opening.
- `Inferred passage (no detected door mesh)`: geometry-derived passable connection without a reliable detected door.
- Inferred passages are shown as subtle dashed lines with hollow circle waypoints to keep the view readable while preserving distinction.
- Debug artifact bundle also includes:
  - `<scene>_preprocess_diagnostics.json` (unmatched rooms/openings and counts)
  - `<scene>_room_stats.csv` (room matching stats; when export module supports CSV write)

If running from source without installation, use:

```bash
PYTHONPATH=src python -m trajectory_generation ...
```

### 3) Programmatic API

```python
from pathlib import Path
from trajectory_generation import generate_from_structural_json

artifacts = generate_from_structural_json(
    scene_input_json=Path("examples/structural/demo_apartment.json"),
    output_dir=Path("outputs/trajectory_generation"),
)
print(artifacts.to_dict())
```

## Canonical Structural JSON Schema

Required top-level keys:
- `schema_version`: currently `scene.schema.v1`
- `scene`: string
- `floors`: list of:
  - `floor_index`: int
  - `z`: float
  - `footprint_xy`: list of `[x, y]` with at least 3 points
- `rooms`: list of:
  - `room_id`: string
  - `semantic`: string
  - `bbox`: `{ "min": [x, y, z], "max": [x, y, z] }`

Optional keys:
- `rooms[].floor_index`: int (if omitted, assigned by nearest floor z)
- `rooms[].polygon_xy`: list of `[x, y]` room polygon points
- `connections`: list of:
  - `room1_id`, `room2_id`
  - optional `waypoint_xy`: `[x, y]`
  - optional `normal_xy`: `[nx, ny]`
- `openings`: passthrough (unused by first version)

Note: `rooms[].polygon_xy` requires `shapely` (`pip install -e ".[geometry]"`).
If `schema_version` is omitted, parser assumes `scene.schema.v1` and emits a warning.

## Output Frame Schema

Each trajectory file is JSON with frames like:

```json
{
  "id": 42,
  "position": [1.2, 0.5, 1.6],
  "look_at": [2.0, 0.5, 1.6],
  "forward": [1.0, 0.0, 0.0],
  "up": [0.0, 0.0, 1.0],
  "fov": 60.0
}
```

Conventions:
- World coordinates are meters.
- Z axis is up.

## Disconnected Components

`walkthrough.behavior.disconnected_component_policy` controls component coverage:
- `largest_component_only` (default): only traverses the largest connected component, skips other rooms, and emits clear warnings.
- `all_components`: traverses all components as independent trajectory chunks and restarts at each disconnected component (no cross-component wall traversal).

`walkthrough.behavior.disconnected_transition_mode`:
- `bridge` / `jump` is only relevant when there is a room-to-room step without a usable connection object inside a traversed component.
- With `all_components`, disconnected components are not bridged; trajectory restarts per component.

Typical warning examples in summaries:
- `[floor 0] skipped 1 room(s) outside largest connected component: balcony`
- `[floor 0] graph has 2 disconnected components; trajectory restarts per component (no cross-component links).`

Frame schema remains unchanged.

## Package Layout

- `src/trajectory_generation/config.py`: workflow + behavior config models
- `src/trajectory_generation/room_graph.py`: explicit shared room graph contract
- `src/trajectory_generation/walkthrough_local.py`: local path generation core
- `src/trajectory_generation/spline.py`: reusable Catmull-Rom spline primitives
- `src/trajectory_generation/adapters/houselayout3d_matterport.py`: HL3D/Matterport adapter
- `src/trajectory_generation/adapters/structural_json.py`: canonical structural JSON adapter
- `src/trajectory_generation/preprocess.py`: compatibility import layer
- `src/trajectory_generation/hl3d_preprocess.py`: dataset-specific HL3D/Matterport preprocessing
- `src/trajectory_generation/geojson_converter.py`: GeoJSON -> structural JSON conversion helpers
- `src/trajectory_generation/validation.py`: post-generation frame validation checks
- `src/trajectory_generation/debug_visualization.py`: modular debug floorplan renderer
- `src/trajectory_generation/artifacts.py`: shared output artifacts + writers
- `src/trajectory_generation/api.py`: minimal public programmatic API
- `src/trajectory_generation/cli.py`: CLI wrapper (`mrstg generate` / `mrstg preprocess`)
- `src/trajectory_generation/pipeline.py`: workflow dispatch + `register_runner(...)`

## Scope

- Core package: preprocessing adapters + connectivity + trajectory generation + visualization
- No GVS or Blender integration code
- Two supported workflows:
  - `houselayout3d_matterport`
  - `structural_json`

## Troubleshooting

- `ModuleNotFoundError: floorplan`:
  this only affects `houselayout3d_matterport`; use `structural_json` unless you have the full HouseLayout3D environment.
- `polygon_xy requires shapely`:
  install optional geometry deps with `pip install -e ".[geometry]"`.
- No floors/rooms or empty output:
  verify room `bbox` z-values are consistent with floor `z` values and that room IDs/connections are valid.

## Tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Type checking:

```bash
uv run mypy src
```

Or in the original conda env:

```bash
conda run -n houselayout3d python -m unittest discover -s tests -p 'test_*.py' -v
```

## Additional Docs

- `docs/USAGE.md`: command/API examples and visualization usage.
- `docs/DATA_REQUIREMENTS.md`: expected input structure for both workflows.
- `docs/ARCHITECTURE.md`: module-level architecture.
- `docs/schema/scene.schema.v1.json`: machine-readable structural scene schema.
- `examples/structural/`: canonical structural JSON input fixtures.
- `samples/`: generated connectivity/trajectory/visualization artifacts.

Runtime benchmark helper:
- `scripts/benchmark_structural_json.py`
- `scripts/regenerate_samples.py`
