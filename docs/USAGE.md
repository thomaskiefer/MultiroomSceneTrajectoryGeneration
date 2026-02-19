# Usage Guide

## Environment setup

Recommended (`uv`):

```bash
uv sync --extra full
```

Alternative (`venv` + `pip`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[full]"
```

## Commands

### Generate trajectories

```bash
mrstg \
  --workflow structural_json \
  --scene-input-json examples/structural/demo_apartment.json \
  --output-dir outputs/demo
```

Optional:
- `--config-json <path>`: load full serialized config.
- `--project-root <path>`: explicit HL3D repo root for `houselayout3d_matterport`.
- `--log-level DEBUG|INFO|WARNING|ERROR|CRITICAL`.
- `--use-local-walkthrough` or `--use-legacy-walkthrough`.

### Minimal one-file flow (generate + visualize)

```bash
python scripts/minimal_generate_and_visualize.py \
  --scene-input-json examples/structural/demo_apartment.json \
  --output-dir outputs/minimal_demo
```

### Validate structural JSON schema only

```bash
mrstg \
  --validate-schema \
  --scene-input-json examples/structural/demo_apartment.json
```

### Preprocess HL3D + Matterport into canonical structural JSON

```bash
mrstg preprocess \
  --dataset-root /path/to/houselayout3d/data \
  --scene 2t7WUuJeko7 \
  --project-root /path/to/houselayout3d \
  --geojson-output outputs/preprocess/2t7WUuJeko7_connectivity.geojson \
  --structural-output outputs/preprocess/2t7WUuJeko7_structural_scene.json
```

`<scene>_structural_scene.json` includes `openings[]` (door/window metadata) when available and `stairs[]` from Matterport stair ranges.

Useful preprocess flags:
- `--geojson-only`: stop after connectivity GeoJSON.
- `--emit-debug-artifacts`: emit floor shards, plots, diagnostics.
- `--use-raw-room-centers`: debug plots use geometric centroids instead of trajectory center logic.

### Convert neutral structural-primitives JSON to canonical structural JSON

```bash
python scripts/convert_structural_primitives_to_scene_json.py \
  --input examples/structural/structural_template.json \
  --output outputs/preprocess/demo_structural_scene.json
```

Connection behavior:
- Uses `connections` if explicitly provided in input.
- Else derives from `openings` (nearest two rooms to each opening center).
- Else falls back to bbox-proximity connectivity.

### Trajectory visualization (image + video)

Quick no-dataset demo:

```bash
uv run python scripts/regenerate_samples.py

uv run mrstg-viz \
  --geojson samples/2t7WUuJeko7/2t7WUuJeko7_connectivity.geojson \
  --trajectory samples/2t7WUuJeko7/2t7WUuJeko7_floor_0_trajectory.json \
  --output-dir samples/2t7WUuJeko7/visualizations \
  --fps 30 \
  --speed 1.0 \
  --image
```

When you already have preprocess + trajectory outputs:

```bash
mrstg-viz \
  --geojson outputs/preprocess/2t7WUuJeko7_connectivity.geojson \
  --trajectory outputs/trajectory_generation/2t7WUuJeko7_floor_0_trajectory.json \
  --output-dir outputs/visualizations \
  --fps 30 \
  --speed 1.0
```

Notes:
- `--speed 1.0` means real-time playback.
- `--speed > 1.0` compresses trajectory time for shorter videos.
- GeoJSON and trajectory should come from the same scene/run context.

### Disconnected component behavior

Defaults:
- `walkthrough.behavior.disconnected_component_policy = "largest_component_only"`
  - traverses only the largest connected component
  - skips other rooms and emits warnings in summary output

Alternative:
- `walkthrough.behavior.disconnected_component_policy = "all_components"`
  - traverses each disconnected component as its own chunk
  - restarts trajectory per component
  - does not create cross-component bridge lines through walls

### Benchmark structural_json runtime

```bash
python scripts/benchmark_structural_json.py \
  --scene-input-json examples/structural/demo_apartment.json \
  --repeat 5 \
  --report-json outputs/bench/demo_apartment_benchmark.json
```

### Regenerate canonical sample bundle

```bash
uv run python scripts/regenerate_samples.py
```

Generated bundle:
- `samples/demo_apartment/`
- `samples/2t7WUuJeko7/`
- `samples/structural_template/`

## Programmatic API

```python
from pathlib import Path
from trajectory_generation import generate_from_structural_json

artifacts = generate_from_structural_json(
    scene_input_json=Path("examples/structural/demo_apartment.json"),
    output_dir=Path("outputs/demo"),
)
print(artifacts.to_dict())
```

## Running with uv

Prefix any command with `uv run`, for example:

```bash
uv run mrstg --workflow structural_json --scene-input-json examples/structural/demo_apartment.json --output-dir outputs/demo
uv run mrstg-viz --geojson samples/2t7WUuJeko7/2t7WUuJeko7_connectivity.geojson --trajectory samples/2t7WUuJeko7/2t7WUuJeko7_floor_0_trajectory.json --output-dir samples/2t7WUuJeko7/visualizations --fps 30 --speed 1.0
```
