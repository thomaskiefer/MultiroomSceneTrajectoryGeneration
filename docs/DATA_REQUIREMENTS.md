# Data Requirements

## 1) Canonical input (recommended): `structural_json`

This workflow requires exactly one JSON file with schema `scene.schema.v1`.

Required:
- `scene`: string
- `floors[]`: `floor_index`, `z`, `footprint_xy`
- `rooms[]`: `room_id`, `semantic`, `bbox.min/max`
- `schema_version`: `scene.schema.v1`

Optional:
- `rooms[].floor_index` (auto-assigned by nearest floor `z` if omitted)
- `rooms[].polygon_xy` (requires `shapely`)
- `connections[]` with optional `waypoint_xy` and `normal_xy`
- `openings[]` (`opening_type` + `waypoint_xy`/`bbox`, optional `normal_xy`)
- `stairs[]` (`z_min`, `z_max`, optional nearest-floor indices)

Reference examples:
- `examples/structural/demo_apartment.json`
- `examples/structural/2t7WUuJeko7.json`
- `samples/2t7WUuJeko7/2t7WUuJeko7_connectivity.geojson` (generated visualization fixture)
- Machine-readable schema:
  - `docs/schema/scene.schema.v1.json`

## 2) Dataset-specific input: `houselayout3d_matterport`

Expected upstream assets:
- HouseLayout3D geometry under:
  - `<dataset_root>/structures/layouts_split_by_entity/<scene>/...`
- Matterport room annotations under:
  - `<house_segmentation_dir>/<scene>/<scene>.house`
  - defaults to `<dataset_root>/house_segmentations/...` when not set.
- Optional openings:
  - `<dataset_root>/doors/<scene>.json`
  - `<dataset_root>/windows/<scene>.json`

This workflow is intended to mirror the original HL3D preprocessing flow and produce:
- connectivity GeoJSON
- canonical structural JSON for portable downstream generation
  - includes `openings[]` (doors/windows) when available from HL3D opening data
  - includes `stairs[]` from Matterport stair height ranges

## 3) Neutral structural-primitives template input

If your pipeline already has structural corner points / bboxes, use:
- `scripts/convert_structural_primitives_to_scene_json.py`
- Example template: `examples/structural/structural_template.json`

Expected template fields:
- `scene`: string
- `floors[]`: `floor_index`, `z`, `footprint_xy`
- `rooms[]`: `room_id`, `bbox.min/max`
- optional `rooms[].semantic`, `rooms[].polygon_xy`, `rooms[].floor_index`
- optional `connections[]`
- optional `openings[]` with `waypoint_xy` or `bbox` (+ optional `normal_xy`)

Connection generation priority:
1. explicit `connections`
2. inferred from `openings`
3. inferred from room bbox proximity

## Synthetic vs observed passages

Connectivity can include two passage classes:
- `observed`: matched to detected door openings.
- `inferred` (synthetic): geometry/proximity-derived connection when no reliable door mesh was matched.

Inferred passages are expected in imperfect scans and are explicitly represented so traversal remains connected.

## GeoJSON center annotations

HL3D preprocess GeoJSON exports include trajectory-center annotations computed with the same
center logic used by local trajectory generation:
- room feature properties:
  - `trajectory_center_xy`
  - `trajectory_center_3d`
- additional point features:
  - `properties.type == "trajectory_room_center"`

## Unit conventions

- Distances and coordinates: meters
- Z axis: up
- Room/floor geometry and trajectory frames use the same world coordinate convention.
