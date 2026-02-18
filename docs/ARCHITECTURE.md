# Architecture

## Core Pipeline
`Scene Input -> RoomGraph -> Control Points -> Spline+Timing -> Frames -> Artifacts`

## Modules
- `config.py`: workflow and behavior configuration contracts
- `cli.py`: command surface (`mrstg generate` and `mrstg preprocess`)
- `adapters/`: input-source translation into internal room graph contracts
- `preprocess.py`: compatibility import layer
- `hl3d_preprocess.py`: dataset-specific preprocessing/orchestration
- `geojson_converter.py`: connectivity GeoJSON -> canonical structural JSON conversion
- `debug_visualization.py`: preprocessing debug plot renderer (rooms/openings/connectivity/legend)
- `room_graph.py`: canonical graph primitives
- `walkthrough_local.py`: trajectory logic (center selection, traversal application, doorway crossing, disconnected-component policy)
- `spline.py`: Catmull-Rom interpolation
- `traversal.py`: room visit ordering
- `artifacts.py`: output metadata and serialization
- `validation.py`: post-generation frame checks and warnings
- `visualization.py`: static and video visualization

## Workflows
- `houselayout3d_matterport`: dataset-specific source adapter
- `structural_json`: portable canonical source adapter

## Preprocess Artifacts
- Connectivity GeoJSON (`<scene>_connectivity.geojson`)
- Optional floor GeoJSON shards (`<scene>_floor_<k>.geojson`)
- Optional debug plots (`<scene>_floorplan.png`, `<scene>_floor_<k>.png`)
- Optional diagnostics (`<scene>_preprocess_diagnostics.json`, `<scene>_room_stats.csv`)

## Design Rules
- Keep workflow-specific code inside adapters
- Keep trajectory logic workflow-agnostic
- Preserve output frame schema stability
- Add new behavior via config knobs, not hardcoded branches
