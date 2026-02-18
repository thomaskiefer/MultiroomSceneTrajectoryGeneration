from __future__ import annotations

import json
from pathlib import Path
import unittest
import sys


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from jsonschema import ValidationError, validate

    _HAS_JSONSCHEMA = True
except Exception:
    _HAS_JSONSCHEMA = False


@unittest.skipUnless(_HAS_JSONSCHEMA, "jsonschema package is required for schema-contract tests")
class JsonSchemaContractTest(unittest.TestCase):
    def _load_schema(self) -> dict:
        repo_root = Path(__file__).resolve().parents[1]
        schema_path = repo_root / "docs" / "schema" / "scene.schema.v1.json"
        self.assertTrue(schema_path.exists(), f"Missing schema file: {schema_path}")
        return json.loads(schema_path.read_text())

    def test_minimal_example_validates_against_json_schema(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        scene_path = repo_root / "examples" / "structural" / "demo_apartment.json"
        payload = json.loads(scene_path.read_text())
        schema = self._load_schema()
        validate(instance=payload, schema=schema)

    def test_invalid_payload_fails_schema_validation(self) -> None:
        schema = self._load_schema()
        invalid = {
            "schema_version": "scene.schema.v1",
            "scene": "invalid",
            "floors": [],
            "rooms": [],
        }
        with self.assertRaises(ValidationError):
            validate(instance=invalid, schema=schema)

    def test_schema_version_const_is_enforced(self) -> None:
        schema = self._load_schema()
        invalid = {
            "schema_version": "scene.schema.v999",
            "scene": "invalid",
            "floors": [{"floor_index": 0, "z": 0.0, "footprint_xy": [[0, 0], [1, 0], [1, 1]]}],
            "rooms": [
                {
                    "room_id": "r1",
                    "semantic": "entryway",
                    "bbox": {"min": [0, 0, 0], "max": [1, 1, 2]},
                }
            ],
        }
        with self.assertRaises(ValidationError):
            validate(instance=invalid, schema=schema)


if __name__ == "__main__":
    unittest.main()
