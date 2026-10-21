import json

import pytest

from app.storyboard.io import load_storyboard_file, parse_storyboard_json


def _storyboard_data():
    return {
        "project": {
            "title": "Engine explainer",
            "description": "A visual explanation of a car engine.",
            "duration_seconds": 5,
            "aspect_ratio": "16:9",
            "visual_style": "Clean instructional 3D animation",
        },
        "characters": [],
        "locations": [
            {
                "id": "LOC_001",
                "name": "Studio",
                "description": "A clean instructional studio.",
                "lighting": "Even studio lighting",
                "visual_style": "Clean 3D animation",
            }
        ],
        "props": [],
        "scenes": [
            {
                "id": "SCENE_001",
                "duration_seconds": 5,
                "location_id": "LOC_001",
                "character_ids": [],
                "prop_ids": [],
                "action": "The engine cover opens.",
                "visual_description": "The engine is shown with its cover open.",
                "camera": {
                    "shot_type": "Medium shot",
                    "angle": "Side view",
                    "movement": "Steady",
                },
                "continuity": {
                    "previous_scene_id": None,
                    "character_state": "The engine cover is open.",
                    "environment_state": "The studio is evenly lit.",
                    "required_start_state": "The engine cover is closed.",
                    "ending_state": "The engine cover is open.",
                },
            }
        ],
    }


def test_parse_storyboard_json_returns_validated_storyboard_without_rewriting():
    data = _storyboard_data()

    storyboard = parse_storyboard_json(json.dumps(data))

    assert storyboard.project.title == "Engine explainer"
    assert storyboard.scenes[0].duration_seconds == 5
    assert storyboard.scenes[0].character_ids == []


def test_parse_storyboard_json_reports_invalid_json_and_top_level_type():
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_storyboard_json("{invalid")
    with pytest.raises(ValueError, match="top-level object"):
        parse_storyboard_json("[]")


def test_parse_storyboard_json_validates_scene_duration_sum():
    data = _storyboard_data()
    data["project"]["duration_seconds"] = 10

    with pytest.raises(ValueError, match="add up exactly"):
        parse_storyboard_json(json.dumps(data))


def test_load_storyboard_file_supports_utf8_bom(tmp_path):
    path = tmp_path / "storyboard.json"
    path.write_text(
        "\ufeff" + json.dumps(_storyboard_data()),
        encoding="utf-8",
    )

    storyboard = load_storyboard_file(path)

    assert storyboard.project.duration_seconds == 5
