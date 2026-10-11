from app.storyboard.generator import (
    normalize_scene_durations,
    repair_storyboard_continuity,
    validate_storyboard_continuity,
)
from app.storyboard.models import Storyboard
import pytest


def test_fifteen_second_storyboard_uses_exact_scene_durations():
    data = {
        "project": {"duration_seconds": 15},
        "scenes": [{"duration_seconds": 5}, {"duration_seconds": 5}],
    }

    normalize_scene_durations(data)

    assert [scene["duration_seconds"] for scene in data["scenes"]] == [10, 5]


def test_non_multiple_duration_is_distributed_exactly():
    data = {
        "project": {"duration_seconds": 12},
        "scenes": [{"duration_seconds": 5}, {"duration_seconds": 5}],
    }

    normalize_scene_durations(data)

    assert [scene["duration_seconds"] for scene in data["scenes"]] == [10, 2]


def test_long_project_requires_enough_scenes():
    data = {
        "project": {"duration_seconds": 21},
        "scenes": [{"duration_seconds": 5}, {"duration_seconds": 5}],
    }

    with pytest.raises(ValueError, match="at least 3 scenes"):
        normalize_scene_durations(data)


def test_any_project_runtime_is_split_into_supported_scene_durations():
    data = {
        "project": {"duration_seconds": 38},
        "scenes": [{"duration_seconds": 5} for _ in range(4)],
    }

    normalize_scene_durations(data)

    assert [scene["duration_seconds"] for scene in data["scenes"]] == [
        10,
        10,
        10,
        8,
    ]


def test_storyboard_model():

    storyboard = Storyboard(
        project={
            "title": "Coffee Shop",
            "description": "A man enters a coffee shop.",
            "duration_seconds": 10,
            "aspect_ratio": "16:9",
            "visual_style": "Cinematic realism",
        },
        characters=[
            {
                "id": "CHAR_001",
                "name": "Daniel",
                "appearance": {
                    "age": 28,
                    "hair": "Short dark brown hair",
                    "clothing": "Blue denim jacket over a white shirt",
                    "facial_features": "Clean-shaven, oval face",
                    "physical_description": "Average height and build",
                },
            }
        ],
        locations=[
            {
                "id": "LOC_001",
                "name": "Modern Coffee Shop",
                "description": "A small modern coffee shop with wooden tables.",
                "lighting": "Warm afternoon sunlight through large windows",
                "visual_style": "Cinematic realistic",
            }
        ],
        props=[
            {
                "id": "PROP_001",
                "name": "Coffee Cup",
                "description": "A white ceramic coffee cup",
            }
        ],
        scenes=[
            {
                "id": "SCENE_001",
                "duration_seconds": 5,
                "location_id": "LOC_001",
                "character_ids": ["CHAR_001"],
                "prop_ids": [],
                "action": "Daniel walks into the coffee shop and approaches the counter.",
                "dialogue": None,
                "visual_description": (
                    "Daniel enters through the glass door and walks toward "
                    "the wooden counter."
                ),
                "camera": {
                    "shot_type": "Medium tracking shot",
                    "angle": "Eye level",
                    "movement": "Camera tracks backward as Daniel walks forward",
                    "lens": "35mm",
                },
                "continuity": {
                    "previous_scene_id": None,
                    "character_state": "Standing and walking toward the counter",
                    "environment_state": "Coffee shop is open and active",
                    "required_start_state": "Daniel is outside the coffee shop",
                    "ending_state": "Daniel is standing at the counter",
                },
            }
        ],
    )

    assert storyboard.project.title == "Coffee Shop"
    assert len(storyboard.scenes) == 1
    assert storyboard.scenes[0].id == "SCENE_001"


def test_symbolic_required_start_state_is_repaired():
    data = {
        "scenes": [
            {
                "id": "SCENE_001",
                "continuity": {
                    "previous_scene_id": "wrong",
                    "required_start_state": "outside",
                    "ending_state": "standing at the counter",
                },
            },
            {
                "id": "SCENE_002",
                "continuity": {
                    "previous_scene_id": None,
                    "required_start_state": "SCENE_001.ending_state",
                    "ending_state": "holding a coffee",
                },
            },
        ]
    }

    repair_storyboard_continuity(data)

    assert data["scenes"][0]["continuity"]["previous_scene_id"] is None
    assert (
        data["scenes"][1]["continuity"]["previous_scene_id"]
        == "SCENE_001"
    )
    assert (
        data["scenes"][1]["continuity"]["required_start_state"]
        == "standing at the counter"
    )


def test_missing_ending_state_is_repaired_from_scene_description():
    data = {
        "scenes": [
            {
                "id": "SCENE_001",
                "action": "The man sits by the window.",
                "visual_description": (
                    "The man is seated at a window table with his coffee."
                ),
                "continuity": {
                    "previous_scene_id": None,
                    "required_start_state": "outside",
                    "ending_state": None,
                },
            }
        ]
    }

    repair_storyboard_continuity(data)

    assert (
        data["scenes"][0]["continuity"]["ending_state"]
        == "The man is seated at a window table with his coffee."
    )