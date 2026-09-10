from app.storyboard.models import Storyboard


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