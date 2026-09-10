from app.assets.image_generator import build_scene_image_prompt
from app.storyboard.models import (
    Storyboard,
    Project,
    Character,
    CharacterAppearance,
    Location,
    Prop,
    Scene,
    Camera,
    Continuity,
)
from app.assets.image_generator import TOKENIZER

def test_scene_image_prompt():

    storyboard = Storyboard(
        project=Project(
            title="Coffee Shop",
            description="A young man enters a coffee shop.",
            duration_seconds=10,
            aspect_ratio="16:9",
            visual_style="cinematic realism",
        ),
        characters=[
            Character(
                id="CHAR_001",
                name="Daniel",
                appearance=CharacterAppearance(
                    age=28,
                    hair="short dark brown",
                    clothing="dark blue denim jacket over white shirt",
                    facial_features="oval face, brown eyes",
                    physical_description="average athletic build",
                ),
            )
        ],
        locations=[
            Location(
                id="LOC_001",
                name="Modern Coffee Shop",
                description="A modern minimalist coffee shop.",
                lighting="natural daylight",
                visual_style="modern cinematic",
            )
        ],
        props=[],
        scenes=[
            Scene(
                id="SCENE_001",
                duration_seconds=10,
                location_id="LOC_001",
                character_ids=["CHAR_001"],
                prop_ids=[],
                action="Daniel enters the coffee shop.",
                dialogue=None,
                visual_description=(
                    "Daniel walks through the entrance "
                    "of the coffee shop."
                ),
                camera=Camera(
                    shot_type="medium shot",
                    angle="eye-level",
                    movement="tracking",
                ),
                continuity=Continuity(
                    previous_scene_id=None,
                    character_state=(
                        "Daniel is entering the coffee shop."
                    ),
                    environment_state=(
                        "The coffee shop is open during daylight."
                    ),
                    required_start_state=(
                        "Daniel is outside the coffee shop entrance."
                    ),
                    ending_state=(
                        "Daniel is inside the coffee shop."
                    ),
                ),
            )
        ],
    )

    scene = storyboard.scenes[0]

    prompt = build_scene_image_prompt(
        storyboard,
        scene,
    )

    assert "Daniel" in prompt
    assert "coffee shop" in prompt.lower()
    assert "dark blue denim jacket" in prompt
    assert scene.action in prompt
        # SD 1.5 has a 77-token text limit.
    # Keep our generated prompt reasonably compact.
    print("\nIMAGE PROMPT:")
    print(prompt)

    tokens = TOKENIZER(
    prompt,
    truncation=False,
    return_tensors="pt",
    verbose=False,
)["input_ids"]

    print(f"\nCLIP tokens: {tokens.shape[1]}")
    assert tokens.shape[1] <= 75