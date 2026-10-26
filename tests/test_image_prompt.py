import json

import pytest
from PIL import Image

import app.assets.image_generator as image_generator
from app.assets.image_generator import build_scene_image_prompt
from app.assets.image_generator import (
    IMAGE_PROMPT_MAX_TOKENS,
    TOKENIZER,
    _token_count,
    build_continuity_image_prompt,
    build_scene_negative_prompt,
)
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
                action="Daniel receives a coffee.",
                dialogue=None,
                visual_description=(
                    "Daniel stands at the counter holding a white cup."
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
                    required_start_state="Daniel stands at the counter.",
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


def test_scene_image_prompt_supports_character_free_subject():
    storyboard = Storyboard(
        project=Project(
            title="Engine Explainer",
            description="Show the parts inside a car engine.",
            duration_seconds=5,
            aspect_ratio="16:9",
            visual_style="Clean instructional 3D animation",
        ),
        characters=[],
        locations=[
            Location(
                id="LOC_001",
                name="Instructional studio",
                description="A plain studio background",
                lighting="Even bright lighting",
                visual_style="Clean 3D animation",
            )
        ],
        props=[],
        scenes=[
            Scene(
                id="SCENE_001",
                duration_seconds=5,
                location_id="LOC_001",
                character_ids=[],
                prop_ids=[],
                action="The engine cover opens to reveal its pistons.",
                visual_description=(
                    "A gasoline engine is centered with the cover open "
                    "and pistons visible."
                ),
                camera=Camera(
                    shot_type="Medium view",
                    angle="Side view",
                    movement="Steady camera",
                ),
                continuity=Continuity(
                    previous_scene_id=None,
                    character_state=(
                        "The engine is open with its pistons visible."
                    ),
                    environment_state="A clean studio with even lighting.",
                    required_start_state="The engine cover is closed.",
                    ending_state="The engine cover is open.",
                ),
            )
        ],
    )

    prompt = build_scene_image_prompt(storyboard, storyboard.scenes[0])

    assert "engine" in prompt.lower()
    assert "start: the engine cover is closed" in prompt.lower()
    assert "pistons visible" in prompt.lower()
    assert _token_count(prompt) <= IMAGE_PROMPT_MAX_TOKENS


def _circuit_storyboard(*, long_optional_context=False):
    style = (
        "An extremely detailed visual style description "
        "that should be omitted before critical visual facts " * 8
        if long_optional_context
        else "Clean instructional 3D"
    )
    location_name = (
        "A highly elaborate laboratory environment with descriptive "
        "architecture and decorations " * 8
        if long_optional_context
        else "Neutral lab"
    )
    return Storyboard(
        project=Project(
            title="Circuit lesson",
            description="Show a circuit and how a bulb lights.",
            duration_seconds=10,
            aspect_ratio="16:9",
            visual_style=style,
        ),
        characters=[],
        locations=[
            Location(
                id="LOC_001",
                name=location_name,
                description="A plain neutral studio.",
                lighting="Bright even light",
                visual_style="Clean",
            )
        ],
        props=[
            Prop(
                id="PROP_BATTERY",
                name="Battery",
                description="Rectangular battery",
                continuity_anchor="+ terminal toward wire",
            ),
            Prop(
                id="PROP_WIRE",
                name="Red wire",
                description="Red wire",
                continuity_anchor="Battery to switch",
            ),
            Prop(
                id="PROP_SWITCH",
                name="Switch",
                description="Lever switch",
                continuity_anchor="Open lever",
            ),
            Prop(
                id="PROP_BULB",
                name="Bulb",
                description="Clear bulb",
                continuity_anchor="Unlit",
            ),
            Prop(
                id="PROP_UNUSED",
                name="Unrelated book",
                description="A large blue textbook",
                continuity_anchor="Closed on a shelf",
            ),
        ],
        scenes=[
            Scene(
                id="SCENE_001",
                duration_seconds=5,
                location_id="LOC_001",
                character_ids=[],
                prop_ids=[
                    "PROP_BATTERY",
                    "PROP_WIRE",
                    "PROP_SWITCH",
                    "PROP_BULB",
                ],
                action=(
                    "Show open switch; bulb stays unlit."
                ),
                visual_description=(
                    "Battery, wire, switch, bulb left-to-right in one "
                    "connected circuit."
                ),
                camera=Camera(
                    shot_type="Wide instructional view",
                    angle="Straight-on",
                    movement="Locked camera",
                ),
                continuity=Continuity(
                    previous_scene_id=None,
                    character_state=(
                        "The open circuit and unlit bulb are clearly visible."
                    ),
                    environment_state="Neutral lab, bright even light.",
                    required_start_state=(
                        "Battery and bulb connected by wire; switch open; "
                        "bulb unlit."
                    ),
                    ending_state=(
                        "The switch remains open and the bulb remains unlit."
                    ),
                ),
            ),
            Scene(
                id="SCENE_002",
                duration_seconds=5,
                location_id="LOC_001",
                character_ids=[],
                prop_ids=["PROP_BATTERY", "PROP_SWITCH", "PROP_BULB"],
                action="Close switch; bulb lights.",
                visual_description=(
                    "Same circuit parts in left-to-right layout."
                ),
                camera=Camera(
                    shot_type="Wide instructional view",
                    angle="Straight-on",
                    movement="Locked camera",
                ),
                continuity=Continuity(
                    previous_scene_id="SCENE_001",
                    character_state="The switch is closed and bulb is lit.",
                    environment_state="Neutral lab, bright even light.",
                    required_start_state="Same battery, wire, open switch, unlit bulb.",
                    ending_state="The switch is closed and bulb is lit.",
                ),
            ),
        ],
    )


def test_image_prompt_preserves_circuit_start_state_and_relevant_props():
    storyboard = _circuit_storyboard()
    prompt = build_scene_image_prompt(storyboard, storyboard.scenes[0])

    for detail in (
        "Battery",
        "Red wire",
        "Switch",
        "Bulb",
        "switch open",
        "bulb unlit",
        "left-to-right in one connected circuit",
    ):
        assert detail.lower() in prompt.lower()
    assert "Unrelated book" not in prompt
    assert _token_count(prompt) <= IMAGE_PROMPT_MAX_TOKENS


def test_long_optional_style_and_location_do_not_remove_scene_facts():
    storyboard = _circuit_storyboard(long_optional_context=True)
    prompt = build_scene_image_prompt(storyboard, storyboard.scenes[0])

    assert "switch lever open" in prompt.lower()
    assert "bulb unlit" in prompt.lower()
    assert "left-to-right in one connected circuit" in prompt.lower()
    assert _token_count(prompt) <= IMAGE_PROMPT_MAX_TOKENS


def test_overlong_required_image_facts_are_balanced_and_warned(caplog):
    caplog.set_level("INFO", logger="app.assets.image_generator")
    storyboard = _circuit_storyboard()
    scene = storyboard.scenes[0]
    scene.action = "The " + "detailed action " * 100

    prompt = build_scene_image_prompt(storyboard, scene)

    assert _token_count(prompt) <= IMAGE_PROMPT_MAX_TOKENS
    assert "Start:" in prompt
    assert "Visual:" in prompt
    assert "switch lever open" in prompt.lower()
    assert "Action: The detailed action detailed action" not in prompt
    assert any(
        record.message
        == "Optional Stable Diffusion prompt context omitted to preserve "
        "required scene facts"
        and "action" in record.omitted_context
        for record in caplog.records
    )


def test_image_prompt_fails_when_no_complete_section_fits():
    with pytest.raises(
        ValueError,
        match="could not fit any complete required section",
    ):
        image_generator._fit_image_prompt(
            ["Subject: " + "unbroken " * 500],
            "Scene image prompt",
        )


def test_continuity_prompt_preserves_next_start_state_and_action():
    storyboard = _circuit_storyboard()
    prompt = build_continuity_image_prompt(
        storyboard,
        storyboard.scenes[1],
        ["previous-final-frame.png"],
    )

    assert "same battery, wire, open switch, unlit bulb" in prompt.lower()
    assert "close switch; bulb lights" in prompt.lower()
    assert "preserve reference composition" in prompt.lower()
    assert _token_count(prompt) <= IMAGE_PROMPT_MAX_TOKENS


def test_continuity_prompt_prioritizes_visible_props_over_motion_detail():
    storyboard = _circuit_storyboard()
    scene = storyboard.scenes[0]
    scene.blocking = (
        "The battery stays centered. Keep the battery, wire, switch, and "
        "bulb visible together."
    )
    scene.continuity.required_start_state = (
        "The same circuit remains in its exact previous state. "
        + "Preserve every original component and position. " * 5
    )
    scene.action = "Continue the demonstration " * 20

    prompt = build_continuity_image_prompt(
        storyboard,
        scene,
        ["previous-final-frame.png"],
    )

    assert "preserve reference composition" in prompt.lower()
    assert "visible together" in prompt.lower()
    for prop in ("Battery", "Red wire", "Switch", "Bulb"):
        assert prop.lower() in prompt.lower()
    assert _token_count(prompt) <= IMAGE_PROMPT_MAX_TOKENS


def test_character_scene_negative_prompt_keeps_defaults_and_custom_text():
    storyboard = _circuit_storyboard()
    scene = storyboard.scenes[0]
    scene.character_ids = ["CHAR_001"]
    scene.negative_prompt = "no extra people"

    negative_prompt = build_scene_negative_prompt(scene)

    assert "low quality" in negative_prompt
    assert "bad hands" in negative_prompt
    assert "no extra people" in negative_prompt


def test_noncharacter_negative_prompt_avoids_human_specific_rules():
    negative_prompt = build_scene_negative_prompt(_circuit_storyboard().scenes[0])

    assert "low quality" in negative_prompt
    assert "bad hands" not in negative_prompt


def test_long_custom_negative_prompt_respects_token_limit_and_warns(caplog):
    storyboard = _circuit_storyboard()
    scene = storyboard.scenes[0]
    scene.negative_prompt = (
        "no different apparatus, no table lamp, no dispenser, no people, "
        "no hands, no extra props, no new filter layers, "
        "no sediment vanishing from the filter, no reverse water flow, "
        "no instant transformation of all water to crystal-clear liquid, "
        "no overflow, no spilling, no drinking, no glass moving, "
        "no camera movement, no cuts, no dramatic effects, no captions, "
        "no logos, no watermarks, no abrupt final transition"
    )

    prompt = build_scene_negative_prompt(scene)

    assert _token_count(prompt) <= IMAGE_PROMPT_MAX_TOKENS
    assert "low quality" in prompt
    assert "blurry" in prompt
    assert "no different apparatus" in prompt
    assert "no table lamp" in prompt
    assert any("exceeded" in record.message for record in caplog.records)


def test_scene_image_generation_passes_validated_prompt_and_negative(
    monkeypatch,
    tmp_path,
):
    captured = {}

    class FakePipeline:
        tokenizer = TOKENIZER

        def __call__(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Result",
                (),
                {"images": [Image.new("RGB", (16, 16))]},
            )()

    monkeypatch.setattr(
        image_generator,
        "get_pipeline",
        lambda: FakePipeline(),
    )
    output = tmp_path / "scene.png"
    prompt = "A red car with the hood open in a clean garage."
    negative_prompt = "low quality, no extra cars"
    metadata_path = tmp_path / "scene_image_generation.json"

    image_generator.generate_scene_image(
        prompt=prompt,
        output_path=str(output),
        negative_prompt=negative_prompt,
        aspect_ratio="16:9",
        metadata_path=metadata_path,
    )

    assert captured["prompt"] == prompt
    assert captured["negative_prompt"] == negative_prompt
    assert captured["width"] == 768
    assert captured["height"] == 432
    assert captured["num_inference_steps"] == 35
    assert captured["guidance_scale"] == 6.5
    assert "generator" not in captured
    assert output.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["prompt"] == prompt
    assert metadata["negative_prompt"] == negative_prompt
    assert metadata["model_id"] == image_generator.MODEL_ID
    assert metadata["dimensions"] == {"width": 768, "height": 432}
    assert metadata["num_inference_steps"] == 35
    assert metadata["guidance_scale"] == 6.5
    assert metadata["generated_at_utc"].endswith("+00:00")


def test_scene_image_generation_rejects_prompt_that_pipeline_would_truncate(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        image_generator,
        "get_pipeline",
        lambda: pytest.fail("The model should not load for invalid prompts"),
    )

    with pytest.raises(ValueError, match="would be truncated"):
        image_generator.generate_scene_image(
            prompt="engine " * 100,
            output_path=str(tmp_path / "scene.png"),
        )


def test_reference_image_generation_passes_reference_prompt_and_start_state(
    monkeypatch,
    tmp_path,
):
    reference_path = tmp_path / "reference.png"
    Image.new("RGB", (32, 32)).save(reference_path)
    captured = {}

    class FakePipeline:
        tokenizer = TOKENIZER

        def enable_model_cpu_offload(self):
            pass

        def __call__(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Result",
                (),
                {"images": [Image.new("RGB", (16, 16))]},
            )()

    class FakeImg2ImgPipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakePipeline()

    import diffusers

    monkeypatch.setattr(
        diffusers,
        "StableDiffusionImg2ImgPipeline",
        FakeImg2ImgPipeline,
    )
    output_path = tmp_path / "scene2.png"
    prompt = (
        "Subject: same battery and bulb. Initial state: switch open. "
        "Action: close the switch."
    )
    negative_prompt = "low quality, no extra lamps"

    image_generator.generate_scene_image_from_reference(
        prompt=prompt,
        reference_image_path=str(reference_path),
        output_path=str(output_path),
        negative_prompt=negative_prompt,
        scene_id="SCENE_002",
    )

    assert captured["prompt"] == prompt
    assert captured["negative_prompt"] == negative_prompt
    assert captured["image"].size == (32, 32)
    assert captured["strength"] == 0.30
    assert output_path.is_file()