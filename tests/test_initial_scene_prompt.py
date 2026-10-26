from app.assets.image_generator import (
    IMAGE_PROMPT_MAX_TOKENS,
    _token_count,
    build_initial_scene_negative_prompt,
    build_continuity_image_prompt,
    build_initial_scene_image_prompt,
    build_scene_negative_prompt,
    build_scene_image_prompt,
)
from app.storyboard.models import (
    Camera,
    Character,
    CharacterAppearance,
    Continuity,
    Location,
    Project,
    Prop,
    Scene,
    Storyboard,
)


def _eclipse_opening_storyboard() -> Storyboard:
    return Storyboard(
        project=Project(
            title="Eclipse lesson",
            description="A science educator introduces a solar eclipse model.",
            duration_seconds=10,
            aspect_ratio="16:9",
            visual_style="Photorealistic educational studio demonstration.",
        ),
        characters=[
            Character(
                id="PERSON_A",
                name="Maya",
                appearance=CharacterAppearance(
                    age=25,
                    hair="dark brown hair in a low ponytail",
                    clothing="navy-blue shirt and dark trousers",
                    facial_features="natural, realistic face",
                    physical_description="adult educator",
                    eye_color="brown",
                    distinctive_features="small mole on right cheek",
                    wardrobe_anchor="same navy-blue shirt",
                ),
            )
        ],
        locations=[
            Location(
                id="ROOM_A",
                name="Science Demonstration Studio",
                description="A tidy studio with a gray wall and wooden table.",
                lighting="Soft, even studio lighting.",
                visual_style="Photorealistic neutral studio.",
                continuity_anchor=(
                    "Lamp on the left, globe on the right, open space "
                    "between them."
                ),
            )
        ],
        props=[
            Prop(
                id="PROP_LAMP",
                name="Sun Demonstration Lamp",
                description="A compact warm-white demonstration lamp.",
                continuity_anchor="Stationary on the far left of the table.",
            ),
            Prop(
                id="PROP_GLOBE",
                name="Earth Globe",
                description="A classroom globe on a dark stand.",
                continuity_anchor="Stationary on the right side of the table.",
            ),
            Prop(
                id="PROP_SPHERE",
                name="Moon Sphere",
                description="One small matte dark-gray sphere.",
                continuity_anchor="The single sphere is held in Maya's right hand.",
            ),
            Prop(
                id="PROP_UNUSED",
                name="Decorative Book",
                description="An unrelated book.",
                continuity_anchor="Stored away from the demonstration.",
            ),
        ],
        scenes=[
            Scene(
                id="OPENING_FRAME",
                duration_seconds=5,
                location_id="ROOM_A",
                character_ids=["PERSON_A"],
                prop_ids=["PROP_LAMP", "PROP_GLOBE", "PROP_SPHERE"],
                action=(
                    "Maya moves the Moon between the lamp and Earth to show "
                    "the eclipse."
                ),
                dialogue=None,
                visual_description=(
                    "A medium-wide eye-level view shows Maya behind the "
                    "wooden table. The lamp is visible on the left and the "
                    "globe on the right. She holds the Moon Sphere in her "
                    "right hand."
                ),
                camera=Camera(
                    shot_type="Medium-wide shot",
                    angle="Eye-level",
                    movement="Locked camera",
                    lens="Natural perspective",
                ),
                continuity=Continuity(
                    previous_scene_id=None,
                    character_state="Maya stands behind the table.",
                    environment_state="A tidy studio with soft lighting.",
                    required_start_state=(
                        "Opening shot: Maya stands behind the table with "
                        "the single Moon Sphere in her right hand near "
                        "chest height."
                    ),
                    ending_state=(
                        "Maya holds the sphere between the lamp and globe."
                    ),
                ),
                blocking=(
                    "Keep the lamp on the left and globe on the right. "
                    "Maya stands centered behind the table."
                ),
                shot_intent="Introduce the eclipse model.",
                visual_continuity="Establish the demonstration layout.",
                negative_prompt=(
                    "No duplicate lamps, globes, or spheres; no clutter."
                ),
            )
        ],
    )


def _solar_system_opening_storyboard() -> Storyboard:
    storyboard = _eclipse_opening_storyboard()
    storyboard.characters = []
    location = storyboard.locations[0]
    location.name = "Space"
    location.description = "Dark space with a sparse starfield."
    location.lighting = "The Sun lights the Moon and Earth's visible side."
    location.continuity_anchor = (
        "Sun stays left; Earth stays right; camera stays fixed."
    )
    storyboard.props = [
        Prop(
            id="SUN",
            name="Sun",
            description="Bright yellow-white Sun, on the left.",
            continuity_anchor="One Sun; fixed position and size.",
        ),
        Prop(
            id="MOON",
            name="Moon",
            description="One small gray cratered Moon.",
            continuity_anchor="One Moon only; it moves between Sun and Earth.",
        ),
        Prop(
            id="EARTH",
            name="Earth",
            description="Blue-green Earth with visible continents, on the right.",
            continuity_anchor="One Earth; fixed position and size.",
        ),
    ]
    scene = storyboard.scenes[0]
    scene.character_ids = []
    scene.prop_ids = ["SUN", "MOON", "EARTH"]
    scene.camera.shot_type = "Wide shot"
    scene.camera.angle = "Side-on view"
    scene.camera.movement = "Locked camera"
    scene.action = (
        "Establish the Sun, Moon, and Earth. The Moon drifts slowly."
    )
    scene.visual_description = (
        "Wide view: bright Sun left, small gray Moon near center, "
        "blue-green Earth right. All three fully visible."
    )
    scene.continuity.required_start_state = (
        "Opening view; Moon slightly above the line between Sun and Earth."
    )
    scene.continuity.ending_state = "Sun and Earth stay fixed."
    scene.blocking = (
        "Sun left; Earth right; Moon near center, slightly above their line."
    )
    return storyboard


def test_initial_scene_prompt_prioritizes_composition_within_clip_budget():
    storyboard = _eclipse_opening_storyboard()
    scene = storyboard.scenes[0]

    prompt = build_initial_scene_image_prompt(storyboard, scene)

    assert _token_count(prompt) <= IMAGE_PROMPT_MAX_TOKENS
    assert scene.camera.shot_type.lower() in prompt.lower()
    assert scene.character_ids and "Maya" in prompt
    for prop in storyboard.props:
        if prop.id in scene.prop_ids:
            assert prop.name in prompt
    assert "full table" in prompt.lower()
    assert "lamp on the left and globe on the right" in prompt.lower()
    assert "single Moon Sphere in her right hand" in prompt
    assert "natural scale" in prompt.lower()
    assert "clean background" in prompt.lower()
    assert "no duplicates" in prompt.lower()
    assert "Decorative Book" not in prompt
    assert scene.action not in prompt


def test_initial_scene_prompt_uses_scene_order_not_a_special_scene_id():
    storyboard = _eclipse_opening_storyboard()
    scene = storyboard.scenes[0]
    scene.id = "FIRST_SCENE_WITH_AN_ARBITRARY_ID"

    prompt = build_initial_scene_image_prompt(storyboard, scene)

    assert "FIRST_SCENE_WITH_AN_ARBITRARY_ID" not in prompt
    assert "Sun Demonstration Lamp" in prompt
    assert _token_count(prompt) <= IMAGE_PROMPT_MAX_TOKENS


def test_initial_scene_prompt_does_not_assume_a_table():
    storyboard = _eclipse_opening_storyboard()
    location = storyboard.locations[0]
    location.description = "A neutral studio."
    location.continuity_anchor = "A clean neutral space."
    scene = storyboard.scenes[0]
    scene.blocking = "Place the lamp left and globe right."
    scene.visual_description = (
        "A clear view of the educator and demonstration objects."
    )
    scene.continuity.required_start_state = (
        "The educator holds one small sphere."
    )

    prompt = build_initial_scene_image_prompt(storyboard, scene)

    assert "complete scene setup" in prompt.lower()
    assert "full table" not in prompt.lower()
    assert _token_count(prompt) <= IMAGE_PROMPT_MAX_TOKENS


def test_shared_and_continuity_prompt_builders_remain_separate():
    storyboard = _eclipse_opening_storyboard()
    scene = storyboard.scenes[0]
    shared_prompt_before = build_scene_image_prompt(storyboard, scene)
    initial_prompt = build_initial_scene_image_prompt(storyboard, scene)
    shared_prompt = build_scene_image_prompt(storyboard, scene)

    assert initial_prompt != shared_prompt
    assert shared_prompt == shared_prompt_before
    assert "Opening state" not in shared_prompt
    assert _token_count(shared_prompt) <= IMAGE_PROMPT_MAX_TOKENS

    continuation_scene = scene.model_copy(deep=True)
    continuation_scene.id = "LATER_SCENE"
    continuation_scene.continuity.previous_scene_id = scene.id
    continuation_scene.continuity.required_start_state = (
        "Maya holds the same Moon Sphere in her right hand."
    )
    continuity_prompt = build_continuity_image_prompt(
        storyboard,
        continuation_scene,
        ["previous-ending-frame.png"],
    )

    assert "preserve reference composition" in continuity_prompt.lower()
    assert continuation_scene.continuity.required_start_state in (
        continuity_prompt
    )
    assert _token_count(continuity_prompt) <= IMAGE_PROMPT_MAX_TOKENS


def test_compiled_initial_solar_prompt_names_and_places_all_three_bodies():
    storyboard = _solar_system_opening_storyboard()

    prompt = build_initial_scene_image_prompt(
        storyboard,
        storyboard.scenes[0],
    )

    assert _token_count(prompt) <= IMAGE_PROMPT_MAX_TOKENS
    assert "one luminous star sun on the left" in prompt.lower()
    assert (
        "one small gray cratered moon near center, slightly above the "
        "sun-earth line" in prompt.lower()
    )
    assert "one recognizable blue earth on the right" in prompt.lower()
    assert "dark space" in prompt.lower()
    assert "all three distinct and fully visible" in prompt.lower()
    assert "complete scene setup" not in prompt.lower()
    assert ".," not in prompt
    assert ",;" not in prompt


def test_initial_solar_negative_prompt_suppresses_duplicates_not_bodies():
    storyboard = _solar_system_opening_storyboard()

    negative_prompt = build_initial_scene_negative_prompt(
        storyboard,
        storyboard.scenes[0],
    )

    assert "duplicate objects" in negative_prompt
    assert "repeated objects" in negative_prompt
    assert "extra planets" not in negative_prompt
    assert "duplicate bodies" not in negative_prompt
    assert not any(
        object_name in negative_prompt.lower()
        for object_name in ("sun", "moon", "earth", "celestial bodies")
    )
    assert _token_count(negative_prompt) <= IMAGE_PROMPT_MAX_TOKENS


def test_non_solar_initial_negative_prompt_preserves_existing_behavior():
    storyboard = _eclipse_opening_storyboard()
    scene = storyboard.scenes[0]

    assert build_initial_scene_negative_prompt(storyboard, scene) == (
        build_scene_negative_prompt(scene)
    )
