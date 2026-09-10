import json
from pathlib import Path

from app.assets.image_generator import (
    build_continuity_image_prompt,
    generate_scene_image_from_reference,
)
from app.video.continuity import (
    build_continuity_reference,
)
from app.storyboard.models import Storyboard


def test_generate_scene2_from_continuity():

    with open(
        "outputs/storyboard.json",
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    storyboard = Storyboard.model_validate(data)

    scene_2 = storyboard.scenes[1]

    frames = sorted(
        str(path)
        for path in Path(
            "outputs/scene_001_end_frames"
        ).glob("frame_*.png")
    )

    reference_frames = build_continuity_reference(
        frames
    )

    prompt = build_continuity_image_prompt(
        storyboard=storyboard,
        scene=scene_2,
        reference_frames=reference_frames,
    )

    # Latest frame is the actual visual conditioning image.
    reference_image = reference_frames[-1]

    output = generate_scene_image_from_reference(
        prompt=prompt,
        reference_image_path=reference_image,
        output_path="outputs/scene_002.png",
    )

    assert Path(output).exists()