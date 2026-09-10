import json
from pathlib import Path

from app.assets.image_generator import (
    build_continuity_image_prompt,
)
from app.video.continuity import (
    build_continuity_reference,
)
from app.storyboard.models import Storyboard


def test_scene2_uses_scene1_reference_frames():

    with open(
        "outputs/storyboard.json",
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    storyboard = Storyboard.model_validate(data)

    assert len(storyboard.scenes) >= 2

    scene_2 = storyboard.scenes[1]

    reference_frames = sorted(
        str(path)
        for path in Path(
            "outputs/scene_001_end_frames"
        ).glob("frame_*.png")
    )

    reference_frames = build_continuity_reference(
        reference_frames
    )

    assert len(reference_frames) == 5

    prompt = build_continuity_image_prompt(
        storyboard=storyboard,
        scene=scene_2,
        reference_frames=reference_frames,
    )

    print("\nContinuity references:")
    for frame in reference_frames:
        print(frame)

    print("\nScene 2 prompt:")
    print(prompt)

    assert "same person" in prompt
    assert "same clothes" in prompt
    assert "same setting" in prompt
    assert "continue action" in prompt