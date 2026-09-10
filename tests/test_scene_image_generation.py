import json
from pathlib import Path

from app.assets.image_generator import (
    build_scene_image_prompt,
    generate_scene_image,
)
from app.storyboard.models import Storyboard


def test_generate_scene_image():

    storyboard_path = Path("outputs/storyboard.json")

    with open(
        storyboard_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    storyboard = Storyboard.model_validate(data)

    scene = storyboard.scenes[0]

    prompt = build_scene_image_prompt(
        storyboard=storyboard,
        scene=scene,
    )

    output_path = "outputs/scene_001.png"

    result = generate_scene_image(
        prompt=prompt,
        output_path=output_path,
    )

    assert Path(result) == Path(output_path)
    assert Path(output_path).exists()