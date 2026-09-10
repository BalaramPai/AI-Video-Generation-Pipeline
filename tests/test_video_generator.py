from pathlib import Path

import pytest

from app.video.generator import generate_scene_video


def test_generate_scene_video_requires_provider():
    image_path = Path("outputs/scene_001.png")

    assert image_path.exists()

    with pytest.raises(NotImplementedError):
        generate_scene_video(
            image_path=str(image_path),
            prompt="A young man walks toward the coffee counter.",
            output_path="outputs/scene_001.mp4",
            duration_seconds=5,
        )