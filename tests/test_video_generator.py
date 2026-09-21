from pathlib import Path

import pytest

import app.video.generator as video_generator
from app.video.generator import generate_scene_video


def test_generate_scene_video_requires_runway_key(monkeypatch):
    image_path = Path("outputs/scene_001.png")

    assert image_path.exists()

    monkeypatch.setattr(video_generator, "VIDEO_PROVIDER", "runway")
    monkeypatch.setattr(video_generator, "RUNWAY_API_SECRET", None)

    with pytest.raises(RuntimeError, match="RUNWAYML_API_SECRET"):
        generate_scene_video(
            image_path=str(image_path),
            prompt="A young man walks toward the coffee counter.",
            output_path="outputs/scene_001.mp4",
            duration_seconds=5,
        )
