from pathlib import Path

from app.video.composer import create_motion_clip


def test_create_motion_clip():

    image = Path("outputs/scene_001.png")

    assert image.exists()

    output = create_motion_clip(
        image_path=str(image),
        output_path="outputs/test_scene.mp4",
        duration=3,
    )

    assert Path(output).exists()