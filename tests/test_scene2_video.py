from pathlib import Path

from app.video.composer import create_motion_clip


def test_create_scene2_video():

    image = Path("outputs/scene_002.png")

    assert image.exists()

    output = create_motion_clip(
        image_path=str(image),
        output_path="outputs/scene_002.mp4",
        duration=5,
    )

    assert Path(output).exists()