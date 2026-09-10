from pathlib import Path

from app.video.composer import concatenate_clips


def test_create_final_video():

    scene_1 = Path("outputs/test_scene.mp4")
    scene_2 = Path("outputs/scene_002.mp4")

    assert scene_1.exists()
    assert scene_2.exists()

    output = concatenate_clips(
        clip_paths=[
            str(scene_1),
            str(scene_2),
        ],
        output_path="outputs/final_demo.mp4",
    )

    assert Path(output).exists()