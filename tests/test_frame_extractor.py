from pathlib import Path

from app.video.frame_extractor import extract_last_frames


def test_extract_last_frames():

    video = Path("outputs/test_scene.mp4")

    assert video.exists()

    frames = extract_last_frames(
        video_path=str(video),
        output_dir="outputs/scene_001_end_frames",
        frame_count=5,
    )

    assert len(frames) == 5

    for frame in frames:
        assert Path(frame).exists()