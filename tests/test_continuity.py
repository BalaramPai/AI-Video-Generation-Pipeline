from pathlib import Path

from app.video.continuity import build_continuity_reference


def test_build_continuity_reference():

    frames = sorted(
        str(path)
        for path in Path(
            "outputs/scene_001_end_frames"
        ).glob("frame_*.png")
    )

    reference_frames = build_continuity_reference(frames)

    assert len(reference_frames) == 5

    for frame in reference_frames:
        assert Path(frame).exists()