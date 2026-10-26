from PIL import Image
import pytest

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


def test_build_continuity_reference_rejects_corrupt_image(tmp_path):
    corrupt_frame = tmp_path / "corrupt.png"
    corrupt_frame.write_bytes(b"not a png")

    with pytest.raises(ValueError, match="not a readable image"):
        build_continuity_reference([str(corrupt_frame)])


def test_build_continuity_reference_accepts_decodable_image(tmp_path):
    frame = tmp_path / "frame.png"
    Image.new("RGB", (8, 6)).save(frame)

    assert build_continuity_reference([str(frame)]) == [str(frame)]