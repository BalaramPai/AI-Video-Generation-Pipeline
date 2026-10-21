from pathlib import Path

import pytest

import app.video.frame_extractor as frame_extractor
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


def test_extract_last_frames_uses_probed_decoded_frame_count(
    monkeypatch,
    tmp_path,
):
    video_path = tmp_path / "scene.mp4"
    video_path.write_bytes(b"video")
    output_dir = tmp_path / "frames"
    output_dir.mkdir()
    stale = output_dir / "frame_99.png"
    stale.write_bytes(b"stale")
    command_holder = {}

    class VideoInfo:
        frame_count = 12

    def fake_run(command, check):
        assert check is True
        command_holder["command"] = command
        pattern = Path(command[-1])
        for index in range(1, 4):
            (pattern.parent / f"frame_{index:02d}.png").write_bytes(b"frame")

    monkeypatch.setattr(frame_extractor, "probe_video", lambda _: VideoInfo())
    monkeypatch.setattr(frame_extractor, "get_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(frame_extractor.subprocess, "run", fake_run)

    frames = extract_last_frames(
        str(video_path),
        str(output_dir),
        frame_count=3,
    )

    assert len(frames) == 3
    assert not stale.exists()
    command = command_holder["command"]
    assert "select='between(n,9,11)'" in command[command.index("-vf") + 1]


def test_extract_last_frames_fails_if_video_has_too_few_frames(
    monkeypatch,
    tmp_path,
):
    video_path = tmp_path / "short.mp4"
    video_path.write_bytes(b"video")

    class VideoInfo:
        frame_count = 2

    monkeypatch.setattr(frame_extractor, "probe_video", lambda _: VideoInfo())
    monkeypatch.setattr(
        frame_extractor,
        "get_ffmpeg",
        lambda: pytest.fail("FFmpeg should not run"),
    )

    with pytest.raises(RuntimeError, match="cannot extract 5 ending frames"):
        extract_last_frames(
            str(video_path),
            str(tmp_path / "frames"),
            frame_count=5,
        )