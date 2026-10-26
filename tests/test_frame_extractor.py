from fractions import Fraction
import logging
from pathlib import Path
import re
import subprocess

from PIL import Image
import pytest

import app.video.frame_extractor as frame_extractor
from app.video.composer import VideoInfo
from app.video.frame_extractor import extract_last_frames


def _video_info(frame_count: int = 12) -> VideoInfo:
    return VideoInfo(
        frame_rate=Fraction(24, 1),
        frame_count=frame_count,
        duration_seconds=Fraction(frame_count, 24),
        video_codec="h264",
        width=1280,
        height=720,
        pixel_format="yuv420p",
        audio_codec=None,
        audio_sample_rate=None,
        audio_channels=None,
    )


def _write_frame(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (16, 12), color).save(path)


def test_extract_last_frames_uses_source_final_frame_and_removes_stale_files(
    monkeypatch,
    tmp_path,
    caplog,
):
    video_path = tmp_path / "scene_001.mp4"
    video_path.write_bytes(b"normalized previous scene")
    output_dir = tmp_path / "scene_001_end_frames"
    output_dir.mkdir()
    stale = output_dir / "frame_000011.png"
    _write_frame(stale, (1, 2, 3))
    calls = []

    def fake_run(command, check, capture_output):
        assert check is True
        assert capture_output is True
        calls.append(command)
        match = re.search(r"eq\(n\\?,(\d+)\)", command[command.index("-vf") + 1])
        assert match
        frame_index = int(match.group(1))
        _write_frame(Path(command[-1]), (frame_index, 0, 0))
        return subprocess.CompletedProcess(command, 0)

    def fake_probe(path):
        assert Path(path) == video_path
        return _video_info()

    monkeypatch.setattr(frame_extractor, "probe_video", fake_probe)
    monkeypatch.setattr(frame_extractor, "get_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(frame_extractor.subprocess, "run", fake_run)

    with caplog.at_level(logging.INFO, logger="app.video.frame_extractor"):
        frames = extract_last_frames(
            str(video_path),
            str(output_dir),
            frame_count=3,
        )

    assert [Path(frame).name for frame in frames] == [
        "frame_000009.png",
        "frame_000010.png",
        "frame_000011.png",
    ]
    assert Path(frames[-1]).is_file()
    assert Path(frames[-1]).stat().st_size > 0
    assert len(calls) == 3
    assert "Extracted and verified" in caplog.text
    assert caplog.records[0].source_video == str(video_path)
    assert caplog.records[0].destination_image == frames[-1]
    assert caplog.records[0].frame_index == 11
    assert caplog.records[0].timestamp_seconds == pytest.approx(11 / 24)
    assert caplog.records[0].image_width == 16
    assert caplog.records[0].image_height == 12
    with Image.open(frames[-1]) as image:
        assert image.size == (16, 12)
        assert image.getpixel((0, 0)) == (11, 0, 0)
        assert image.getpixel((0, 0)) != (1, 2, 3)


def test_extract_last_frames_falls_back_to_nearest_earlier_valid_frame(
    monkeypatch,
    tmp_path,
    caplog,
):
    video_path = tmp_path / "scene.mp4"
    video_path.write_bytes(b"normalized video")
    output_dir = tmp_path / "frames"
    attempts = []

    def fake_run(command, check, capture_output):
        match = re.search(r"eq\(n\\?,(\d+)\)", command[command.index("-vf") + 1])
        assert match
        frame_index = int(match.group(1))
        attempts.append(frame_index)
        output_path = Path(command[-1])
        if frame_index == 11:
            output_path.write_bytes(b"damaged png")
        else:
            _write_frame(output_path, (frame_index, 20, 30))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        frame_extractor,
        "probe_video",
        lambda _: _video_info(),
    )
    monkeypatch.setattr(frame_extractor, "get_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(frame_extractor.subprocess, "run", fake_run)

    frames = extract_last_frames(
        str(video_path),
        str(output_dir),
        frame_count=1,
    )

    assert attempts == [11, 10]
    assert [Path(frame).name for frame in frames] == ["frame_000010.png"]
    assert "earlier valid frame" in caplog.text
    with Image.open(frames[0]) as image:
        assert image.getpixel((0, 0)) == (10, 20, 30)


def test_extract_last_frames_fails_if_no_candidate_decodes(
    monkeypatch,
    tmp_path,
):
    video_path = tmp_path / "scene.mp4"
    video_path.write_bytes(b"video")

    def fake_run(command, check, capture_output):
        Path(command[-1]).write_bytes(b"not an image")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        frame_extractor,
        "probe_video",
        lambda _: _video_info(frame_count=2),
    )
    monkeypatch.setattr(frame_extractor, "get_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(frame_extractor.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Could not extract a valid"):
        extract_last_frames(
            str(video_path),
            str(tmp_path / "frames"),
            frame_count=5,
        )


def test_extract_last_frames_tries_earlier_frame_after_ffmpeg_error(
    monkeypatch,
    tmp_path,
):
    video_path = tmp_path / "scene.mp4"
    video_path.write_bytes(b"video")
    attempted_indexes = []

    def fake_run(command, check, capture_output):
        match = re.search(r"eq\(n\\?,(\d+)\)", command[command.index("-vf") + 1])
        assert match
        frame_index = int(match.group(1))
        attempted_indexes.append(frame_index)
        if frame_index == 11:
            raise subprocess.CalledProcessError(1, command, stderr=b"decode")
        _write_frame(Path(command[-1]), (frame_index, 0, 0))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(
        frame_extractor,
        "probe_video",
        lambda _: _video_info(),
    )
    monkeypatch.setattr(frame_extractor, "get_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(frame_extractor.subprocess, "run", fake_run)

    frames = extract_last_frames(
        str(video_path),
        str(tmp_path / "frames"),
        frame_count=1,
    )

    assert attempted_indexes == [11, 10]
    assert Path(frames[0]).name == "frame_000010.png"
