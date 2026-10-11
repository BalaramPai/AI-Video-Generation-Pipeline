import json
from fractions import Fraction
from pathlib import Path
import subprocess

import pytest

from app.video import composer


def test_get_ffmpeg_uses_path_search(monkeypatch, tmp_path):
    expected = tmp_path / "ffmpeg.exe"
    expected.write_bytes(b"executable")
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    monkeypatch.setattr(composer.shutil, "which", lambda _: str(expected))

    assert composer.get_ffmpeg() == str(expected)


def test_get_ffmpeg_reports_missing_install(monkeypatch):
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(composer.shutil, "which", lambda _: None)

    with pytest.raises(FileNotFoundError, match="FFMPEG_PATH"):
        composer.get_ffmpeg()


def test_get_ffmpeg_finds_standard_winget_install(monkeypatch, tmp_path):
    executable = (
        tmp_path
        / "Microsoft"
        / "WinGet"
        / "Packages"
        / "Gyan.FFmpeg.Shared"
        / "ffmpeg-9.0"
        / "bin"
        / "ffmpeg.exe"
    )
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"executable")
    monkeypatch.delenv("FFMPEG_PATH", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(composer.shutil, "which", lambda _: None)

    assert composer.get_ffmpeg() == str(executable)


def test_concatenate_clips_reencodes_and_cleans_temporary_files(
    monkeypatch,
    tmp_path,
):
    clip_paths = [tmp_path / "one.mp4", tmp_path / "two.mp4"]
    for path in clip_paths:
        path.write_bytes(b"source video")
    output_path = tmp_path / "final.mp4"
    command_holder = {}

    def fake_run(command, check):
        command_holder["command"] = command
        assert check is True
        concat_index = command.index("-i") + 1
        concat_list = Path(command[concat_index])
        command_holder["concat_list"] = concat_list.read_text(
            encoding="utf-8"
        )
        Path(command[-1]).write_bytes(b"joined video")

    monkeypatch.setattr(composer, "get_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(composer.subprocess, "run", fake_run)

    assert composer.concatenate_clips(
        clip_paths=[str(path) for path in clip_paths],
        output_path=str(output_path),
    ) == str(output_path)
    assert output_path.read_bytes() == b"joined video"
    assert "-c:v" in command_holder["command"]
    assert command_holder["command"][
        command_holder["command"].index("-c:v") + 1
    ] == "libx264"
    assert all(path.resolve().as_posix() in command_holder["concat_list"]
               for path in clip_paths)
    assert not list(tmp_path.glob("*.ffconcat"))


def test_concatenate_requires_existing_clips(tmp_path):
    with pytest.raises(ValueError, match="At least one"):
        composer.concatenate_clips([], str(tmp_path / "out.mp4"))

    with pytest.raises(FileNotFoundError, match="Video clip not found"):
        composer.concatenate_clips(
            [str(tmp_path / "missing.mp4")],
            str(tmp_path / "out.mp4"),
        )


@pytest.mark.parametrize(
    ("durations", "fps", "expected"),
    [
        ([10, 5], Fraction(24, 1), [240, 120]),
        ([10, 5], Fraction(30000, 1001), [300, 150]),
        ([2.5, 3.5, 1], Fraction(30, 1), [75, 105, 30]),
        ([0.0625, 0.0625], Fraction(24, 1), [2, 1]),
        ([0.1, 0.1, 0.1], Fraction(25, 1), [3, 2, 3]),
    ],
)
def test_calculate_scene_frame_counts_uses_cumulative_rounding(
    durations,
    fps,
    expected,
):
    counts = composer.calculate_scene_frame_counts(durations, fps)
    assert counts == expected
    assert sum(counts) == round(float(sum(durations) * fps))


def test_calculate_scene_frame_counts_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="positive"):
        composer.calculate_scene_frame_counts([0], Fraction(24, 1))
    with pytest.raises(ValueError, match="representable frame"):
        composer.calculate_scene_frame_counts(
            [0.001, 0.001],
            Fraction(24, 1),
        )


def test_normalize_video_clip_trims_extra_frames_and_checks_output(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "generated.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "normalized.mp4"
    normalized_temp = tmp_path / ".normalized.normalized.mp4"
    infos = iter(
        [
            _video_info(Fraction(24), 145),
            _video_info(Fraction(24), 120),
        ]
    )
    command_holder = {}

    def fake_run(command, check):
        command_holder["command"] = command
        assert check is True
        Path(command[-1]).write_bytes(b"normalized")

    monkeypatch.setattr(composer, "probe_video", lambda _: next(infos))
    monkeypatch.setattr(composer, "get_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(composer.subprocess, "run", fake_run)

    info = composer.normalize_video_clip(
        source,
        output,
        target_frame_rate=Fraction(24),
        target_frame_count=120,
    )

    command = command_holder["command"]
    assert "fps=fps=24/1:round=near,trim=end_frame=120" in command[
        command.index("-vf") + 1
    ]
    assert command[command.index("-frames:v") + 1] == "120"
    assert info.frame_count == 120
    assert output.read_bytes() == b"normalized"
    assert not normalized_temp.exists()


def test_normalize_video_clip_rejects_short_source_before_encoding(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "short.mp4"
    source.write_bytes(b"source")
    monkeypatch.setattr(
        composer,
        "probe_video",
        lambda _: _video_info(Fraction(24), 119),
    )
    monkeypatch.setattr(
        composer.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("FFmpeg should not run"),
    )

    with pytest.raises(ValueError, match="too short"):
        composer.normalize_video_clip(
            source,
            tmp_path / "normalized.mp4",
            target_frame_rate=Fraction(24),
            target_frame_count=120,
        )


def test_normalize_video_clip_reports_ffmpeg_failure_and_cleans_temp(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "generated.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "normalized.mp4"
    temp_output = tmp_path / ".normalized.normalized.mp4"
    monkeypatch.setattr(
        composer,
        "probe_video",
        lambda _: _video_info(Fraction(24), 121),
    )
    monkeypatch.setattr(composer, "get_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        composer.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, "ffmpeg")
        ),
    )

    with pytest.raises(RuntimeError, match="FFmpeg failed to normalize"):
        composer.normalize_video_clip(
            source,
            output,
            target_frame_rate=Fraction(24),
            target_frame_count=120,
        )
    assert not temp_output.exists()
    assert not output.exists()


def test_probe_video_parses_fractional_metadata(monkeypatch, tmp_path):
    source = tmp_path / "scene.mp4"
    source.write_bytes(b"video")
    response = {
        "streams": [
            {
                "codec_type": "video",
                "avg_frame_rate": "30000/1001",
                "nb_read_frames": "301",
                "duration": "10.043367",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "pix_fmt": "yuv420p",
            }
        ],
        "format": {},
    }

    class FakeCompletedProcess:
        stdout = json.dumps(response)

    monkeypatch.setattr(composer, "get_ffprobe", lambda: "ffprobe")
    monkeypatch.setattr(
        composer.subprocess,
        "run",
        lambda _command, **_kwargs: FakeCompletedProcess(),
    )

    info = composer.probe_video(source)
    assert info.frame_rate == Fraction(30000, 1001)
    assert info.frame_count == 301


def test_validated_concatenation_uses_stream_copy_and_checks_total(
    monkeypatch,
    tmp_path,
):
    clips = [tmp_path / "one.mp4", tmp_path / "two.mp4"]
    for clip in clips:
        clip.write_bytes(b"clip")
    output = tmp_path / "joined.mp4"
    infos = iter(
        [
            _video_info(Fraction(24), 240),
            _video_info(Fraction(24), 120),
            _video_info(Fraction(24), 360),
        ]
    )
    command_holder = {}

    def fake_run(command, check):
        assert check is True
        command_holder["command"] = command
        Path(command[-1]).write_bytes(b"joined")

    monkeypatch.setattr(composer, "get_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(composer, "probe_video", lambda _: next(infos))
    monkeypatch.setattr(composer.subprocess, "run", fake_run)

    result = composer.concatenate_clips(
        [str(clip) for clip in clips],
        str(output),
        scene_durations_seconds=[10, 5],
        frame_rate=Fraction(24),
    )

    command = command_holder["command"]
    assert command[command.index("-c") + 1] == "copy"
    assert "-c:v" not in command
    assert result == str(output)
    assert output.read_bytes() == b"joined"


def test_validated_concatenation_rejects_short_or_failed_join(
    monkeypatch,
    tmp_path,
):
    clip = tmp_path / "scene.mp4"
    clip.write_bytes(b"clip")
    output = tmp_path / "joined.mp4"
    monkeypatch.setattr(composer, "get_ffmpeg", lambda: "ffmpeg")

    with pytest.raises(ValueError, match="not normalized"):
        monkeypatch.setattr(
            composer,
            "probe_video",
            lambda _: _video_info(Fraction(24), 119),
        )
        composer.concatenate_clips(
            [str(clip)],
            str(output),
            scene_durations_seconds=[5],
            frame_rate=Fraction(24),
        )

    monkeypatch.setattr(
        composer,
        "probe_video",
        lambda _: _video_info(Fraction(24), 120),
    )
    monkeypatch.setattr(
        composer.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, "ffmpeg")
        ),
    )
    with pytest.raises(RuntimeError, match="FFmpeg failed to concatenate"):
        composer.concatenate_clips(
            [str(clip)],
            str(output),
            scene_durations_seconds=[5],
            frame_rate=Fraction(24),
        )
    assert not (tmp_path / ".joined.joined.mp4").exists()


def _video_info(frame_rate: Fraction, frame_count: int):
    return composer.VideoInfo(
        frame_rate=frame_rate,
        frame_count=frame_count,
        duration_seconds=Fraction(frame_count, 1) / frame_rate,
        video_codec="h264",
        width=1280,
        height=720,
        pixel_format="yuv420p",
        audio_codec=None,
        audio_sample_rate=None,
        audio_channels=None,
    )
