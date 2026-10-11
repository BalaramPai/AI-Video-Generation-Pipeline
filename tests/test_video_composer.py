from pathlib import Path

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
