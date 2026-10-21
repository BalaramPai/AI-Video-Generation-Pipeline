import json
from fractions import Fraction
from pathlib import Path

import pytest

from app.storyboard.io import parse_storyboard_json
from app.video import pipeline
from app.video.composer import VideoInfo
from tests.test_image_prompt import _circuit_storyboard
from tests.test_storyboard_io import _storyboard_data


def test_generate_video_from_storyboard_runs_without_storyboard_generation(
    monkeypatch,
    tmp_path,
):
    storyboard = parse_storyboard_json(
        json.dumps(_storyboard_data())
    )
    calls = {}

    def fake_generate_image(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"image")
        calls["image_prompt"] = kwargs["prompt"]

    def fake_generate_video(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"video")
        calls["video_prompt"] = kwargs["prompt"]
        calls["duration"] = kwargs["duration_seconds"]
        calls["target_frame_rate"] = kwargs["target_frame_rate"]
        calls["target_frame_count"] = kwargs["target_frame_count"]
        calls["normalize_output"] = kwargs["normalize_output"]
        return kwargs["output_path"]

    def fake_normalize(source, output, **kwargs):
        Path(output).write_bytes(b"normalized video")
        calls["normalization_source"] = source
        calls["normalized_frame_count"] = kwargs["target_frame_count"]

    def fake_concatenate(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"final video")
        calls["join_durations"] = kwargs["scene_durations_seconds"]
        calls["join_frame_rate"] = kwargs["frame_rate"]

    monkeypatch.setattr(pipeline, "generate_scene_image", fake_generate_image)
    monkeypatch.setattr(pipeline, "generate_scene_video", fake_generate_video)
    monkeypatch.setattr(
        pipeline,
        "probe_video",
        lambda _: VideoInfo(
            frame_rate=Fraction(24, 1),
            frame_count=120,
            duration_seconds=Fraction(5, 1),
            video_codec="h264",
            width=1280,
            height=720,
            pixel_format="yuv420p",
            audio_codec=None,
            audio_sample_rate=None,
            audio_channels=None,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "normalize_video_clip",
        fake_normalize,
    )
    monkeypatch.setattr(pipeline, "concatenate_clips", fake_concatenate)

    final_path = pipeline.generate_video_from_storyboard(
        storyboard,
        output_dir=tmp_path,
    )

    assert final_path == tmp_path / "final_video.mp4"
    assert final_path.read_bytes() == b"final video"
    assert "engine" in calls["image_prompt"].lower()
    assert "Subject lock" in calls["video_prompt"]
    assert calls["duration"] == 5
    assert calls["target_frame_rate"] is None
    assert calls["target_frame_count"] is None
    assert calls["normalize_output"] is False
    assert calls["normalization_source"] == str(tmp_path / "scene_001.mp4")
    assert calls["normalized_frame_count"] == 120
    assert calls["join_durations"] == [5]
    assert calls["join_frame_rate"] == Fraction(24, 1)
    assert (tmp_path / "scene_001_image_prompt.txt").is_file()
    assert (tmp_path / "scene_001_video_prompt.txt").is_file()
    diagnostics = json.loads(
        (tmp_path / "scene_001_generation.json").read_text(encoding="utf-8")
    )
    assert diagnostics["scene_id"] == "SCENE_001"
    assert diagnostics["provider"] == pipeline.VIDEO_PROVIDER
    assert diagnostics["model"]
    assert diagnostics["image_prompt"]
    assert diagnostics["video_prompt"]
    assert diagnostics["generated_duration_seconds"] == "5"
    assert diagnostics["audio_generation"] == (
        "not supported by this video pipeline"
    )


def test_motion_prompt_preserves_required_scene_content_before_optional_style(
    monkeypatch,
):
    storyboard = _circuit_storyboard()
    storyboard.project.visual_style = "educational style " * 300
    scene = storyboard.scenes[0]
    monkeypatch.setattr(pipeline, "VIDEO_PROVIDER", "runway")

    prompt = pipeline._build_motion_prompt(storyboard, scene)

    assert scene.continuity.required_start_state in prompt
    assert scene.action in prompt
    assert scene.continuity.ending_state in prompt
    assert scene.visual_description in prompt
    assert scene.camera.shot_type in prompt
    assert len(prompt.encode("utf-16-le")) // 2 <= 1000
    assert "educational style educational style" not in prompt


def test_motion_prompt_fits_overlong_required_content_and_warns(
    monkeypatch,
    caplog,
):
    storyboard = _circuit_storyboard()
    scene = storyboard.scenes[0]
    scene.action = "perform " + "critical action " * 300
    monkeypatch.setattr(pipeline, "VIDEO_PROVIDER", "runway")

    prompt = pipeline._build_motion_prompt(storyboard, scene)

    assert len(prompt.encode("utf-16-le")) // 2 <= 1000
    assert "Starting state:" in prompt
    assert "Action:" in prompt
    assert "Ending state:" in prompt
    assert "Visual:" in prompt
    assert "Camera:" in prompt
    assert any(
        "shortened" in record.message.lower()
        for record in caplog.records
    )
