import json
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image

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


def test_motion_prompt_fails_instead_of_truncating_required_action(
    monkeypatch,
):
    storyboard = _circuit_storyboard()
    scene = storyboard.scenes[0]
    scene.action = "perform " + "critical action " * 300
    scene.negative_prompt = (
        "No camera movement; no cuts; no extra filters."
    )
    monkeypatch.setattr(pipeline, "VIDEO_PROVIDER", "runway")
    with pytest.raises(
        ValueError,
        match="Required Runway motion details exceed the provider limit",
    ):
        pipeline._build_motion_prompt(storyboard, scene)


def test_grok_prompt_compiler_uses_its_verified_model_limit(monkeypatch):
    storyboard = _circuit_storyboard()
    scene = storyboard.scenes[0]
    scene.action = "Continue the movement " * 70
    monkeypatch.setattr(pipeline, "VIDEO_PROVIDER", "runway")
    monkeypatch.setattr(pipeline, "RUNWAY_MODEL", "grok_imagine_1_5")

    prompt = pipeline._build_motion_prompt(storyboard, scene)

    assert scene.action in prompt
    assert len(prompt.encode("utf-16-le")) // 2 <= 2500
    assert len(prompt.encode("utf-16-le")) // 2 > 1000


def test_motion_prompt_preserves_complete_core_and_priority_negatives(
    monkeypatch,
    caplog,
):
    storyboard = _circuit_storyboard()
    scene = storyboard.scenes[0]
    scene.negative_prompt = (
        "No camera cuts; no reverse flow; no extra switches; no text."
    )
    scene.visual_description = "Detailed visual context " * 250
    monkeypatch.setattr(pipeline, "VIDEO_PROVIDER", "runway")
    caplog.set_level("INFO", logger="app.video.pipeline")

    prompt = pipeline._build_motion_prompt(storyboard, scene)

    assert len(prompt.encode("utf-16-le")) // 2 <= 1000
    assert scene.continuity.required_start_state in prompt
    assert scene.action in prompt
    assert scene.continuity.ending_state in prompt
    assert scene.camera.movement in prompt
    assert "No camera cuts" in prompt
    assert "no reverse flow" in prompt
    assert any(
        record.message == "Optional Runway prompt context omitted to preserve "
        "required subject, action, state, and camera details"
        for record in caplog.records
    )


def test_later_scenes_use_verified_predecessor_video_frame_end_to_end(
    monkeypatch,
    tmp_path,
):
    storyboard = _circuit_storyboard()
    storyboard.project.duration_seconds = 15
    scene_three = storyboard.scenes[1].model_copy(deep=True)
    scene_three.id = "SCENE_003"
    scene_three.continuity.previous_scene_id = "SCENE_002"
    scene_three.continuity.required_start_state = (
        "The switch remains closed and the bulb remains lit."
    )
    scene_three.continuity.ending_state = (
        "The circuit remains closed and the bulb remains lit."
    )
    scene_three.continuity.character_state = (
        "The switch is closed and the bulb is lit."
    )
    storyboard.scenes.append(scene_three)
    extracted_sources = []
    img2img_handoffs = []
    img2img_strengths = []
    video_first_frames = []

    def fake_generate_image(**kwargs):
        Image.new("RGB", (32, 24), (1, 1, 1)).save(kwargs["output_path"])

    def fake_extract_last_frames(video_path, output_dir, frame_count=5):
        source = Path(video_path)
        assert source.read_bytes().startswith(b"normalized:")
        extracted_sources.append(source)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        reference = output / "frame_000119.png"
        color = (len(extracted_sources), 10, 20)
        Image.new("RGB", (32, 24), color).save(reference)
        return [str(reference)]

    def fake_generate_from_reference(**kwargs):
        reference = Path(kwargs["reference_image_path"])
        img2img_strengths.append(kwargs["strength"])
        with Image.open(reference) as image:
            image.load()
            img2img_handoffs.append((reference, image.getpixel((0, 0))))
        Image.new(
            "RGB",
            (32, 24),
            (20 + len(img2img_handoffs), 30, 40),
        ).save(kwargs["output_path"])
        return kwargs["output_path"]

    def fake_generate_video(**kwargs):
        image_path = Path(kwargs["image_path"])
        with Image.open(image_path) as image:
            image.load()
            video_first_frames.append(
                (image_path, image.getpixel((0, 0)))
            )
        output_path = Path(kwargs["output_path"])
        output_path.write_bytes(
            f"normalized:{output_path.stem}".encode("utf-8")
        )
        return str(output_path)

    def fake_normalize(source, output, **kwargs):
        Path(output).write_bytes(b"normalized:scene_001")

    def fake_probe(_):
        return VideoInfo(
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
        )

    def fake_concatenate(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"final")

    monkeypatch.setattr(pipeline, "VIDEO_PROVIDER", "runway")
    monkeypatch.setattr(pipeline, "generate_scene_image", fake_generate_image)
    monkeypatch.setattr(
        pipeline,
        "generate_scene_image_from_reference",
        fake_generate_from_reference,
    )
    monkeypatch.setattr(
        pipeline,
        "extract_last_frames",
        fake_extract_last_frames,
    )
    monkeypatch.setattr(
        pipeline,
        "generate_scene_video",
        fake_generate_video,
    )
    monkeypatch.setattr(pipeline, "probe_video", fake_probe)
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

    scene_one_video = tmp_path / "scene_001.mp4"
    scene_two_video = tmp_path / "scene_002.mp4"
    assert extracted_sources == [scene_one_video, scene_two_video]
    assert [path.read_bytes() for path in extracted_sources] == [
        b"normalized:scene_001",
        b"normalized:scene_002",
    ]
    assert [pixel for _, pixel in img2img_handoffs] == [
        (1, 10, 20),
        (2, 10, 20),
    ]
    assert [path for path, _ in img2img_handoffs] == [
        tmp_path / "scene_001_end_frames" / "frame_000119.png",
        tmp_path / "scene_002_end_frames" / "frame_000119.png",
    ]
    assert [path for path, _ in video_first_frames] == [
        tmp_path / "scene_001.png",
        tmp_path / "scene_002.png",
        tmp_path / "scene_003.png",
    ]
    assert [pixel for _, pixel in video_first_frames] == [
        (1, 1, 1),
        (21, 30, 40),
        (22, 30, 40),
    ]
    assert img2img_strengths == [pipeline.IMAGE_TO_IMAGE_STRENGTH] * 2
    scene_two_diagnostics = json.loads(
        (tmp_path / "scene_002_generation.json").read_text(encoding="utf-8")
    )
    assert scene_two_diagnostics["video_reference_images_submitted"] == []
    assert scene_two_diagnostics["reference_source_video_path"] == str(
        scene_one_video
    )
    assert scene_two_diagnostics["reference_frame_used_for_img2img"] == str(
        tmp_path / "scene_001_end_frames" / "frame_000119.png"
    )
    assert scene_two_diagnostics["continuity_image_generation_mode"] == "img2img"
    assert scene_two_diagnostics["first_frame_image_path"] == str(
        tmp_path / "scene_002.png"
    )
    boundary = scene_two_diagnostics["continuity_boundary"]
    assert boundary["status"] == "completed"
    assert boundary["previous_scene_id"] == "SCENE_001"
    assert boundary["next_scene_id"] == "SCENE_002"
    assert boundary["candidate_frames"] == [
        {
            "path": str(
                tmp_path / "scene_001_end_frames" / "frame_000119.png"
            ),
            "frame_index": 119,
            "timestamp_seconds": pytest.approx(119 / 24),
        }
    ]
    assert boundary["selected_frame"]["path"] == str(
        tmp_path / "scene_001_end_frames" / "frame_000119.png"
    )
    assert boundary["source_video_frame_rate"] == "24"
    assert boundary["candidate_timestamp_basis"] == (
        "frame_index / normalized CFR frame rate"
    )
    assert boundary["img2img_input_path"] == str(
        tmp_path / "scene_001_end_frames" / "frame_000119.png"
    )
    assert boundary["img2img_output_path"] == str(tmp_path / "scene_002.png")
    assert boundary["provider_input_image_path"] == str(
        tmp_path / "scene_002.png"
    )
    assert boundary["img2img_strength"] == pipeline.IMAGE_TO_IMAGE_STRENGTH
    assert scene_two_diagnostics["video_prompt_character_count"] <= (
        scene_two_diagnostics["video_prompt_character_limit"]
    )
    assert scene_two_diagnostics["video_prompt_character_count_unit"] == (
        "UTF-16 code units"
    )
    assert scene_two_diagnostics["transition_strategy"] == "hard_cut"
    assert final_path.read_bytes() == b"final"
