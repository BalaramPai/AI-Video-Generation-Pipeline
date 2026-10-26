import json
import sys

import pytest

from app import main
from app.storyboard.io import parse_storyboard_json
from app.video import pipeline
from tests.test_image_prompt import _circuit_storyboard
from tests.test_storyboard_io import _storyboard_data


def _three_scene_storyboard():
    storyboard = _circuit_storyboard()
    first_scene = storyboard.scenes[0]
    second_scene = storyboard.scenes[1]
    second_scene.id = "SCENE_002"
    second_scene.previous_scene_id = first_scene.id
    second_scene.continuity.previous_scene_id = first_scene.id
    second_scene.continuity.required_start_state = (
        first_scene.continuity.ending_state
    )
    second_scene.continuity.ending_state = (
        "The switch is closed and bulb is lit."
    )
    third_scene = second_scene.model_copy(deep=True)
    third_scene.id = "SCENE_003"
    third_scene.previous_scene_id = second_scene.id
    third_scene.continuity.previous_scene_id = second_scene.id
    third_scene.continuity.required_start_state = (
        second_scene.continuity.ending_state
    )
    third_scene.continuity.ending_state = (
        "The complete circuit remains visible."
    )
    storyboard.scenes = [first_scene, second_scene, third_scene]
    storyboard.project.duration_seconds = 15
    return storyboard


@pytest.mark.parametrize("scene_index", [1, 2], ids=["scene-2", "scene-3"])
def test_later_scene_prompt_overflow_blocks_all_generation(
    monkeypatch,
    tmp_path,
    scene_index,
):
    storyboard = _three_scene_storyboard()
    invalid_scene = storyboard.scenes[scene_index]
    invalid_scene.action = "critical-action " * 300
    generation_calls = []
    monkeypatch.setattr(
        pipeline,
        "generate_scene_image",
        lambda **kwargs: generation_calls.append(("image", kwargs)),
    )
    monkeypatch.setattr(
        pipeline,
        "generate_scene_video",
        lambda **kwargs: generation_calls.append(("video", kwargs)),
    )

    with pytest.raises(pipeline.PreflightValidationError) as exc_info:
        pipeline.generate_video_from_storyboard(
            storyboard,
            output_dir=tmp_path,
        )

    assert generation_calls == []
    assert list(tmp_path.iterdir()) == []
    assert any(
        error.scene_id == invalid_scene.id
        and error.actual_length > error.limit
        and "action" in error.required_fields
        for error in exc_info.value.prompt_errors
    )
    assert invalid_scene.id in str(exc_info.value)


def test_preflight_aggregates_motion_prompt_overflows_with_measurements():
    storyboard = _three_scene_storyboard()
    storyboard.scenes[1].action = "scene-two-action " * 300
    storyboard.scenes[2].action = "scene-three-action " * 300

    with pytest.raises(pipeline.PreflightValidationError) as exc_info:
        pipeline.preflight_storyboard(storyboard)

    errors = exc_info.value.prompt_errors
    assert [error.scene_id for error in errors] == [
        "SCENE_002",
        "SCENE_003",
    ]
    assert all(error.actual_length > error.limit == 1000 for error in errors)
    assert all(error.length_unit == "UTF-16 code units" for error in errors)
    assert all("action" in error.required_fields for error in errors)
    assert "actual prompt length" in str(exc_info.value).lower()


def test_preflight_checks_initial_image_prompt_token_count(monkeypatch):
    storyboard = _three_scene_storyboard()
    monkeypatch.setattr(
        pipeline,
        "build_initial_scene_image_prompt",
        lambda *_args: "oversized image prompt " * 80,
    )

    with pytest.raises(pipeline.PreflightValidationError) as exc_info:
        pipeline.preflight_storyboard(storyboard)

    image_errors = [
        error for error in exc_info.value.prompt_errors
        if error.prompt_type == "initial image prompt"
    ]
    assert len(image_errors) == 1
    assert image_errors[0].scene_id == storyboard.scenes[0].id
    assert image_errors[0].actual_length > image_errors[0].limit == 75
    assert image_errors[0].length_unit == "tokens"


def test_preflight_uses_selected_runway_model_utf16_limit(monkeypatch):
    storyboard = _three_scene_storyboard()
    storyboard.scenes[0].action = "Continue the movement " * 70
    monkeypatch.setattr(pipeline, "VIDEO_PROVIDER", "runway")
    monkeypatch.setattr(pipeline, "RUNWAY_MODEL", "grok_imagine_1_5")

    report = pipeline.preflight_storyboard(storyboard)

    diagnostic = report.motion_prompt_diagnostics[0]
    assert diagnostic["video_prompt_character_count"] > 1000
    assert diagnostic["video_prompt_character_count"] <= 2500
    assert diagnostic["video_prompt_character_limit"] == 2500
    assert diagnostic["video_prompt_character_count_unit"] == (
        "UTF-16 code units"
    )


def test_preflight_cli_validates_and_exits_without_generation(
    monkeypatch,
    capsys,
):
    storyboard = parse_storyboard_json(json.dumps(_storyboard_data()))
    monkeypatch.setattr(sys, "argv", ["app.main", "--preflight"])
    monkeypatch.setattr(main, "_load_input_storyboard", lambda _: storyboard)
    monkeypatch.setattr(
        main,
        "generate_video_from_storyboard",
        lambda *_args, **_kwargs: pytest.fail(
            "Preflight must not run generation."
        ),
    )

    main.main()

    output = capsys.readouterr().out
    assert "Preflight passed: 1 scene;" in output
    assert "initial image prompt" in output
    assert "SCENE_001: motion prompt" in output
