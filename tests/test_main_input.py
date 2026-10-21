import json
import sys

import pytest

from app import main
from tests.test_storyboard_io import _storyboard_data


def test_load_input_storyboard_from_path(tmp_path):
    path = tmp_path / "storyboard.json"
    path.write_text(json.dumps(_storyboard_data()), encoding="utf-8")

    storyboard = main._load_input_storyboard(str(path))

    assert storyboard.project.title == "Engine explainer"


def test_load_input_storyboard_from_pasted_json(monkeypatch):
    json_lines = json.dumps(_storyboard_data(), indent=2).splitlines()
    lines = iter(["", *json_lines, "END"])
    prompts = []
    monkeypatch.setattr(
        "builtins.input",
        lambda *args: (prompts.append(args), next(lines))[1],
    )

    storyboard = main._load_input_storyboard(None)

    assert storyboard.project.title == "Engine explainer"
    assert prompts[0][0].startswith("Storyboard JSON file path")


def test_pasted_storyboard_requires_content(monkeypatch):
    lines = iter(["END"])
    monkeypatch.setattr(
        "builtins.input",
        lambda *args: (tuple(args), next(lines))[1],
    )

    with pytest.raises(ValueError, match="No storyboard JSON"):
        main._read_pasted_storyboard()


def test_cli_reports_interrupted_generation_and_saved_task(
    monkeypatch,
    tmp_path,
    capsys,
):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "scene_001.runway-task.json").write_text(
        "{}",
        encoding="utf-8",
    )

    def interrupt_generation(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(sys, "argv", ["app.main"])
    monkeypatch.setattr(
        main,
        "_load_input_storyboard",
        lambda _: _storyboard_data(),
    )
    monkeypatch.setattr(
        main,
        "generate_video_from_storyboard",
        interrupt_generation,
    )
    monkeypatch.chdir(tmp_path)

    main.main()

    captured = capsys.readouterr()
    assert "Generation interrupted" in captured.err
    assert "resume without submitting another" in captured.err
    assert "scene_001.runway-task.json" in captured.err
