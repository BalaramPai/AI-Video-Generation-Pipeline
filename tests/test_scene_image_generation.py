from pathlib import Path

import app.assets.image_generator as image_generator
from tests.test_image_prompt import _circuit_storyboard


def test_generate_scene_image_uses_built_storyboard_prompt(
    monkeypatch,
    tmp_path,
):
    storyboard = _circuit_storyboard()
    scene = storyboard.scenes[0]
    captured = {}
    output_path = tmp_path / "scene.png"

    def fake_generate_scene_image(**kwargs):
        captured.update(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"image")
        return kwargs["output_path"]

    monkeypatch.setattr(
        image_generator,
        "generate_scene_image",
        fake_generate_scene_image,
    )
    prompt = image_generator.build_scene_image_prompt(storyboard, scene)

    result = image_generator.generate_scene_image(
        prompt=prompt,
        output_path=str(output_path),
    )

    assert result == str(output_path)
    assert output_path.is_file()
    assert captured["prompt"] == prompt
    assert "open switch" in prompt.lower()
    assert "bulb stays unlit" in prompt.lower()
