from PIL import Image
import pytest

import app.assets.image_generator as image_generator
from tests.test_image_prompt import _circuit_storyboard


def test_generate_scene2_from_continuity_uses_reference_and_prompt(
    monkeypatch,
    tmp_path,
):
    reference_path = tmp_path / "reference.png"
    Image.new("RGB", (32, 32)).save(reference_path)
    captured = {}

    class FakePipeline:
        tokenizer = image_generator.TOKENIZER

        def enable_model_cpu_offload(self):
            pass

        def __call__(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Result",
                (),
                {"images": [Image.new("RGB", (16, 16))]},
            )()

    class FakeImg2ImgPipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return FakePipeline()

    import diffusers

    monkeypatch.setattr(
        diffusers,
        "StableDiffusionImg2ImgPipeline",
        FakeImg2ImgPipeline,
    )
    storyboard = _circuit_storyboard()
    scene = storyboard.scenes[1]
    prompt = image_generator.build_continuity_image_prompt(
        storyboard=storyboard,
        scene=scene,
        reference_frames=[str(reference_path)],
    )
    output_path = tmp_path / "scene2.png"

    result = image_generator.generate_scene_image_from_reference(
        prompt=prompt,
        reference_image_path=str(reference_path),
        output_path=str(output_path),
        strength=0.47,
    )

    assert output_path.is_file()
    assert result == str(output_path)
    assert captured["prompt"] == prompt
    assert captured["image"].size == (32, 32)
    assert captured["strength"] == 0.47
    assert "same battery, wire, open switch, unlit bulb" in prompt.lower()
    assert "close switch; bulb lights" in prompt.lower()


def test_generate_scene2_rejects_out_of_range_img2img_strength(tmp_path):
    reference_path = tmp_path / "reference.png"
    Image.new("RGB", (16, 16)).save(reference_path)

    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        image_generator.generate_scene_image_from_reference(
            prompt="Keep the same scene.",
            reference_image_path=str(reference_path),
            output_path=str(tmp_path / "scene.png"),
            strength=1.1,
        )
