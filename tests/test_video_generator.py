from fractions import Fraction
import logging

import pytest
from PIL import Image

import app.video.generator as video_generator
import app.video.providers.factory as video_provider_factory
from app.assets.image_generator import TOKENIZER, build_scene_image_prompt
from app.storyboard.models import (
    Camera,
    Character,
    CharacterAppearance,
    Continuity,
    Location,
    Project,
    Prop,
    Scene,
    Storyboard,
)
from app.video.providers import (
    VideoGenerationResult,
    VideoProviderCapabilities,
)
from app.video.generator import (
    generate_scene_video,
)
from app.video import composer


def _write_test_image(path, color=(1, 2, 3)):
    Image.new("RGB", (16, 12), color).save(path)


def test_build_scene_image_prompt_stays_under_sd15_token_limit():
    storyboard = Storyboard(
        project=Project(
            title="Coffee Shop",
            description="A cinematic coffee shop scene.",
            duration_seconds=15,
            aspect_ratio="16:9",
            visual_style="cinematic realism",
        ),
        characters=[
            Character(
                id="CHAR_001",
                name="Daniel",
                appearance=CharacterAppearance(
                    age=28,
                    hair="short dark brown hair",
                    clothing="dark blue denim jacket over a white shirt",
                    facial_features="oval clean-shaven face, brown eyes",
                    physical_description="average build",
                    eye_color="brown",
                    distinctive_features="clean-shaven",
                    wardrobe_anchor="dark blue denim jacket",
                ),
            ),
            Character(
                id="CHAR_002",
                name="Maya",
                appearance=CharacterAppearance(
                    age=26,
                    hair="dark shoulder-length hair",
                    clothing="white apron over black shirt",
                    facial_features="warm smiling face",
                    physical_description="slim build",
                    eye_color="brown",
                    distinctive_features="friendly smile",
                    wardrobe_anchor="white apron",
                ),
            ),
        ],
        locations=[
            Location(
                id="LOC_001",
                name="Modern Coffee Shop",
                description="Warm neighborhood cafe",
                lighting="soft warm natural light",
                visual_style="cinematic realism",
                continuity_anchor="same counter, window, and seating layout throughout",
            )
        ],
        props=[
            Prop(
                id="PROP_001",
                name="Coffee Cup",
                description="white ceramic coffee cup",
                continuity_anchor="ceramic latte cup held in Daniel's hand",
            )
        ],
        scenes=[
            Scene(
                id="SCENE_001",
                duration_seconds=5,
                location_id="LOC_001",
                character_ids=["CHAR_001"],
                prop_ids=["PROP_001"],
                action="Daniel receives a coffee.",
                dialogue=None,
                visual_description="Daniel stands at the counter holding coffee.",
                camera=Camera(
                    shot_type="medium close-up",
                    angle="eye level",
                    movement="slow push-in",
                    lens="35mm",
                ),
                continuity=Continuity(
                    previous_scene_id=None,
                    character_state="Daniel stands at the counter while Maya serves him",
                    environment_state="warm coffee shop interior with the same counter and window layout",
                    required_start_state="Daniel stands at the counter.",
                    ending_state="Daniel receives his coffee and holds it at the counter",
                ),
            )
        ],
    )

    prompt = build_scene_image_prompt(storyboard, storyboard.scenes[0])
    token_count = TOKENIZER(
        prompt,
        truncation=False,
        return_tensors="pt",
        verbose=False,
    )["input_ids"].shape[1]

    assert token_count <= 75
    assert "daniel" in prompt.lower()


def test_generate_scene_video_requires_gemini_key(monkeypatch, tmp_path):
    image_path = tmp_path / "scene.png"
    _write_test_image(image_path)

    monkeypatch.setattr(video_generator, "VIDEO_PROVIDER", "veo")
    monkeypatch.setattr(video_provider_factory, "GEMINI_API_KEY", None)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        generate_scene_video(
            image_path=str(image_path),
            prompt="A young man walks toward the coffee counter.",
            output_path=str(tmp_path / "scene.mp4"),
            duration_seconds=5,
        )


def test_generate_scene_video_normalizes_provider_duration(
    monkeypatch,
    tmp_path,
):
    image_path = tmp_path / "scene.png"
    _write_test_image(image_path)
    output_path = tmp_path / "scene.mp4"

    class FakeProvider:
        name = "veo"
        capabilities = VideoProviderCapabilities(
            text_to_video=False,
            image_to_video=True,
            reference_images=True,
            first_frame=True,
            last_frame=True,
            video_extension=False,
            supported_aspect_ratios=("16:9", "9:16"),
        )

        def __init__(self, provider_name):
            assert provider_name == "veo"

        def generate_video(self, **kwargs):
            assert kwargs["first_frame"] == image_path
            assert kwargs["aspect_ratio"] == "9:16"
            return VideoGenerationResult(
                video_path=output_path,
                provider="veo",
                generation_id="operations/test",
                duration_seconds=6,
            )

    normalization_calls = []
    generation_results = []
    monkeypatch.setattr(video_generator, "VIDEO_PROVIDER", "veo")
    monkeypatch.setattr(video_generator, "create_video_provider", FakeProvider)
    monkeypatch.setattr(
        video_generator,
        "probe_video",
        lambda _: composer.VideoInfo(
            frame_rate=Fraction(24, 1),
            frame_count=144,
            duration_seconds=Fraction(6, 1),
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
        video_generator,
        "normalize_video_clip",
        lambda *args, **kwargs: normalization_calls.append((args, kwargs)),
    )

    video_generator.generate_scene_video(
        image_path=str(image_path),
        prompt="A young man walks toward the coffee counter.",
        output_path=str(output_path),
        duration_seconds=5,
        aspect_ratio="9:16",
        on_generation_result=lambda result, first_frame: (
            generation_results.append((result, first_frame))
        ),
    )

    assert len(normalization_calls) == 1
    assert generation_results[0][0].generation_id == "operations/test"
    assert generation_results[0][1] == image_path
    assert normalization_calls[0][1] == {
        "target_frame_rate": Fraction(24, 1),
        "target_frame_count": 120,
    }


def test_generate_scene_video_uses_first_frame_when_provider_lacks_references(
    monkeypatch,
    tmp_path,
    caplog,
):
    image_path = tmp_path / "scene.png"
    _write_test_image(image_path)
    reference_path = tmp_path / "previous.png"
    _write_test_image(reference_path, (4, 5, 6))
    output_path = tmp_path / "scene.mp4"

    class FakeProvider:
        name = "runway"
        capabilities = VideoProviderCapabilities(
            text_to_video=False,
            image_to_video=True,
            reference_images=False,
            first_frame=True,
            last_frame=False,
            video_extension=False,
            supported_aspect_ratios=("16:9",),
        )

        def __init__(self, provider_name):
            assert provider_name == "runway"

        def generate_video(self, **kwargs):
            assert kwargs["first_frame"] == image_path
            assert kwargs["reference_images"] == ()
            output_path.write_bytes(b"video")
            return VideoGenerationResult(
                video_path=output_path,
                provider="runway",
                generation_id="task-123",
                duration_seconds=5,
            )

    monkeypatch.setattr(video_generator, "VIDEO_PROVIDER", "runway")
    monkeypatch.setattr(video_generator, "create_video_provider", FakeProvider)
    monkeypatch.setattr(
        video_generator,
        "probe_video",
        lambda _: composer.VideoInfo(
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
        video_generator,
        "normalize_video_clip",
        lambda *_args, **_kwargs: None,
    )

    with caplog.at_level(logging.INFO):
        assert video_generator.generate_scene_video(
            image_path=str(image_path),
            prompt="Continue the action.",
            output_path=str(output_path),
            reference_images=(str(reference_path),),
        ) == str(output_path)
    capability_log = next(
        record
        for record in caplog.records
        if "does not accept separate reference images" in record.message
    )
    assert capability_log.first_frame == str(image_path)
    assert capability_log.reference_images_submitted == []
    assert capability_log.first_frame_is_uploaded_conditioning_image is True


def test_generate_scene_video_rejects_corrupt_first_frame_before_provider(
    monkeypatch,
    tmp_path,
):
    image_path = tmp_path / "corrupt.png"
    image_path.write_bytes(b"not a png")
    monkeypatch.setattr(video_generator, "VIDEO_PROVIDER", "runway")
    monkeypatch.setattr(
        video_generator,
        "create_video_provider",
        lambda *_: pytest.fail("Provider must not receive an invalid image"),
    )
    with pytest.raises(ValueError, match="not a readable image"):
        generate_scene_video(
            image_path=str(image_path),
            prompt="Continue the scene.",
            output_path=str(tmp_path / "scene.mp4"),
        )


def test_generate_scene_video_reports_uploaded_first_frame_to_provider(
    monkeypatch,
    tmp_path,
):
    image_path = tmp_path / "scene.png"
    reference_path = tmp_path / "reference.png"
    output_path = tmp_path / "scene.mp4"
    _write_test_image(image_path)
    _write_test_image(reference_path, (9, 8, 7))

    class FakeProvider:
        name = "veo"
        capabilities = VideoProviderCapabilities(
            text_to_video=False,
            image_to_video=True,
            reference_images=True,
            first_frame=True,
            last_frame=True,
            video_extension=False,
            supported_aspect_ratios=("16:9",),
        )

        def __init__(self, _provider_name):
            pass

        def generate_video(self, **kwargs):
            assert kwargs["first_frame"] == image_path
            assert kwargs["reference_images"] == (reference_path,)
            output_path.write_bytes(b"video")
            return VideoGenerationResult(
                video_path=output_path,
                provider="veo",
                generation_id="operations/reference-test",
                duration_seconds=5,
            )

    monkeypatch.setattr(video_generator, "VIDEO_PROVIDER", "veo")
    monkeypatch.setattr(video_generator, "create_video_provider", FakeProvider)
    monkeypatch.setattr(
        video_generator,
        "probe_video",
        lambda _: composer.VideoInfo(
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
        video_generator,
        "normalize_video_clip",
        lambda *_args, **_kwargs: None,
    )

    result = []
    video_generator.generate_scene_video(
        image_path=str(image_path),
        prompt="Continue the scene.",
        output_path=str(output_path),
        reference_images=(str(reference_path),),
        on_generation_result=lambda generation, first_frame: result.append(
            (generation, first_frame)
        ),
    )

    assert result[0][0].generation_id == "operations/reference-test"
    assert result[0][1] == image_path


def test_trim_video_writes_normalized_output(monkeypatch, tmp_path):
    input_path = tmp_path / "generated.mp4"
    input_path.write_bytes(b"generated data")
    output_path = tmp_path / "scene.mp4"
    normalization_calls = []
    source_info = composer.VideoInfo(
        frame_rate=Fraction(24, 1),
        frame_count=144,
        duration_seconds=Fraction(6, 1),
        video_codec="h264",
        width=1280,
        height=720,
        pixel_format="yuv420p",
        audio_codec=None,
        audio_sample_rate=None,
        audio_channels=None,
    )
    monkeypatch.setattr(composer, "probe_video", lambda _: source_info)
    monkeypatch.setattr(
        composer,
        "normalize_video_clip",
        lambda *args, **kwargs: normalization_calls.append((args, kwargs)),
    )

    assert composer.trim_video(
        input_path=str(input_path),
        output_path=str(output_path),
        duration_seconds=5,
    ) == str(output_path)
    assert normalization_calls[0][1] == {
        "target_frame_rate": Fraction(24, 1),
        "target_frame_count": 120,
    }
