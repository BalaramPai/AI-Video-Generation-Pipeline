import json
import logging
from fractions import Fraction
from pathlib import Path

from app.assets.image_generator import (
    build_scene_negative_prompt,
    build_continuity_image_prompt,
    build_scene_image_prompt,
    generate_scene_image,
    generate_scene_image_from_reference,
)
from app.config import (
    RUNWAY_MODEL,
    VEO_MODEL,
    VIDEO_PROVIDER,
)
from app.storyboard.generator import validate_storyboard_continuity
from app.storyboard.models import Scene, Storyboard
from app.video.composer import (
    calculate_scene_frame_counts,
    concatenate_clips,
    normalize_video_clip,
    probe_video,
)
from app.video.continuity import build_continuity_reference
from app.video.frame_extractor import extract_last_frames
from app.video.generator import generate_scene_video


logger = logging.getLogger(__name__)


def _selected_video_model() -> str:
    if VIDEO_PROVIDER == "runway":
        return RUNWAY_MODEL
    if VIDEO_PROVIDER == "veo":
        return VEO_MODEL
    return "local-ffmpeg"


def _write_scene_diagnostics(path: Path, diagnostics: dict) -> None:
    path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _normalize_state_text(value: str) -> str:
    return " ".join(value.casefold().strip().rstrip(". ").split())


def _fit_required_motion_prompt(
    components: list[str],
    maximum_characters: int,
    character_length,
) -> str:
    normalized = [" ".join(component.split()) for component in components]
    if character_length(" ".join(normalized)) <= maximum_characters:
        return " ".join(normalized)

    labeled_components = []
    for component in normalized:
        label, separator, body = component.partition(":")
        labeled_components.append(
            (f"{label}:" if separator else "", body.strip().split())
        )

    selected_words = [[] for _ in labeled_components]
    word_indexes = [0 for _ in labeled_components]

    def render() -> str:
        rendered = []
        for (label, _), words in zip(
            labeled_components,
            selected_words,
            strict=True,
        ):
            if words:
                rendered.append(" ".join((label, *words)).strip())
            elif label:
                rendered.append(label)
        return " ".join(rendered)

    while True:
        added_word = False
        for index, (_, words) in enumerate(labeled_components):
            if word_indexes[index] >= len(words):
                continue
            selected_words[index].append(words[word_indexes[index]])
            if character_length(render()) <= maximum_characters:
                word_indexes[index] += 1
                added_word = True
            else:
                selected_words[index].pop()
        if not added_word:
            break

    shortened_components = [
        labeled_components[index][0].rstrip(":") or f"detail {index + 1}"
        for index, (_, words) in enumerate(labeled_components)
        if word_indexes[index] < len(words)
    ]
    logger.warning(
        "Required Runway prompt content exceeded the provider character "
        "limit; details were shortened at word boundaries across the "
        "required sections. Inspect the saved prompt and shorten these "
        "storyboard fields for best results.",
        extra={
            "shortened_components": shortened_components,
            "maximum_characters": maximum_characters,
        },
    )
    return render()


def _build_motion_prompt(storyboard: Storyboard, scene: Scene) -> str:
    location = next(
        location for location in storyboard.locations
        if location.id == scene.location_id
    )
    characters = [
        character for character in storyboard.characters
        if character.id in scene.character_ids
    ]
    character_lock = "; ".join(
        (
            f"{character.name}: age {character.appearance.age}, "
            f"{character.appearance.hair}, "
            f"{character.appearance.facial_features}, "
            f"{character.appearance.clothing}"
        )
        for character in characters
    )
    prop_lock = ", ".join(
        ", ".join(
            detail for detail in
            (prop.description, prop.continuity_anchor) if detail
        )
        for prop in storyboard.props
        if prop.id in scene.prop_ids
    )
    negative = scene.negative_prompt or (
        "No identity changes, face morphing, duplicate people, extra limbs, "
        "warped hands, costume changes, prop teleportation, text, logos, "
        "flicker, melting objects, or sudden camera cuts."
    )
    subject_instruction = (
        f"Identity lock: {character_lock}."
        if character_lock
        else (
            "Subject lock: preserve the same main subject, design, "
            "components, and appearance throughout the shot."
        )
    )
    appearance_instruction = (
        "Preserve faces, proportions, wardrobe, lighting, composition, "
        "and prop positions; animate only this action naturally."
        if character_lock
        else (
            "Preserve the subject's shape, components, lighting, and "
            "composition; animate only this action naturally."
        )
    )
    required_components = [
        subject_instruction,
        f"Starting state: {scene.continuity.required_start_state}.",
        f"Action: {scene.action}.",
        f"Ending state: {scene.continuity.ending_state}.",
        f"Visual: {scene.visual_description}.",
        (
            f"Camera: {scene.camera.shot_type}, {scene.camera.angle}, "
            f"{scene.camera.movement}, {scene.camera.lens or 'natural lens'}."
        ),
        f"Avoid: {negative}.",
        "One continuous shot.",
    ]
    required_prompt = " ".join(required_components)
    if VIDEO_PROVIDER != "runway":
        return (
            f"{required_prompt} "
            f"Visual style: {storyboard.project.visual_style}. "
            f"Location: {location.name}, {location.lighting}. "
            f"Subject state: {scene.continuity.character_state}. "
            f"Environment: {scene.continuity.environment_state}. "
            f"Relevant props: {prop_lock or 'none'}. "
            f"Blocking: {scene.blocking}. "
            f"Shot intent: {scene.shot_intent}. "
            f"Continuity: {scene.visual_continuity}. "
            f"{appearance_instruction}"
        )

    from app.video.providers.runway import (
        RUNWAY_PROMPT_MAX_CHARACTERS,
        _utf16_length,
    )

    required_prompt = _fit_required_motion_prompt(
        required_components,
        RUNWAY_PROMPT_MAX_CHARACTERS,
        _utf16_length,
    )
    if _utf16_length(required_prompt) > RUNWAY_PROMPT_MAX_CHARACTERS:
        raise ValueError(
            f"{scene.id} motion prompt could not be safely fit within Runway's "
            f"{RUNWAY_PROMPT_MAX_CHARACTERS}-character limit."
        )

    optional_components = [
        f"Relevant props: {prop_lock}." if prop_lock else "",
        f"Blocking: {scene.blocking}." if scene.blocking else "",
        f"Subject state: {scene.continuity.character_state}.",
        f"Environment: {scene.continuity.environment_state}.",
        f"Location and light: {location.name}, {location.lighting}.",
        f"Visual style: {storyboard.project.visual_style}.",
        f"Shot intent: {scene.shot_intent}." if scene.shot_intent else "",
        (
            f"Visual continuity: {scene.visual_continuity}."
            if scene.visual_continuity
            else ""
        ),
        appearance_instruction,
    ]
    prompt = required_prompt
    omitted_context: list[str] = []
    for component in optional_components:
        if not component:
            continue
        candidate = f"{prompt} {component}"
        if _utf16_length(candidate) <= RUNWAY_PROMPT_MAX_CHARACTERS:
            prompt = candidate
        else:
            omitted_context.append(component.split(":", 1)[0])
    if omitted_context:
        logger.info(
            "Optional Runway prompt context omitted to preserve the complete "
            "scene action and state",
            extra={"scene_id": scene.id, "omitted_context": omitted_context},
        )
    return prompt


def generate_video_from_storyboard(
    storyboard: Storyboard,
    output_dir: str | Path = Path("outputs"),
) -> Path:
    """Generate a video from a validated storyboard without creating one."""
    validate_storyboard_continuity(storyboard)
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "storyboard.json").write_text(
        storyboard.model_dump_json(indent=2),
        encoding="utf-8",
    )

    clip_paths: list[str] = []
    previous_video: str | None = None
    scene_durations = [
        scene.duration_seconds for scene in storyboard.scenes
    ]
    output_frame_rate: Fraction | None = None
    scene_frame_counts: list[int] | None = None

    for index, scene in enumerate(storyboard.scenes):
        scene_number = index + 1
        image_path = output_directory / f"scene_{scene_number:03d}.png"
        video_path = output_directory / f"scene_{scene_number:03d}.mp4"
        diagnostics_path = output_directory / (
            f"scene_{scene_number:03d}_generation.json"
        )
        continuity_references: tuple[str, ...] = ()
        image_prompt_path = output_directory / (
            f"scene_{scene_number:03d}_image_prompt.txt"
        )
        motion_prompt_path = output_directory / (
            f"scene_{scene_number:03d}_video_prompt.txt"
        )
        negative_prompt = build_scene_negative_prompt(scene)
        diagnostics = {
            "scene_id": scene.id,
            "requested_duration_seconds": scene.duration_seconds,
            "aspect_ratio": storyboard.project.aspect_ratio,
            "provider": VIDEO_PROVIDER,
            "model": _selected_video_model(),
            "image_generation_parameters": {
                "model": "runwayml/stable-diffusion-v1-5",
                "aspect_ratio": storyboard.project.aspect_ratio,
                "num_inference_steps": 35,
                "guidance_scale": 6.5,
                "image_to_image_strength": 0.30 if index > 0 else None,
            },
            "video_generation_parameters": {
                "duration_seconds": scene.duration_seconds,
                "aspect_ratio": storyboard.project.aspect_ratio,
                "target_frame_rate": (
                    str(output_frame_rate)
                    if output_frame_rate is not None
                    else "measured from first generated clip"
                ),
                "target_frame_count": (
                    scene_frame_counts[index]
                    if scene_frame_counts is not None
                    else "calculated from measured frame rate"
                ),
            },
            "image_prompt": None,
            "negative_prompt": negative_prompt,
            "dialogue": scene.dialogue,
            "audio_generation": "not supported by this video pipeline",
            "image_prompt_path": str(image_prompt_path),
            "scene_image_path": str(image_path),
            "reference_frame_paths": [],
            "video_prompt": None,
            "video_prompt_path": str(motion_prompt_path),
            "video_reference_images_requested": [],
            "scene_video_path": str(video_path),
            "generated_frame_count": None,
            "generated_frame_rate": None,
            "generated_duration_seconds": None,
            "generated_audio_codec": None,
        }
        if index > 0:
            previous_scene = storyboard.scenes[index - 1]
            previous_ending_state = previous_scene.continuity.ending_state
            current_start_state = scene.continuity.required_start_state
            states_match = (
                _normalize_state_text(previous_ending_state)
                == _normalize_state_text(current_start_state)
            )
            diagnostics["continuity_check"] = {
                "previous_scene_id": previous_scene.id,
                "previous_ending_state": previous_ending_state,
                "current_required_start_state": current_start_state,
                "textually_matches": states_match,
                "note": (
                    "Text comparison is diagnostic only and cannot prove "
                    "visual or semantic compatibility."
                ),
            }
            if not states_match:
                logger.warning(
                    "Adjacent scene states differ textually; inspect the "
                    "saved reference frame and prompt before generation",
                    extra={
                        "previous_scene_id": previous_scene.id,
                        "scene_id": scene.id,
                        "previous_ending_state": previous_ending_state,
                        "required_start_state": current_start_state,
                    },
                )

        if index == 0:
            image_prompt = build_scene_image_prompt(storyboard, scene)
            diagnostics["image_prompt"] = image_prompt
            image_prompt_path.write_text(image_prompt, encoding="utf-8")
            _write_scene_diagnostics(diagnostics_path, diagnostics)
            logger.info(
                "Generating scene image",
                extra={
                    "scene_id": scene.id,
                    "duration_seconds": scene.duration_seconds,
                    "image_prompt": image_prompt,
                    "negative_prompt": negative_prompt,
                    "image_path": str(image_path),
                    "provider": VIDEO_PROVIDER,
                    "model": diagnostics["model"],
                },
            )
            try:
                generate_scene_image(
                    prompt=image_prompt,
                    output_path=str(image_path),
                    negative_prompt=negative_prompt,
                    aspect_ratio=storyboard.project.aspect_ratio,
                )
            except Exception as exc:
                diagnostics["error"] = (
                    f"Image generation failed: {type(exc).__name__}: {exc}"
                )
                _write_scene_diagnostics(diagnostics_path, diagnostics)
                logger.exception(
                    "Scene image generation failed",
                    extra={"scene_id": scene.id},
                )
                raise
        else:
            if previous_video is None:
                raise RuntimeError(
                    "A previous scene video is required for continuity."
                )

            reference_dir = output_directory / (
                f"scene_{index:03d}_end_frames"
            )
            reference_frames = build_continuity_reference(
                extract_last_frames(
                    video_path=previous_video,
                    output_dir=str(reference_dir),
                )
            )
            continuity_references = (reference_frames[-1],)
            image_prompt = build_continuity_image_prompt(
                storyboard=storyboard,
                scene=scene,
                reference_frames=reference_frames,
            )
            diagnostics["image_prompt"] = image_prompt
            diagnostics["reference_frame_paths"] = reference_frames
            diagnostics["video_reference_images_requested"] = list(
                continuity_references
            )
            diagnostics["video_reference_images_submitted"] = (
                list(continuity_references)
                if VIDEO_PROVIDER == "veo"
                else []
            )
            image_prompt_path.write_text(image_prompt, encoding="utf-8")
            _write_scene_diagnostics(diagnostics_path, diagnostics)
            logger.info(
                "Generating continuity-adjusted scene image",
                extra={
                    "scene_id": scene.id,
                    "duration_seconds": scene.duration_seconds,
                    "image_prompt": image_prompt,
                    "negative_prompt": negative_prompt,
                    "reference_image_path": reference_frames[-1],
                    "image_path": str(image_path),
                },
            )
            try:
                generate_scene_image_from_reference(
                    prompt=image_prompt,
                    reference_image_path=reference_frames[-1],
                    output_path=str(image_path),
                    aspect_ratio=storyboard.project.aspect_ratio,
                    negative_prompt=negative_prompt,
                    scene_id=scene.id,
                )
            except Exception as exc:
                diagnostics["error"] = (
                    f"Continuity image generation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                _write_scene_diagnostics(diagnostics_path, diagnostics)
                logger.exception(
                    "Continuity image generation failed",
                    extra={"scene_id": scene.id},
                )
                raise

        motion_prompt = _build_motion_prompt(storyboard, scene)
        diagnostics["video_prompt"] = motion_prompt
        motion_prompt_path.write_text(motion_prompt, encoding="utf-8")
        diagnostics["first_frame_image_path"] = str(image_path)
        diagnostics.setdefault("video_reference_images_submitted", [])
        _write_scene_diagnostics(diagnostics_path, diagnostics)
        logger.info(
            "Generating scene video",
            extra={
                "scene_id": scene.id,
                "duration_seconds": scene.duration_seconds,
                "provider": VIDEO_PROVIDER,
                "model": diagnostics["model"],
                "video_prompt": motion_prompt,
                "first_frame": str(image_path),
                "reference_images_requested": list(continuity_references),
                "aspect_ratio": storyboard.project.aspect_ratio,
            },
        )
        try:
            previous_video = generate_scene_video(
                image_path=str(image_path),
                prompt=motion_prompt,
                output_path=str(video_path),
                duration_seconds=scene.duration_seconds,
                aspect_ratio=storyboard.project.aspect_ratio,
                reference_images=continuity_references,
                target_frame_rate=output_frame_rate,
                target_frame_count=(
                    scene_frame_counts[index]
                    if scene_frame_counts is not None
                    else None
                ),
                normalize_output=index != 0,
            )
        except KeyboardInterrupt:
            diagnostics["error"] = (
                "Video generation interrupted. A submitted remote task may "
                "still be running; check for its saved provider task state "
                "before starting another generation."
            )
            _write_scene_diagnostics(diagnostics_path, diagnostics)
            raise
        except Exception as exc:
            diagnostics["error"] = (
                f"Video generation failed: {type(exc).__name__}: {exc}"
            )
            _write_scene_diagnostics(diagnostics_path, diagnostics)
            logger.exception(
                "Scene video generation failed",
                extra={"scene_id": scene.id},
            )
            raise
        if output_frame_rate is None:
            first_clip = probe_video(previous_video)
            output_frame_rate = first_clip.frame_rate
            scene_frame_counts = calculate_scene_frame_counts(
                scene_durations,
                output_frame_rate,
            )
            normalize_video_clip(
                previous_video,
                video_path,
                target_frame_rate=output_frame_rate,
                target_frame_count=scene_frame_counts[0],
            )
            previous_video = str(video_path)
        video_info = probe_video(previous_video)
        diagnostics["scene_video_path"] = previous_video
        diagnostics["generated_frame_count"] = video_info.frame_count
        diagnostics["generated_frame_rate"] = str(video_info.frame_rate)
        diagnostics["generated_duration_seconds"] = str(
            video_info.duration_seconds
        )
        diagnostics["generated_audio_codec"] = video_info.audio_codec
        _write_scene_diagnostics(diagnostics_path, diagnostics)
        clip_paths.append(previous_video)

    final_path = output_directory / "final_video.mp4"
    concatenate_clips(
        clip_paths=clip_paths,
        output_path=str(final_path),
        scene_durations_seconds=scene_durations,
        frame_rate=output_frame_rate,
    )
    return final_path
