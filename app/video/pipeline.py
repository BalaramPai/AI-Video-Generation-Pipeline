import hashlib
import json
import logging
import re
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Callable

from app.assets.image_generator import (
    build_scene_negative_prompt,
    build_continuity_image_prompt,
    build_scene_image_prompt,
    generate_scene_image,
    generate_scene_image_from_reference,
)
from app.config import (
    CONTINUITY_IMAGE_MODE,
    IMAGE_TO_IMAGE_STRENGTH,
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
from app.video.providers.base import VideoGenerationResult


logger = logging.getLogger(__name__)

ImageReviewCallback = Callable[[Scene, Path], bool]


class ImageReviewDeclined(RuntimeError):
    """Raised when a user declines the pre-generation image review."""


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_required_motion_prompt(
    components: list[str],
    maximum_characters: int,
    character_length: Callable[[str], int],
) -> str:
    normalized = [" ".join(component.split()) for component in components]
    prompt = " ".join(component for component in normalized if component)
    if character_length(prompt) > maximum_characters:
        raise ValueError(
            "Required Runway motion details exceed the provider limit; "
            "shorten the subject, starting state, action, ending state, or "
            f"camera instructions to fit within {maximum_characters} "
            "UTF-16 characters. Required details were not truncated."
        )
    return prompt


def _layout_prompt_sentences(blocking: str) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(blocking.split()))
        if sentence.strip()
    ]
    layout_terms = (
        ("visible", 6),
        ("in frame", 6),
        ("obscur", 2),
        ("cover", 1),
        ("between", 2),
        ("left", 3),
        ("right", 3),
        ("center", 2),
        ("position", 2),
        ("stationary", 2),
    )
    return [
        sentence
        for _, sentence in sorted(
            enumerate(sentences),
            key=lambda item: (
                -sum(
                    weight
                    for term, weight in layout_terms
                    if term in item[1].casefold()
                ),
                item[0],
            ),
        )
    ]


def _build_motion_prompt(
    storyboard: Storyboard,
    scene: Scene,
    compilation_diagnostics: dict | None = None,
) -> str:
    location = next(
        location for location in storyboard.locations
        if location.id == scene.location_id
    )
    characters = [
        character for character in storyboard.characters
        if character.id in scene.character_ids
    ]
    relevant_props = [
        prop for prop in storyboard.props if prop.id in scene.prop_ids
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
    runway_character_lock = "; ".join(
        (
            f"{character.name}, age {character.appearance.age}, "
            f"{character.appearance.hair}, {character.appearance.clothing}"
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
            "Subject lock: same "
            + (
                ", ".join(prop.name for prop in relevant_props)
                if relevant_props
                else "main subject"
            )
            + "; preserve its shape and appearance throughout the shot."
        )
    )
    runway_subject_instruction = (
        f"Identity lock: {runway_character_lock}."
        if runway_character_lock
        else subject_instruction
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
    core_components = [
        (
            runway_subject_instruction
            if VIDEO_PROVIDER == "runway"
            else subject_instruction
        ),
        f"Starting state: {scene.continuity.required_start_state}.",
        f"Action: {scene.action}.",
        f"Ending state: {scene.continuity.ending_state}.",
        (
            f"Camera: {scene.camera.shot_type}, {scene.camera.angle}, "
            f"{scene.camera.movement}"
            + (
                f", {scene.camera.lens or 'natural lens'}."
                if VIDEO_PROVIDER != "runway"
                else "."
            )
        ),
    ]
    required_components = [
        *core_components,
        f"Visual: {scene.visual_description}.",
        f"Avoid: {negative}.",
    ]
    required_prompt = " ".join(required_components)
    if VIDEO_PROVIDER != "runway":
        prompt = (
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
        if compilation_diagnostics is not None:
            compilation_diagnostics.update(
                {
                    "video_prompt_character_count": len(prompt),
                    "video_prompt_character_limit": None,
                    "video_prompt_character_count_unit": (
                        "Unicode code points; provider-specific limit "
                        "not declared"
                    ),
                    "video_prompt_omitted_context": [],
                    "video_prompt_omitted_negative_constraints": [],
                }
            )
        return prompt

    from app.video.providers.runway import (
        RUNWAY_PROMPT_MAX_CHARACTERS,
        RUNWAY_MODEL_PROMPT_MAX_CHARACTERS,
        _utf16_length,
    )
    maximum_prompt_characters = RUNWAY_MODEL_PROMPT_MAX_CHARACTERS.get(
        RUNWAY_MODEL,
        RUNWAY_PROMPT_MAX_CHARACTERS,
    )

    try:
        required_prompt = _fit_required_motion_prompt(
            core_components,
            maximum_prompt_characters,
            _utf16_length,
        )
    except ValueError as exc:
        raise ValueError(f"{scene.id}: {exc}") from exc

    def negative_priority(clause: str) -> int:
        normalized = clause.casefold()
        if any(word in normalized for word in ("camera", "cut", "transition")):
            return 0
        if any(
            word in normalized
            for word in (
                "reverse",
                "upward",
                "backward",
                "reset",
                "teleport",
                "different",
                "extra",
                "duplicate",
                "missing",
                "rearranged",
                "geometry",
                "shape",
                "layer",
            )
        ):
            return 1
        return 2

    negative_clauses = [
        clause.strip()
        for clause in re.split(r"[,;]", negative)
        if clause.strip()
    ]
    negative_clauses = [
        clause
        if re.match(r"^(no|not|without|avoid)\b", clause, re.IGNORECASE)
        else f"No {clause}"
        for clause in negative_clauses
    ]
    negative_clauses = [
        clause
        for _, clause in sorted(
            enumerate(negative_clauses),
            key=lambda item: (negative_priority(item[1]), item[0]),
        )
    ]

    prop_instruction = (
        "Props: "
        + "; ".join(prop.name for prop in relevant_props)
        + ". No extra or duplicate props."
        if relevant_props
        else ""
    )
    priority_components = [f"Visual: {scene.visual_description}."]
    optional_components = [
        f"Relevant props: {prop_lock}." if prop_lock else "",
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
    omitted_negative_constraints: list[str] = []
    blocking_sentences = _layout_prompt_sentences(scene.blocking)
    omitted_blocking = len(blocking_sentences) > 1
    if prop_instruction:
        candidate = f"{prompt} {prop_instruction}"
        if _utf16_length(candidate) <= maximum_prompt_characters:
            prompt = candidate
        else:
            omitted_context.append("Prop identities and duplicate safeguard")
    for sentence in blocking_sentences[:1]:
        candidate = f"{prompt} Blocking: {sentence}"
        if _utf16_length(candidate) <= maximum_prompt_characters:
            prompt = candidate
        else:
            omitted_blocking = True
    if omitted_blocking:
        omitted_context.append("Additional blocking details")
    if scene.camera.lens:
        candidate = f"{prompt} Lens: {scene.camera.lens}."
        if _utf16_length(candidate) <= maximum_prompt_characters:
            prompt = candidate
        else:
            omitted_context.append("Camera lens")
    fitted_negative: list[str] = []
    for clause in negative_clauses:
        candidate_clauses = [*fitted_negative, clause]
        candidate = (
            f"{prompt} Avoid: {'; '.join(candidate_clauses)}"
        )
        if _utf16_length(candidate) <= maximum_prompt_characters:
            fitted_negative.append(clause)
        else:
            omitted_negative_constraints.append(clause)
    if fitted_negative:
        prompt = f"{prompt} Avoid: {'; '.join(fitted_negative)}"
    for component in priority_components:
        if not component:
            continue
        candidate = f"{prompt} {component}"
        if _utf16_length(candidate) <= maximum_prompt_characters:
            prompt = candidate
        else:
            omitted_context.append(component.split(":", 1)[0])
    for component in optional_components:
        if not component:
            continue
        candidate = f"{prompt} {component}"
        if _utf16_length(candidate) <= maximum_prompt_characters:
            prompt = candidate
        else:
            omitted_context.append(component.split(":", 1)[0])
    final_character_count = _utf16_length(prompt)
    if final_character_count > maximum_prompt_characters:
        raise ValueError(
            f"{scene.id} assembled motion prompt is "
            f"{final_character_count} UTF-16 characters; Runway supports "
            f"{maximum_prompt_characters} for {RUNWAY_MODEL}."
        )
    logger.info(
        "Final assembled Runway motion prompt",
        extra={
            "scene_id": scene.id,
            "character_count": final_character_count,
            "maximum_characters": maximum_prompt_characters,
        },
    )
    if omitted_context:
        logger.info(
            "Optional Runway prompt context omitted to preserve required "
            "subject, action, state, and camera details",
            extra={"scene_id": scene.id, "omitted_context": omitted_context},
        )
    if omitted_negative_constraints:
        logger.warning(
            "Runway prompt negative constraints omitted as complete clauses "
            "because the provider limit was reached",
            extra={
                "scene_id": scene.id,
                "omitted_negative_constraints": omitted_negative_constraints,
            },
        )
    if compilation_diagnostics is not None:
        compilation_diagnostics.update(
            {
                "video_prompt_character_count": final_character_count,
                "video_prompt_character_limit": maximum_prompt_characters,
                "video_prompt_character_count_unit": "UTF-16 code units",
                "video_prompt_omitted_context": omitted_context,
                "video_prompt_omitted_negative_constraints": (
                    omitted_negative_constraints
                ),
            }
        )
    return prompt


def generate_video_from_storyboard(
    storyboard: Storyboard,
    output_dir: str | Path = Path("outputs"),
    image_review_callback: ImageReviewCallback | None = None,
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
                "image_to_image_strength": (
                    IMAGE_TO_IMAGE_STRENGTH
                    if index > 0 and CONTINUITY_IMAGE_MODE == "img2img"
                    else None
                ),
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
            "image_prompt_applied": index == 0,
            "negative_prompt": negative_prompt,
            "dialogue": scene.dialogue,
            "audio_generation": "not supported by this video pipeline",
            "image_prompt_path": str(image_prompt_path),
            "scene_image_path": str(image_path),
            "reference_frame_paths": [],
            "continuity_image_generation_mode": (
                CONTINUITY_IMAGE_MODE if index > 0 else "text_to_image"
            ),
            "continuity_reference_sha256": None,
            "first_frame_image_sha256": None,
            "provider_generation_id": None,
            "provider_generation_metadata": {},
            "image_review_status": "not_requested",
            "video_prompt": None,
            "video_prompt_path": str(motion_prompt_path),
            "video_prompt_character_count": None,
            "video_prompt_character_limit": None,
            "video_prompt_character_count_unit": None,
            "video_prompt_omitted_context": [],
            "video_prompt_omitted_negative_constraints": [],
            "video_reference_images_requested": [],
            "scene_video_path": str(video_path),
            "generated_frame_count": None,
            "generated_frame_rate": None,
            "generated_duration_seconds": None,
            "generated_audio_codec": None,
            "transition_strategy": "hard_cut",
        }
        _write_scene_diagnostics(diagnostics_path, diagnostics)
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
                if image_path.exists():
                    image_path.unlink()
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

            boundary = {
                "previous_scene_id": storyboard.scenes[index - 1].id,
                "next_scene_id": scene.id,
                "source_video_path": previous_video,
                "provider": VIDEO_PROVIDER,
                "model": diagnostics["model"],
                "status": "extracting_reference_frames",
                "candidate_frames": [],
                "selected_frame": None,
                "img2img_input_path": None,
                "img2img_output_path": (
                    str(image_path)
                    if CONTINUITY_IMAGE_MODE == "img2img"
                    else None
                ),
                "provider_input_image_path": str(image_path),
            }
            diagnostics["continuity_boundary"] = boundary
            _write_scene_diagnostics(diagnostics_path, diagnostics)
            reference_dir = output_directory / (
                f"scene_{index:03d}_end_frames"
            )
            try:
                source_info = probe_video(previous_video)
                reference_frames = build_continuity_reference(
                    extract_last_frames(
                        video_path=previous_video,
                        output_dir=str(reference_dir),
                    )
                )
                candidate_frames = []
                for frame in reference_frames:
                    frame_path = Path(frame)
                    frame_index = int(frame_path.stem.rsplit("_", 1)[-1])
                    timestamp = (
                        Fraction(frame_index, 1) / source_info.frame_rate
                    )
                    candidate_frames.append(
                        {
                            "path": str(frame_path),
                            "frame_index": frame_index,
                            "timestamp_seconds": float(timestamp),
                        }
                    )
                boundary.update(
                    {
                        "status": "reference_selected",
                        "source_video_duration_seconds": str(
                            source_info.duration_seconds
                        ),
                        "source_video_frame_count": source_info.frame_count,
                        "source_video_frame_rate": str(
                            source_info.frame_rate
                        ),
                        "candidate_timestamp_basis": (
                            "frame_index / normalized CFR frame rate"
                        ),
                        "candidate_frames": candidate_frames,
                    }
                )
            except Exception as exc:
                boundary["status"] = "reference_extraction_failed"
                boundary["error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                _write_scene_diagnostics(diagnostics_path, diagnostics)
                logger.exception(
                    "Previous-scene reference extraction failed",
                    extra={
                        "previous_scene_id": boundary["previous_scene_id"],
                        "scene_id": scene.id,
                        "source_video": previous_video,
                    },
                )
                raise
            continuity_references = (reference_frames[-1],)
            selected_candidate = candidate_frames[-1]
            boundary["selected_frame"] = selected_candidate
            diagnostics["reference_source_video_path"] = previous_video
            reference_path = Path(continuity_references[0])
            reference_sha256 = _sha256_file(reference_path)
            boundary["continuity_image_mode"] = CONTINUITY_IMAGE_MODE
            boundary["reference_frame_sha256"] = reference_sha256
            diagnostics["continuity_reference_sha256"] = reference_sha256
            diagnostics["continuity_image_generation_mode"] = (
                CONTINUITY_IMAGE_MODE
            )
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
            try:
                if image_path.exists():
                    image_path.unlink()
                if CONTINUITY_IMAGE_MODE == "img2img":
                    boundary["img2img_input_path"] = str(reference_path)
                    boundary["img2img_strength"] = IMAGE_TO_IMAGE_STRENGTH
                    diagnostics["reference_frame_used_for_img2img"] = str(
                        reference_path
                    )
                    diagnostics["image_prompt_applied"] = True
                    logger.info(
                        "Generating continuity-adjusted scene image from "
                        "the extracted predecessor frame",
                        extra={
                            "scene_id": scene.id,
                            "reference_image_path": str(reference_path),
                            "image_path": str(image_path),
                            "strength": IMAGE_TO_IMAGE_STRENGTH,
                        },
                    )
                    generate_scene_image_from_reference(
                        prompt=image_prompt,
                        reference_image_path=str(reference_path),
                        output_path=str(image_path),
                        aspect_ratio=storyboard.project.aspect_ratio,
                        negative_prompt=negative_prompt,
                        scene_id=scene.id,
                        strength=IMAGE_TO_IMAGE_STRENGTH,
                    )
                    boundary["status"] = "img2img_completed"
                else:
                    boundary["img2img_input_path"] = None
                    boundary["img2img_strength"] = None
                    diagnostics["reference_frame_used_for_img2img"] = None
                    diagnostics["image_prompt_applied"] = False
                    shutil.copyfile(reference_path, image_path)
                    boundary["status"] = "previous_frame_copied"
                    logger.info(
                        "Using the extracted predecessor frame directly as "
                        "the next scene's first-frame image",
                        extra={
                            "scene_id": scene.id,
                            "source_video": previous_video,
                            "reference_image_path": str(reference_path),
                            "first_frame_image_path": str(image_path),
                            "image_sha256": reference_sha256,
                        },
                    )
                continuity_image_sha256 = _sha256_file(image_path)
                boundary["continuity_image_sha256"] = (
                    continuity_image_sha256
                )
                boundary["continuity_image_matches_reference"] = (
                    continuity_image_sha256 == reference_sha256
                )
                diagnostics["first_frame_image_sha256"] = (
                    continuity_image_sha256
                )
                boundary["img2img_output_path"] = (
                    str(image_path)
                    if CONTINUITY_IMAGE_MODE == "img2img"
                    else None
                )
                _write_scene_diagnostics(diagnostics_path, diagnostics)
            except Exception as exc:
                boundary["status"] = "continuity_image_failed"
                boundary["error"] = f"{type(exc).__name__}: {exc}"
                diagnostics["error"] = (
                    f"Continuity image preparation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                _write_scene_diagnostics(diagnostics_path, diagnostics)
                logger.exception(
                    "Continuity image preparation failed",
                    extra={"scene_id": scene.id},
                )
                raise
        runway_task_state_path = video_path.with_name(
                f"{video_path.stem}.runway-task.json"
        )
        try:
            motion_prompt = _build_motion_prompt(
                storyboard,
                scene,
                compilation_diagnostics=diagnostics,
            )
        except Exception as exc:
            diagnostics["error"] = (
                f"Motion prompt compilation failed: "
                f"{type(exc).__name__}: {exc}"
            )
            _write_scene_diagnostics(diagnostics_path, diagnostics)
            logger.exception(
                "Scene motion prompt compilation failed",
                extra={"scene_id": scene.id},
            )
            raise
        diagnostics["video_prompt"] = motion_prompt
        motion_prompt_path.write_text(motion_prompt, encoding="utf-8")
        diagnostics["first_frame_image_path"] = str(image_path)
        diagnostics.setdefault("video_reference_images_submitted", [])
        _write_scene_diagnostics(diagnostics_path, diagnostics)
        if image_review_callback is not None:
            if VIDEO_PROVIDER == "runway" and runway_task_state_path.is_file():
                diagnostics["image_review_status"] = (
                    "skipped_existing_provider_task"
                )
                logger.info(
                    "Skipping image review while resuming a saved Runway "
                    "task; the provider will restore its submitted input "
                    "image",
                    extra={
                        "scene_id": scene.id,
                        "task_state_path": str(runway_task_state_path),
                    },
                )
            elif image_review_callback(scene, image_path):
                diagnostics["image_review_status"] = "approved"
            else:
                diagnostics["image_review_status"] = "declined"
                diagnostics["error"] = (
                    "Image review declined before video generation."
                )
                _write_scene_diagnostics(diagnostics_path, diagnostics)
                raise ImageReviewDeclined(
                    f"Image review declined for {scene.id}; "
                    "no video request was made."
                )
            _write_scene_diagnostics(diagnostics_path, diagnostics)

        def record_generation_result(
            result: VideoGenerationResult,
            provider_input_image: Path,
        ) -> None:
            actual_first_frame_sha256 = _sha256_file(
                provider_input_image
            )
            diagnostics["provider_generation_id"] = result.generation_id
            diagnostics["provider_generation_metadata"] = dict(
                result.metadata
            )
            diagnostics["first_frame_image_sha256"] = (
                actual_first_frame_sha256
            )
            if index > 0:
                boundary = diagnostics["continuity_boundary"]
                boundary["provider_input_image_path"] = str(
                    provider_input_image
                )
                boundary["provider_input_image_sha256"] = (
                    actual_first_frame_sha256
                )
                boundary["provider_input_matches_reference"] = (
                    actual_first_frame_sha256
                    == boundary["reference_frame_sha256"]
                )
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
                on_generation_result=record_generation_result,
            )
        except KeyboardInterrupt:
            if index > 0:
                diagnostics["continuity_boundary"]["status"] = (
                    "provider_task_pending_interrupted"
                    if VIDEO_PROVIDER == "runway"
                    and runway_task_state_path.is_file()
                    else "provider_generation_interrupted"
                )
                if runway_task_state_path.is_file():
                    diagnostics["continuity_boundary"][
                        "pending_task_state_path"
                    ] = str(runway_task_state_path)
            diagnostics["error"] = (
                "Video generation interrupted. A submitted remote task may "
                "still be running; check for its saved provider task state "
                "before starting another generation."
            )
            _write_scene_diagnostics(diagnostics_path, diagnostics)
            raise
        except Exception as exc:
            if index > 0:
                pending_runway_task = (
                    VIDEO_PROVIDER == "runway"
                    and runway_task_state_path.is_file()
                )
                diagnostics["continuity_boundary"]["status"] = (
                    "provider_task_pending"
                    if pending_runway_task
                    else "provider_generation_failed"
                )
                diagnostics["continuity_boundary"]["error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                if pending_runway_task:
                    diagnostics["continuity_boundary"][
                        "pending_task_state_path"
                    ] = str(runway_task_state_path)
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
        if index > 0:
            diagnostics["continuity_boundary"].update(
                {
                    "status": "completed",
                    "provider_input_image_path": str(image_path),
                    "normalized_video_path": previous_video,
                    "normalized_frame_rate": str(video_info.frame_rate),
                    "normalized_frame_count": video_info.frame_count,
                    "normalized_duration_seconds": str(
                        video_info.duration_seconds
                    ),
                }
            )
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
