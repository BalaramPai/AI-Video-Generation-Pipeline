import logging
import re
from pathlib import Path
from typing import Sequence

import torch
from diffusers import StableDiffusionPipeline

from app.config import IMAGE_TO_IMAGE_STRENGTH
from app.storyboard.models import (
    Prop,
    Storyboard,
    Scene,
)

from transformers import CLIPTokenizer
from PIL import Image


logger = logging.getLogger(__name__)

IMAGE_NEGATIVE_PROMPT = "low quality, blurry, watermark, logo"
CHARACTER_IMAGE_NEGATIVE_PROMPT = (
    "deformed, warped anatomy, extra fingers, missing fingers, extra limbs, "
    "duplicate person, duplicate face, merged faces, changing identity, "
    "changed clothing, bad hands"
)
# Keep two CLIP positions free for BOS/EOS, leaving 75 input IDs total.
IMAGE_PROMPT_MAX_TOKENS = 75

_IMAGE_DIMENSIONS = {
    "16:9": (768, 432),
    "9:16": (432, 768),
    "4:3": (576, 432),
    "3:4": (432, 576),
    "1:1": (512, 512),
    "21:9": (672, 288),
    "3:2": (768, 512),
    "2:3": (512, 768),
}


def _image_dimensions(aspect_ratio: str) -> tuple[int, int]:
    try:
        return _IMAGE_DIMENSIONS[aspect_ratio]
    except KeyError as exc:
        supported = ", ".join(_IMAGE_DIMENSIONS)
        raise ValueError(
            f"Unsupported image aspect ratio {aspect_ratio!r}. "
            f"Choose one of: {supported}."
        ) from exc


def _character_prompt(character) -> str:
    appearance = character.appearance

    def compact(value: str, limit: int = 72) -> str:
        value = value.strip()
        return value if len(value) <= limit else value[:limit].rsplit(" ", 1)[0]

    details = [
        f"{character.name}, age {appearance.age}",
        compact(appearance.hair),
        compact(appearance.clothing),
        compact(appearance.facial_features),
        compact(appearance.physical_description),
    ]
    existing_words = {
        word.casefold()
        for detail in details
        for word in re.findall(r"\w+", detail)
    }
    for extra in (
        f"{appearance.eye_color} eyes" if appearance.eye_color else "",
        appearance.distinctive_features,
        appearance.wardrobe_anchor,
    ):
        extra = compact(extra)
        extra_words = {
            word.casefold() for word in re.findall(r"\w+", extra)
        }
        if extra and not extra_words.issubset(existing_words):
            details.append(extra)
            existing_words.update(extra_words)
    return ", ".join(detail.strip() for detail in details if detail.strip())


def _token_count(
    prompt: str,
    tokenizer: CLIPTokenizer | None = None,
) -> int:
    if tokenizer is None:
        tokenizer = TOKENIZER
    encoded = tokenizer(
        prompt,
        truncation=False,
        add_special_tokens=True,
        return_tensors="pt",
        verbose=False,
    )
    return int(encoded["input_ids"].shape[1])


def _fit_image_prompt(
    required_components: Sequence[str],
    label: str,
    optional_components: Sequence[tuple[str, str]] = (),
) -> str:
    """Fit required scene details fairly, then add optional context."""
    required = [
        " ".join(component.split())
        for component in required_components
        if component and component.strip()
    ]
    if not required:
        raise ValueError(f"{label} has no required scene information.")

    prompt = ", ".join(required)
    if _token_count(prompt) > IMAGE_PROMPT_MAX_TOKENS:
        components: list[tuple[str, list[str]]] = []
        for component in required:
            heading, separator, body = component.partition(":")
            if separator:
                components.append((f"{heading}:", body.strip().split()))
            else:
                components.append(("", component.split()))

        selected_words = [[] for _ in components]
        word_indexes = [0 for _ in components]

        def render_required() -> str:
            rendered = []
            for (heading, _), words in zip(
                components,
                selected_words,
                strict=True,
            ):
                if words:
                    rendered.append(" ".join((heading, *words)).strip())
                elif heading:
                    rendered.append(heading)
            return ", ".join(rendered)

        while True:
            added_word = False
            for index, (_, words) in enumerate(components):
                if word_indexes[index] >= len(words):
                    continue
                selected_words[index].append(words[word_indexes[index]])
                candidate = render_required()
                if _token_count(candidate) <= IMAGE_PROMPT_MAX_TOKENS:
                    word_indexes[index] += 1
                    added_word = True
                else:
                    selected_words[index].pop()
            if not added_word:
                break

        prompt = render_required()
        shortened_components = [
            components[index][0].rstrip(":") or f"detail {index + 1}"
            for index, words in enumerate(components)
            if word_indexes[index] < len(words)
        ]
        logger.warning(
            "%s required details exceeded the Stable Diffusion prompt limit; "
            "some detail text was shortened at word boundaries. Inspect the "
            "saved prompt and shorten these storyboard fields for best "
            "results.",
            label,
            extra={
                "prompt_label": label,
                "shortened_components": shortened_components,
                "token_limit": IMAGE_PROMPT_MAX_TOKENS,
            },
        )

    omitted: list[str] = []
    for name, component in optional_components:
        compact_component = " ".join(component.split())
        if not compact_component:
            continue
        candidate = f"{prompt}, {compact_component}"
        if _token_count(candidate) <= IMAGE_PROMPT_MAX_TOKENS:
            prompt = candidate
        else:
            omitted.append(name)

    if omitted:
        logger.info(
            "Optional Stable Diffusion prompt context omitted to preserve "
            "required scene facts",
            extra={"prompt_label": label, "omitted_context": omitted},
        )
    return prompt


def build_scene_negative_prompt(
    scene: Scene,
    negative_prompt: str | None = None,
) -> str:
    components = [IMAGE_NEGATIVE_PROMPT]
    if scene.character_ids:
        components.append(CHARACTER_IMAGE_NEGATIVE_PROMPT)
    prompt = ", ".join(components)
    custom = (
        scene.negative_prompt if negative_prompt is None else negative_prompt
    )
    if custom and custom.strip():
        custom_components = [
            " ".join(component.split())
            for component in re.split(r"[,;]", custom)
            if component.strip()
        ]
        omitted = []
        for component in custom_components:
            candidate = f"{prompt}, {component}"
            if _token_count(candidate) <= IMAGE_PROMPT_MAX_TOKENS:
                prompt = candidate
            else:
                omitted.append(component)

        if omitted:
            logger.warning(
                "Custom negative-prompt exclusions exceeded the Stable "
                "Diffusion prompt limit; some exclusions were omitted. "
                "Shorten the scene negative_prompt to preserve them.",
                extra={
                    "scene_id": scene.id,
                    "omitted_negative_prompt_components": omitted,
                    "token_limit": IMAGE_PROMPT_MAX_TOKENS,
                },
            )
    return prompt



def _scene_image_prompt_components(
    storyboard: Storyboard,
    scene: Scene,
) -> tuple[list[str], list[tuple[str, str]]]:
    location = next(
        location
        for location in storyboard.locations
        if location.id == scene.location_id
    )

    characters = [
        character
        for character in storyboard.characters
        if character.id in scene.character_ids
    ]

    character_details = "; ".join(
        _character_prompt(character) for character in characters
    )
    relevant_props = [
        prop
        for prop in storyboard.props
        if prop.id in scene.prop_ids
    ]
    prop_details = "; ".join(
        _compact_prop_prompt(prop)
        for prop in relevant_props
    )
    subject = character_details or prop_details or scene.continuity.character_state
    required = [
        f"Subject: {subject}",
        f"Start: {scene.continuity.required_start_state}",
        f"Action: {scene.action}",
        f"Layout: {scene.visual_description}",
    ]
    if scene.blocking.strip():
        required.append(f"Layout: {scene.blocking}")
    if character_details and prop_details:
        required.append(f"Relevant objects: {prop_details}")

    optional = [
        ("environment state", scene.continuity.environment_state),
        (
            "location",
            ", ".join(
                detail
                for detail in (
                    location.name,
                    location.lighting,
                    location.continuity_anchor,
                )
                if detail
            ),
        ),
        ("visual style", storyboard.project.visual_style),
        (
            "camera",
            ", ".join(
                detail
                for detail in (
                    scene.camera.shot_type,
                    scene.camera.angle,
                    scene.camera.movement,
                    scene.camera.lens or "",
                )
                if detail
            ),
        ),
    ]
    return required, optional


def _compact_prop_prompt(prop: Prop) -> str:
    def remove_repeated_name(detail: str) -> str:
        without_name = re.sub(
            rf"\b{re.escape(prop.name)}\b",
            "",
            detail,
            flags=re.IGNORECASE,
        )
        return " ".join(
            without_name.strip(" ,;:.").split()
        )

    details = [prop.name]
    known_words = {
        word.casefold() for word in re.findall(r"\w+", prop.name)
    }
    for detail in (prop.description, prop.continuity_anchor):
        if detail and detail.strip():
            compact_detail = remove_repeated_name(detail)
            if compact_detail:
                words = re.findall(r"\w+", compact_detail)
                unique_words = [
                    word
                    for word in words
                    if word.casefold() not in known_words
                ]
                if unique_words:
                    details.append(" ".join(unique_words))
                    known_words.update(word.casefold() for word in unique_words)
    return " ".join(details)


def build_scene_image_prompt(
    storyboard: Storyboard,
    scene: Scene,
) -> str:
    required, optional = _scene_image_prompt_components(storyboard, scene)
    return _fit_image_prompt(
        required,
        "Scene image prompt",
        optional_components=optional,
    )


MODEL_ID = "runwayml/stable-diffusion-v1-5"
TOKENIZER = CLIPTokenizer.from_pretrained(
    MODEL_ID,
    subfolder="tokenizer",
)
_pipe = None

def get_pipeline():
    global _pipe

    if _pipe is None:
        print("\nLoading Stable Diffusion...\n")

        _pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
        )

        _pipe.enable_model_cpu_offload()

    return _pipe


def generate_scene_image(
    prompt: str,
    output_path: str,
    negative_prompt: str | None = None,
    aspect_ratio: str = "16:9",
) -> str:

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    effective_negative_prompt = negative_prompt or IMAGE_NEGATIVE_PROMPT
    _validate_pipeline_prompts(prompt, effective_negative_prompt, TOKENIZER)
    pipe = get_pipeline()
    _validate_pipeline_prompts(
        prompt,
        effective_negative_prompt,
        pipe.tokenizer,
    )

    print("\nGenerating image...\n")

    width, height = _image_dimensions(aspect_ratio)
    image = pipe(
        prompt=prompt,
        negative_prompt=effective_negative_prompt,
        width=width,
        height=height,
        num_inference_steps=35,
        guidance_scale=6.5,
    ).images[0]

    image.save(output)

    print(f"Image saved to: {output}")

    return str(output)


def build_continuity_image_prompt(
    storyboard: Storyboard,
    scene: Scene,
    reference_frames: list[str],
) -> str:

    if not reference_frames:
        raise ValueError(
            "Scene 2 requires continuity reference frames."
        )

    continuity_instruction = (
        "Use reference for identity and setting; honor this initial state."
        if scene.character_ids
        else (
            "Use reference for subject identity and setting; honor this "
            "initial state."
        )
    )
    required, optional = _scene_image_prompt_components(storyboard, scene)
    required.append(continuity_instruction)
    return _fit_image_prompt(
        required,
        "Continuity image prompt",
        optional_components=optional,
    )

def generate_scene_image_from_reference(
    prompt: str,
    reference_image_path: str,
    output_path: str,
    aspect_ratio: str = "16:9",
    negative_prompt: str | None = None,
    scene_id: str | None = None,
    strength: float = IMAGE_TO_IMAGE_STRENGTH,
) -> str:

    if not 0.0 <= strength <= 1.0:
        raise ValueError(
            "IMAGE_TO_IMAGE_STRENGTH must be between 0.0 and 1.0."
        )
    effective_negative_prompt = negative_prompt or IMAGE_NEGATIVE_PROMPT
    _validate_pipeline_prompts(prompt, effective_negative_prompt, TOKENIZER)

    from diffusers import StableDiffusionImg2ImgPipeline

    reference = Path(reference_image_path)

    if not reference.exists():
        raise FileNotFoundError(
            f"Reference image not found: {reference}"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("\nGenerating Scene 2 from continuity reference...\n")

    width, height = _image_dimensions(aspect_ratio)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )

    pipe.enable_model_cpu_offload()

    _validate_pipeline_prompts(
        prompt,
        effective_negative_prompt,
        pipe.tokenizer,
    )
    logger.info(
        "Generating scene image from previous-scene reference",
        extra={
            "scene_id": scene_id,
            "reference_image_path": str(reference),
            "output_image_path": str(output),
            "image_prompt": prompt,
            "negative_prompt": effective_negative_prompt,
            "strength": strength,
        },
    )
    with Image.open(reference) as reference_image:
        image = reference_image.convert("RGB")

    result = pipe(
        prompt=prompt,
        negative_prompt=effective_negative_prompt,
        image=image,
        width=width,
        height=height,
        # A low denoise strength preserves the reference identity and
        # composition instead of redrawing the subject from scratch.
        strength=strength,
        guidance_scale=6.5,
        num_inference_steps=35,
    ).images[0]

    result.save(output)

    print(f"Scene 2 image saved to: {output}")

    return str(output)


def _validate_pipeline_prompts(
    prompt: str,
    negative_prompt: str,
    tokenizer: CLIPTokenizer,
) -> None:
    for label, value in (
        ("positive", prompt),
        ("negative", negative_prompt),
    ):
        token_count = _token_count(value, tokenizer)
        if token_count > IMAGE_PROMPT_MAX_TOKENS:
            raise ValueError(
                f"The {label} Stable Diffusion prompt contains "
                f"{token_count} tokens, but this model supports "
                f"{IMAGE_PROMPT_MAX_TOKENS}; it would be truncated."
            )