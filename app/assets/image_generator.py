from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

from app.storyboard.models import (
    Storyboard,
    Scene,
)

from transformers import CLIPTokenizer
from PIL import Image


IMAGE_NEGATIVE_PROMPT = (
    "low quality, blurry, deformed, warped anatomy, extra fingers, missing "
    "fingers, extra limbs, duplicate person, duplicate face, merged faces, "
    "changing identity, changed clothing, text, watermark, logo, bad hands"
)


def _character_prompt(character) -> str:
    appearance = character.appearance

    def compact(value: str, limit: int = 72) -> str:
        value = value.strip()
        return value if len(value) <= limit else value[:limit].rsplit(" ", 1)[0]

    details = [
        f"{character.name}, age {appearance.age}",
        compact(appearance.hair),
        compact(appearance.facial_features),
        compact(appearance.clothing),
    ]
    if appearance.eye_color:
        details.append(f"{compact(appearance.eye_color)} eyes")
    if appearance.distinctive_features:
        details.append(compact(appearance.distinctive_features))
    if appearance.wardrobe_anchor:
        details.append(compact(appearance.wardrobe_anchor))
    return ", ".join(detail.strip() for detail in details if detail.strip())


def _fit_image_prompt(
    components: list[str],
    label: str,
    minimum_components: int = 5,
) -> str:
    """Keep SD 1.5 prompts inside CLIP's usable 75-token budget."""
    prompt = ", ".join(component for component in components if component)
    token_count = TOKENIZER(
        prompt,
        truncation=False,
        return_tensors="pt",
        verbose=False,
    )["input_ids"].shape[1]

    if token_count <= 75:
        return prompt

    # Remove optional prose before shortening identity or action details.
    required = components[:]
    while len(required) > minimum_components:
        required.pop()
        prompt = ", ".join(component for component in required if component)
        token_count = TOKENIZER(
            prompt,
            truncation=False,
            return_tensors="pt",
            verbose=False,
        )["input_ids"].shape[1]
        if token_count <= 75:
            return prompt

    raise ValueError(
        f"{label} is too long for SD 1.5 CLIP after removing optional "
        f"details ({token_count} tokens; maximum is 75)."
    )



def build_scene_image_prompt(
    storyboard: Storyboard,
    scene: Scene,
) -> str:

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

    if not characters:
        raise ValueError(f"{scene.id} must contain at least one character.")

    # Keep the image prompt compact; the storyboard retains the full
    # production detail, while SD 1.5 has a hard CLIP context limit.
    location_detail = ", ".join(
        detail
        for detail in (
            location.name,
            location.lighting,
            location.continuity_anchor,
        )
        if detail
    )
    character_details = "; ".join(
        _character_prompt(character) for character in characters
    )
    prop_details = "; ".join(
        ", ".join(detail for detail in (prop.description, prop.continuity_anchor) if detail)
        for prop in storyboard.props
        if prop.id in scene.prop_ids
    )
    components = [
        "photorealistic cinematic still",
        location_detail,
        character_details,
        scene.action,
        prop_details,
        scene.visual_description,
        scene.blocking,
        scene.camera.shot_type,
        scene.camera.angle,
        scene.camera.lens or "",
    ]
    return _fit_image_prompt(
        components,
        "Scene image prompt",
        minimum_components=4,
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
    negative_prompt: str = IMAGE_NEGATIVE_PROMPT,
) -> str:

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipe = get_pipeline()

    print("\nGenerating image...\n")

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or IMAGE_NEGATIVE_PROMPT,
        width=768,
        height=432,
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

    base_prompt = build_scene_image_prompt(
        storyboard=storyboard,
        scene=scene,
    )
    base_tokens = TOKENIZER(
        base_prompt,
        truncation=False,
        return_tensors="pt",
        verbose=False,
    )["input_ids"].shape[1]
    if base_tokens > 50:
        compact_tokens = TOKENIZER(
            base_prompt,
            truncation=True,
            max_length=50,
            return_tensors="pt",
            verbose=False,
        )["input_ids"][0]
        base_prompt = TOKENIZER.decode(
            compact_tokens,
            skip_special_tokens=True,
        )

    return _fit_image_prompt(
        [
            base_prompt,
            (
                "same exact people and faces as the reference image, "
                "same person, same clothes, same setting, continue action"
            ),
            "same clothes and accessories",
            "same setting and lighting",
            "no redesign",
        ],
        "Continuity image prompt",
        minimum_components=2,
    )

def generate_scene_image_from_reference(
    prompt: str,
    reference_image_path: str,
    output_path: str,
) -> str:

    from diffusers import StableDiffusionImg2ImgPipeline

    reference = Path(reference_image_path)

    if not reference.exists():
        raise FileNotFoundError(
            f"Reference image not found: {reference}"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("\nGenerating Scene 2 from continuity reference...\n")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )

    pipe.enable_model_cpu_offload()

    image = Image.open(reference).convert("RGB")

    result = pipe(
        prompt=prompt,
        negative_prompt=IMAGE_NEGATIVE_PROMPT,
        image=image,
        # A low denoise strength preserves the reference identity and
        # composition instead of redrawing the subject from scratch.
        strength=0.30,
        guidance_scale=6.5,
        num_inference_steps=35,
    ).images[0]

    result.save(output)

    print(f"Scene 2 image saved to: {output}")

    return str(output)