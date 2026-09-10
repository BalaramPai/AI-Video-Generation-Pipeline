from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

from app.storyboard.models import (
    Storyboard,
    Scene,
)

from transformers import CLIPTokenizer
from PIL import Image



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

    # Main character
    main_character = characters[0]

    appearance = main_character.appearance

    prompt = (
        "photorealistic cinematic photo, "
        f"{location.name}, "
        f"{main_character.name}, "
        f"age {appearance.age}, "
        f"{appearance.hair} hair, "
        f"{appearance.clothing}, "
        f"{scene.action}, "
        f"{scene.camera.shot_type}, "
        f"{scene.camera.angle}"
    )

    # Validate against the ACTUAL tokenizer.
    tokens = TOKENIZER(
        prompt,
        truncation=False,
        return_tensors="pt",
        verbose=False,
    )["input_ids"]

    token_count = tokens.shape[1]

    if token_count > 75:
        raise ValueError(
            f"Image prompt is too long for SD 1.5 CLIP: "
            f"{token_count} tokens. Maximum is 75."
        )

    return prompt


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
        width=768,
        height=432,
        num_inference_steps=25,
        guidance_scale=7.5,
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

    prompt = (
        f"{base_prompt}, "
        "same person, same clothes, same setting, "
        "continue action"
    )

    tokens = TOKENIZER(
        prompt,
        truncation=False,
        return_tensors="pt",
        verbose=False,
    )["input_ids"]

    token_count = tokens.shape[1]

    if token_count > 75:
        raise ValueError(
            f"Continuity image prompt is too long: "
            f"{token_count} tokens."
        )

    return prompt

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
        image=image,
        strength=0.55,
        guidance_scale=7.5,
        num_inference_steps=25,
    ).images[0]

    result.save(output)

    print(f"Scene 2 image saved to: {output}")

    return str(output)