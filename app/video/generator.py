from pathlib import Path


def generate_scene_video(
    image_path: str,
    prompt: str,
    output_path: str,
    duration_seconds: int = 5,
) -> str:
    """
    Generate a video clip from a scene image and motion prompt.

    Video generation provider will be connected here.
    """

    image = Path(image_path)

    if not image.exists():
        raise FileNotFoundError(
            f"Scene image not found: {image}"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "Video generation provider has not been connected yet."
    )