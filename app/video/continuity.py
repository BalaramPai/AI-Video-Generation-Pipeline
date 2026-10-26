from pathlib import Path

from PIL import Image, UnidentifiedImageError


def build_continuity_reference(
    frame_paths: list[str],
) -> list[str]:

    if not frame_paths:
        raise ValueError(
            "At least one reference frame is required."
        )

    for frame in frame_paths:
        path = Path(frame)
        if not path.is_file():
            raise FileNotFoundError(
                f"Reference frame not found: {frame}"
            )
        if path.stat().st_size == 0:
            raise ValueError(f"Reference frame is empty: {frame}")
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                image.load()
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise ValueError(
                f"Reference frame is not a readable image: {frame}"
            ) from exc
        if width <= 0 or height <= 0:
            raise ValueError(
                f"Reference frame has invalid dimensions "
                f"{width}x{height}: {frame}"
            )

    return frame_paths


def get_primary_reference_frame(
    frame_paths: list[str],
) -> str:

    references = build_continuity_reference(frame_paths)

    # The final frame represents the latest visual state.
    return references[-1]