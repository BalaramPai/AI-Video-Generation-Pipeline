import logging
from pathlib import Path
from threading import Event
from typing import Sequence

from app.config import (
    VIDEO_PROVIDER,
)
from app.video.composer import create_motion_clip, trim_video
from app.video.providers.factory import create_video_provider


logger = logging.getLogger(__name__)


def generate_scene_video(
    image_path: str,
    prompt: str,
    output_path: str,
    duration_seconds: int = 5,
    aspect_ratio: str = "16:9",
    reference_images: Sequence[str] = (),
    last_frame_path: str | None = None,
    cancel_event: Event | None = None,
) -> str:
    image = Path(image_path)

    if not image.exists():
        raise FileNotFoundError(
            f"Scene image not found: {image}"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if VIDEO_PROVIDER == "ffmpeg":
        print("\nUsing FFmpeg still-image fallback.\n")
        return create_motion_clip(
            image_path=str(image),
            output_path=str(output),
            duration=duration_seconds,
        )

    provider = create_video_provider(VIDEO_PROVIDER)
    if not provider.capabilities.image_to_video:
        raise ValueError(
            f"VIDEO_PROVIDER={VIDEO_PROVIDER!r} does not support "
            "image-to-video generation."
        )
    if reference_images and not provider.capabilities.reference_images:
        logger.info(
            "Using the continuity-adjusted first frame without extra "
            "reference images",
            extra={"provider": provider.name},
        )
        reference_images = ()
    if last_frame_path and not provider.capabilities.last_frame:
        raise ValueError(
            f"VIDEO_PROVIDER={VIDEO_PROVIDER!r} does not support an explicit "
            "last frame."
        )
    if not provider.capabilities.first_frame:
        raise ValueError(
            f"VIDEO_PROVIDER={VIDEO_PROVIDER!r} cannot use the scene image "
            "as a starting frame."
        )

    logger.info(
        "Generating scene video",
        extra={"provider": provider.name, "output_path": str(output)},
    )
    result = provider.generate_video(
        prompt=prompt,
        output_path=output,
        duration_seconds=duration_seconds,
        first_frame=image,
        last_frame=Path(last_frame_path) if last_frame_path else None,
        reference_images=tuple(Path(path) for path in reference_images),
        aspect_ratio=aspect_ratio,
        cancel_event=cancel_event,
    )
    if result.duration_seconds != duration_seconds:
        trim_video(
            input_path=str(result.video_path),
            output_path=str(output),
            duration_seconds=duration_seconds,
        )
    return str(output)