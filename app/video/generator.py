import logging
from fractions import Fraction
from pathlib import Path
from threading import Event
from typing import Sequence

from app.config import (
    VIDEO_PROVIDER,
)
from app.video.composer import (
    calculate_scene_frame_counts,
    create_motion_clip,
    normalize_video_clip,
    probe_video,
)
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
    target_frame_rate: Fraction | str | None = None,
    target_frame_count: int | None = None,
    normalize_output: bool = True,
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
        generated_path = create_motion_clip(
            image_path=str(image),
            output_path=str(output),
            duration=duration_seconds,
        )
    else:
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
                f"VIDEO_PROVIDER={VIDEO_PROVIDER!r} does not support an "
                "explicit last frame."
            )
        if not provider.capabilities.first_frame:
            raise ValueError(
                f"VIDEO_PROVIDER={VIDEO_PROVIDER!r} cannot use the scene "
                "image as a starting frame."
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
        generated_path = result.video_path

    if (target_frame_rate is None) != (target_frame_count is None):
        raise ValueError(
            "target_frame_rate and target_frame_count must be provided "
            "together."
        )
    if not normalize_output:
        return str(generated_path)
    if target_frame_rate is None:
        target_frame_rate = probe_video(generated_path).frame_rate
    if target_frame_count is None:
        target_frame_count = calculate_scene_frame_counts(
            [duration_seconds],
            target_frame_rate,
        )[0]

    normalize_video_clip(
        generated_path,
        output,
        target_frame_rate=target_frame_rate,
        target_frame_count=target_frame_count,
    )
    return str(output)