import logging
from fractions import Fraction
from pathlib import Path
from threading import Event
from typing import Sequence

from PIL import Image, UnidentifiedImageError

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


def _validate_first_frame_image(image: Path) -> tuple[int, int]:
    if not image.is_file():
        raise FileNotFoundError(f"Scene image not found: {image}")
    if image.stat().st_size <= 0:
        raise ValueError(f"Scene image is empty: {image}")
    try:
        with Image.open(image) as candidate:
            candidate.verify()
        with Image.open(image) as candidate:
            width, height = candidate.size
            candidate.load()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(
            f"Scene image is not a readable image: {image}"
        ) from exc
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Scene image has invalid dimensions {width}x{height}: {image}"
        )
    return width, height


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
    image_width, image_height = _validate_first_frame_image(image)

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
                "Provider does not accept separate reference images; the "
                "continuity-adjusted first frame is the conditioning image "
                "sent to the provider",
                extra={
                    "provider": provider.name,
                    "model": getattr(provider, "_model", None),
                    "reference_images_requested": [
                        str(path) for path in reference_images
                    ],
                    "reference_images_submitted": [],
                    "first_frame": str(image),
                    "first_frame_is_uploaded_conditioning_image": True,
                },
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
            extra={
                "provider": provider.name,
                "model": getattr(provider, "_model", None),
                "output_path": str(output),
                "duration_seconds": duration_seconds,
                "aspect_ratio": aspect_ratio,
                "prompt": prompt,
                "first_frame": str(image),
                "first_frame_width": image_width,
                "first_frame_height": image_height,
                "last_frame": (
                    str(last_frame_path) if last_frame_path else None
                ),
                "reference_images_submitted": [
                    str(path) for path in reference_images
                ],
            },
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