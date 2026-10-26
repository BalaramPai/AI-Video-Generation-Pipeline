import logging
from pathlib import Path
import subprocess

from PIL import Image, UnidentifiedImageError

from app.video.composer import get_ffmpeg, probe_video


logger = logging.getLogger(__name__)


def _validated_image_dimensions(image_path: Path) -> tuple[int, int]:
    if not image_path.is_file():
        raise ValueError(f"Extracted frame was not created: {image_path}")
    if image_path.stat().st_size <= 0:
        raise ValueError(f"Extracted frame is empty: {image_path}")
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
            image.load()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(
            f"Extracted frame is not a readable image: {image_path}"
        ) from exc
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Extracted frame has invalid dimensions {width}x{height}: "
            f"{image_path}"
        )
    return width, height


def extract_last_frames(
    video_path: str,
    output_dir: str,
    frame_count: int = 5,
) -> list[str]:
    """Extract up to ``frame_count`` valid trailing frames, newest last.

    Extraction attempts decoded frame indexes from newest to oldest. A
    damaged/unreadable frame is skipped in favor of the closest earlier valid
    frame; no reference frames from an earlier run are retained.
    """
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if frame_count <= 0:
        raise ValueError("Frame count must be positive.")

    video_info = probe_video(video)
    total_frames = video_info.frame_count
    if total_frames <= 0:
        raise RuntimeError(f"Video contains no decoded frames: {video}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for previous_frame in output.glob("frame_*.png"):
        previous_frame.unlink()

    ffmpeg = get_ffmpeg()
    valid_frames: list[tuple[int, Path]] = []

    for frame_index in range(total_frames - 1, -1, -1):
        frame_path = output / f"frame_{frame_index:06d}.png"
        command = [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-frames:v",
            "1",
            "-fps_mode",
            "vfr",
            str(frame_path),
        ]
        extraction_error: subprocess.CalledProcessError | None = None
        try:
            subprocess.run(command, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            extraction_error = exc
        try:
            dimensions = _validated_image_dimensions(frame_path)
        except ValueError as exc:
            frame_path.unlink(missing_ok=True)
            logger.warning(
                "Trailing video frame failed image validation; trying the "
                "nearest earlier frame",
                extra={
                    "source_video": str(video),
                    "frame_index": frame_index,
                    "error": (
                        str(extraction_error)
                        if extraction_error is not None
                        else str(exc)
                    ),
                },
            )
            continue
        if extraction_error is not None:
            logger.warning(
                "FFmpeg reported a decode error, but the selected frame "
                "passed image validation",
                extra={
                    "source_video": str(video),
                    "frame_index": frame_index,
                    "error": str(extraction_error),
                },
            )

        timestamp = (
            video_info.frame_rate.denominator * frame_index
            / video_info.frame_rate.numerator
        )
        logger.info(
            "Extracted and verified a previous-scene frame",
            extra={
                "source_video": str(video),
                "destination_image": str(frame_path),
                "frame_index": frame_index,
                "timestamp_seconds": timestamp,
                "image_width": dimensions[0],
                "image_height": dimensions[1],
            },
        )
        valid_frames.append((frame_index, frame_path))
        if len(valid_frames) == frame_count:
            break

    if not valid_frames:
        raise RuntimeError(
            f"Could not extract a valid ending reference frame from {video}."
        )

    valid_frames.reverse()
    if valid_frames[-1][0] != total_frames - 1:
        logger.warning(
            "Using an earlier valid frame because the final video frame "
            "could not be decoded or validated",
            extra={
                "source_video": str(video),
                "requested_final_frame_index": total_frames - 1,
                "fallback_frame_index": valid_frames[-1][0],
                "fallback_image": str(valid_frames[-1][1]),
            },
        )
    if len(valid_frames) < min(frame_count, total_frames):
        logger.warning(
            "Fewer than the requested trailing reference frames were valid",
            extra={
                "source_video": str(video),
                "requested_frame_count": frame_count,
                "valid_frame_count": len(valid_frames),
            },
        )

    return [str(path) for _, path in valid_frames]
