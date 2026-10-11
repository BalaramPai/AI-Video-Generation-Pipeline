import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class VideoInfo:
    frame_rate: Fraction
    frame_count: int
    duration_seconds: Fraction
    video_codec: str
    width: int
    height: int
    pixel_format: str
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None


def get_ffmpeg() -> str:
    configured_path = os.getenv("FFMPEG_PATH")
    if configured_path:
        ffmpeg = Path(configured_path)
        if ffmpeg.is_file():
            return str(ffmpeg)
        raise FileNotFoundError(
            f"FFMPEG_PATH does not point to an executable file: {ffmpeg}"
        )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    local_app_data = os.getenv("LOCALAPPDATA")
    if os.name == "nt" and local_app_data:
        winget_packages = (
            Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        )
        installed_versions = sorted(
            winget_packages.glob(
                "Gyan.FFmpeg*/ffmpeg-*/bin/ffmpeg.exe"
            ),
            reverse=True,
        )
        if installed_versions:
            return str(installed_versions[0])
    raise FileNotFoundError(
        "FFmpeg was not found on PATH or in the standard Windows WinGet "
        "location. Install FFmpeg or set FFMPEG_PATH to its executable."
    )


def get_ffprobe() -> str:
    ffmpeg = Path(get_ffmpeg())
    ffprobe = ffmpeg.with_name(
        "ffprobe.exe" if ffmpeg.suffix.lower() == ".exe" else "ffprobe"
    )
    if ffprobe.is_file():
        return str(ffprobe)

    discovered = shutil.which("ffprobe")
    if discovered:
        return discovered
    raise FileNotFoundError(
        "FFprobe was not found beside FFmpeg or on PATH. Install the FFmpeg "
        "suite, including ffprobe, to verify video frame timing."
    )


def _parse_fraction(value: object, description: str) -> Fraction:
    if not isinstance(value, str) or value in {"", "N/A", "0/0"}:
        raise ValueError(f"Video {description} is missing or invalid: {value!r}")
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(
            f"Video {description} is missing or invalid: {value!r}"
        ) from exc
    if parsed <= 0:
        raise ValueError(f"Video {description} must be positive: {value!r}")
    return parsed


def probe_video(video_path: str | Path) -> VideoInfo:
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    command = [
        get_ffprobe(),
        "-v",
        "error",
        "-count_frames",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Could not inspect video timing: {path}") from exc

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"FFprobe returned invalid metadata for {path}") from exc
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise RuntimeError(f"FFprobe returned no stream metadata for {path}")

    video_stream = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "video"
        ),
        None,
    )
    if not isinstance(video_stream, dict):
        raise RuntimeError(f"Video contains no video stream: {path}")

    frame_rate = _parse_fraction(
        video_stream.get("avg_frame_rate")
        or video_stream.get("r_frame_rate"),
        "effective frame rate",
    )
    frame_count_value = video_stream.get("nb_read_frames")
    if frame_count_value in {None, "N/A"}:
        frame_count_value = video_stream.get("nb_frames")
    try:
        frame_count = int(frame_count_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"FFprobe could not determine the decoded frame count for {path}"
        ) from exc
    if frame_count <= 0:
        raise RuntimeError(f"Video contains no decoded frames: {path}")

    try:
        duration = _parse_fraction(
            str(video_stream.get("duration") or ""),
            "stream duration",
        )
    except ValueError:
        duration = Fraction(frame_count, 1) / frame_rate

    audio_stream = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "audio"
        ),
        None,
    )
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    audio_codec: str | None = None
    if isinstance(audio_stream, dict):
        audio_codec = str(audio_stream.get("codec_name") or "")
        try:
            audio_sample_rate = int(audio_stream.get("sample_rate"))
            audio_channels = int(audio_stream.get("channels"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"FFprobe returned invalid audio metadata for {path}"
            ) from exc

    try:
        width = int(video_stream.get("width"))
        height = int(video_stream.get("height"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"FFprobe returned invalid video dimensions for {path}"
        ) from exc

    return VideoInfo(
        frame_rate=frame_rate,
        frame_count=frame_count,
        duration_seconds=duration,
        video_codec=str(video_stream.get("codec_name") or ""),
        width=width,
        height=height,
        pixel_format=str(video_stream.get("pix_fmt") or ""),
        audio_codec=audio_codec,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
    )


def _round_fraction_half_up(value: Fraction) -> int:
    """Round positive frame boundaries to nearest frame; ties round up."""
    return (2 * value.numerator + value.denominator) // (
        2 * value.denominator
    )


def calculate_scene_frame_counts(
    durations_seconds: Sequence[int | float | str],
    frame_rate: Fraction | str,
) -> list[int]:
    """Round cumulative boundaries; half-frame ties round up.

    A scene receives the difference between its rounded end and start
    boundaries. Thus scene frame counts sum to the rounded project duration.
    """
    fps = (
        frame_rate
        if isinstance(frame_rate, Fraction)
        else _parse_fraction(frame_rate, "target frame rate")
    )
    if fps <= 0:
        raise ValueError("Target frame rate must be positive.")

    frame_counts: list[int] = []
    elapsed = Fraction(0)
    previous_boundary = 0
    for duration in durations_seconds:
        try:
            scene_duration = Fraction(str(duration))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(
                f"Invalid scene duration: {duration!r}"
            ) from exc
        if scene_duration <= 0:
            raise ValueError(
                f"Scene duration must be positive: {duration!r}"
            )
        elapsed += scene_duration
        boundary = _round_fraction_half_up(elapsed * fps)
        scene_frames = boundary - previous_boundary
        if scene_frames <= 0:
            raise ValueError(
                f"Scene duration {duration!r}s is shorter than one "
                f"representable frame at {fps} fps."
            )
        frame_counts.append(scene_frames)
        previous_boundary = boundary
    return frame_counts


def normalize_video_clip(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_frame_rate: Fraction | str,
    target_frame_count: int,
) -> VideoInfo:
    """Encode a clip to an exact CFR frame count and retain optional audio."""
    source = Path(input_path)
    source_info = probe_video(source)
    fps = (
        target_frame_rate
        if isinstance(target_frame_rate, Fraction)
        else _parse_fraction(target_frame_rate, "target frame rate")
    )
    if fps <= 0:
        raise ValueError("Target frame rate must be positive.")
    if target_frame_count <= 0:
        raise ValueError("Target frame count must be positive.")

    target_duration = Fraction(target_frame_count, 1) / fps
    available_duration = Fraction(
        source_info.frame_count, 1
    ) / source_info.frame_rate
    if available_duration < target_duration:
        raise ValueError(
            f"Video {source} is too short to normalize safely: "
            f"{source_info.frame_count} frames at {source_info.frame_rate} fps "
            f"provide {float(available_duration):.6f}s, but "
            f"{target_frame_count} frames at {fps} fps require "
            f"{float(target_duration):.6f}s."
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(
        f".{output.stem}.normalized{output.suffix}"
    )
    frame_rate_text = f"{fps.numerator}/{fps.denominator}"
    command = [
        get_ffmpeg(),
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        (
            f"fps=fps={frame_rate_text}:round=near,"
            f"trim=end_frame={target_frame_count},"
            "setpts=PTS-STARTPTS"
        ),
        "-frames:v",
        str(target_frame_count),
        "-r",
        frame_rate_text,
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
    ]
    if source_info.audio_codec is not None:
        command.extend(
            [
                "-map",
                "0:a:0",
                "-af",
                (
                    f"atrim=duration={float(target_duration):.9f},"
                    "asetpts=PTS-STARTPTS"
                ),
                "-c:a",
                "aac",
                "-ar",
                "48000",
                "-ac",
                "2",
            ]
        )
    else:
        command.append("-an")
    command.extend(
        [
            "-movflags",
            "+faststart",
            str(temporary_output),
        ]
    )

    try:
        subprocess.run(command, check=True)
        normalized_info = probe_video(temporary_output)
        if (
            normalized_info.frame_count != target_frame_count
            or normalized_info.frame_rate != fps
        ):
            raise RuntimeError(
                f"FFmpeg produced {normalized_info.frame_count} frames at "
                f"{normalized_info.frame_rate} fps; expected "
                f"{target_frame_count} frames at {fps} fps."
            )
        temporary_output.replace(output)
        return normalized_info
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"FFmpeg failed to normalize video clip: {source}"
        ) from exc
    finally:
        temporary_output.unlink(missing_ok=True)


def create_motion_clip(
    image_path: str,
    output_path: str,
    duration: int = 5,
) -> str:

    image = Path(image_path)

    if not image.exists():
        raise FileNotFoundError(
            f"Image not found: {image}"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg()

    command = [
        ffmpeg,
        "-y",
        "-loop",
        "1",
        "-i",
        str(image),
        "-t",
        str(duration),
        "-vf",
        (
            "scale=768:432,"
            "zoompan="
            "z='min(zoom+0.0015,1.08)':"
            "x='iw/2-(iw/zoom/2)':"
            "y='ih/2-(ih/zoom/2)':"
            "d=1:"
            "s=768x432:"
            "fps=24"
        ),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]

    subprocess.run(command, check=True)

    return str(output)


def trim_video(
    input_path: str,
    output_path: str,
    duration_seconds: int | float,
) -> str:
    output = Path(output_path)
    source_info = probe_video(input_path)
    frame_count = calculate_scene_frame_counts(
        [duration_seconds],
        source_info.frame_rate,
    )[0]
    normalize_video_clip(
        input_path,
        output_path,
        target_frame_rate=source_info.frame_rate,
        target_frame_count=frame_count,
    )
    return str(output)


def concatenate_clips(
    clip_paths: list[str],
    output_path: str,
    *,
    scene_durations_seconds: Sequence[int | float | str] | None = None,
    frame_rate: Fraction | str | None = None,
) -> str:

    if not clip_paths:
        raise ValueError("At least one video clip is required.")
    missing_clips = [
        Path(clip) for clip in clip_paths if not Path(clip).is_file()
    ]
    if missing_clips:
        raise FileNotFoundError(f"Video clip not found: {missing_clips[0]}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg()
    expected_frame_count: int | None = None
    target_frame_rate: Fraction | None = None

    if scene_durations_seconds is not None:
        if len(scene_durations_seconds) != len(clip_paths):
            raise ValueError(
                "The number of scene durations must match the number of clips."
            )
        if frame_rate is None:
            raise ValueError(
                "A target frame rate is required when validating scene "
                "durations."
            )
        target_frame_rate = (
            frame_rate
            if isinstance(frame_rate, Fraction)
            else _parse_fraction(frame_rate, "target frame rate")
        )
        expected_counts = calculate_scene_frame_counts(
            scene_durations_seconds,
            target_frame_rate,
        )
        expected_frame_count = sum(expected_counts)
        clip_infos: list[VideoInfo] = []
        for index, (clip, frame_count) in enumerate(
            zip(clip_paths, expected_counts)
        ):
            info = probe_video(clip)
            if (
                info.frame_count != frame_count
                or info.frame_rate != target_frame_rate
            ):
                raise ValueError(
                    f"Scene clip {index + 1} is not normalized: expected "
                    f"{frame_count} frames at {target_frame_rate} fps, got "
                    f"{info.frame_count} frames at {info.frame_rate} fps."
                )
            clip_infos.append(info)

        signatures = {
            (
                info.video_codec,
                info.width,
                info.height,
                info.pixel_format,
                info.frame_rate,
                info.audio_codec,
                info.audio_sample_rate,
                info.audio_channels,
            )
            for info in clip_infos
        }
        if len(signatures) != 1:
            raise ValueError(
                "Scene clips do not have matching normalized video/audio "
                "streams; refusing a potentially mistimed concatenation."
            )

    temporary_output = output.with_name(
        f".{output.stem}.joined{output.suffix}"
    )
    concat_file_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".ffconcat",
            dir=output.parent,
            delete=False,
        ) as concat_file:
            concat_file_path = concat_file.name
            for clip in clip_paths:
                absolute = Path(clip).resolve().as_posix()
                escaped = absolute.replace("'", "'\\''")
                concat_file.write(f"file '{escaped}'\n")

        command = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file_path,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
        ]
        if expected_frame_count is None:
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                ]
            )
        else:
            command.extend(["-c", "copy"])
        command.extend(
            [
                "-movflags",
                "+faststart",
                str(temporary_output),
            ]
        )
        subprocess.run(command, check=True)
        if expected_frame_count is not None and target_frame_rate is not None:
            final_info = probe_video(temporary_output)
            if (
                final_info.frame_count != expected_frame_count
                or final_info.frame_rate != target_frame_rate
            ):
                raise RuntimeError(
                    f"Concatenated video has {final_info.frame_count} frames "
                    f"at {final_info.frame_rate} fps; expected "
                    f"{expected_frame_count} frames at {target_frame_rate} fps."
                )
        temporary_output.replace(output)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"FFmpeg failed to concatenate video clips into {output}."
        ) from exc
    finally:
        temporary_output.unlink(missing_ok=True)
        if concat_file_path:
            Path(concat_file_path).unlink(missing_ok=True)
    return str(output)