from pathlib import Path
import os
import shutil
import subprocess
import tempfile


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
    duration_seconds: int,
) -> str:
    if duration_seconds <= 0:
        raise ValueError("Video duration must be positive.")

    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(
        f".{output.stem}.trimmed{output.suffix}"
    )
    ffmpeg = get_ffmpeg()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-t",
        str(duration_seconds),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(temporary_output),
    ]

    try:
        subprocess.run(command, check=True)
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)

    return str(output)


def concatenate_clips(
    clip_paths: list[str],
    output_path: str,
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
            "-movflags",
            "+faststart",
            str(temporary_output),
        ]
        subprocess.run(command, check=True)
        temporary_output.replace(output)
    finally:
        temporary_output.unlink(missing_ok=True)
        if concat_file_path:
            Path(concat_file_path).unlink(missing_ok=True)
    return str(output)