from pathlib import Path
import shutil
import subprocess


FFMPEG_PATH = (
    r"C:\Users\91636\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0.1-full_build-shared\bin\ffmpeg.exe"
)


def get_ffmpeg() -> str:
    ffmpeg = Path(FFMPEG_PATH)

    if not ffmpeg.exists():
        raise FileNotFoundError(
            f"FFmpeg executable not found: {ffmpeg}"
        )

    return str(ffmpeg)


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


def concatenate_clips(
    clip_paths: list[str],
    output_path: str,
) -> str:

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg()

    concat_file = output.parent / "concat.txt"

    with open(concat_file, "w", encoding="utf-8") as file:
        for clip in clip_paths:
            absolute = Path(clip).resolve()
            file.write(f"file '{absolute.as_posix()}'\n")

    command = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(output),
    ]

    subprocess.run(command, check=True)

    return str(output)