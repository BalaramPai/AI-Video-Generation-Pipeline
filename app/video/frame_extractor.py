from pathlib import Path
import subprocess
import re

from app.video.composer import get_ffmpeg


def extract_last_frames(
    video_path: str,
    output_dir: str,
    frame_count: int = 5,
) -> list[str]:

    video = Path(video_path)

    if not video.exists():
        raise FileNotFoundError(
            f"Video not found: {video}"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg()

    # Get the total number of frames.
    probe_command = [
        ffmpeg,
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-f",
        "null",
        "-",
    ]

    result = subprocess.run(
        probe_command,
        capture_output=True,
        text=True,
    )

    matches = re.findall(
        r"frame=\s*(\d+)",
        result.stderr,
    )

    if not matches:
        raise RuntimeError(
            "Could not determine video frame count."
        )

    total_frames = int(matches[-1])

    if total_frames < frame_count:
        raise RuntimeError(
            f"Video contains only {total_frames} frames."
        )

    start_frame = total_frames - frame_count

    pattern = output / "frame_%02d.png"

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video),
        "-vf",
        f"select='between(n,{start_frame},{total_frames - 1})'",
        "-frames:v",
        str(frame_count),
        "-fps_mode",
        "vfr",
        str(pattern),
    ]

    subprocess.run(command, check=True)

    frames = sorted(output.glob("frame_*.png"))

    if len(frames) != frame_count:
        raise RuntimeError(
            f"Expected {frame_count} frames, "
            f"but extracted {len(frames)}."
        )

    return [str(frame) for frame in frames]