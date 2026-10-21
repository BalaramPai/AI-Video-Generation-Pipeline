from pathlib import Path
import subprocess

from app.video.composer import get_ffmpeg, probe_video


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

    if frame_count <= 0:
        raise ValueError("Frame count must be positive.")

    total_frames = probe_video(video).frame_count
    if total_frames < frame_count:
        raise RuntimeError(
            f"Video contains only {total_frames} frames; "
            f"cannot extract {frame_count} ending frames."
        )

    ffmpeg = get_ffmpeg()
    for previous_frame in output.glob("frame_*.png"):
        previous_frame.unlink()

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
    empty_frames = [frame for frame in frames if frame.stat().st_size == 0]
    if empty_frames:
        raise RuntimeError(
            f"FFmpeg produced an empty reference frame: {empty_frames[0]}"
        )

    return [str(frame) for frame in frames]