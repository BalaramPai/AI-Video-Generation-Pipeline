import base64
import mimetypes
import time
from pathlib import Path

import httpx

from app.config import (
    RUNWAY_API_BASE,
    RUNWAY_API_SECRET,
    RUNWAY_MODEL,
    RUNWAY_POLL_INTERVAL_SECONDS,
    RUNWAY_TIMEOUT_SECONDS,
    VIDEO_PROVIDER,
)
from app.video.composer import create_motion_clip


RUNWAY_VERSION = "2024-11-06"


def _image_data_uri(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError(
            f"Unsupported image format for Runway: {image_path.suffix}"
        )

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    data_uri = f"data:{mime_type};base64,{encoded}"

    if len(data_uri.encode("utf-8")) > 5 * 1024 * 1024:
        raise ValueError(
            "The image is too large for Runway's data-URI input limit."
        )

    return data_uri


def _runway_headers() -> dict[str, str]:
    if not RUNWAY_API_SECRET:
        raise RuntimeError(
            "RUNWAYML_API_SECRET is required when VIDEO_PROVIDER=runway. "
            "Add it to .env or choose VIDEO_PROVIDER=ffmpeg."
        )

    return {
        "Authorization": f"Bearer {RUNWAY_API_SECRET}",
        "Content-Type": "application/json",
        "X-Runway-Version": RUNWAY_VERSION,
    }


def _raise_for_runway_response(response: httpx.Response) -> None:
    if response.is_success:
        return

    detail = response.text
    raise RuntimeError(
        f"Runway API request failed ({response.status_code}): {detail}"
    )


def _generate_with_runway(
    image: Path,
    prompt: str,
    output: Path,
    duration_seconds: int,
) -> str:
    requested_duration = 10 if duration_seconds >= 10 else 5
    payload = {
        "model": RUNWAY_MODEL,
        "promptImage": _image_data_uri(image),
        "promptText": prompt,
        "ratio": "1280:720",
        "duration": requested_duration,
    }
    headers = _runway_headers()

    with httpx.Client(timeout=120) as client:
        response = client.post(
            f"{RUNWAY_API_BASE}/v1/image_to_video",
            headers=headers,
            json=payload,
        )
        _raise_for_runway_response(response)
        task_id = response.json().get("id")

        if not task_id:
            raise RuntimeError(
                f"Runway returned no task ID: {response.text}"
            )

        deadline = time.monotonic() + RUNWAY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            time.sleep(RUNWAY_POLL_INTERVAL_SECONDS)
            status_response = client.get(
                f"{RUNWAY_API_BASE}/v1/tasks/{task_id}",
                headers=headers,
            )
            _raise_for_runway_response(status_response)
            task = status_response.json()
            status = task.get("status")

            if status == "SUCCEEDED":
                output_urls = task.get("output", [])
                if not output_urls:
                    raise RuntimeError(
                        f"Runway task succeeded without an output: {task}"
                    )

                video_response = client.get(output_urls[0])
                _raise_for_runway_response(video_response)
                output.write_bytes(video_response.content)
                return str(output)

            if status in {"FAILED", "CANCELED"}:
                raise RuntimeError(f"Runway task {status.lower()}: {task}")

            print(f"Runway task {task_id}: {status or 'unknown'}")

    raise TimeoutError(
        f"Runway task {task_id} did not finish within "
        f"{RUNWAY_TIMEOUT_SECONDS:g} seconds."
    )


def generate_scene_video(
    image_path: str,
    prompt: str,
    output_path: str,
    duration_seconds: int = 5,
) -> str:
    image = Path(image_path)

    if not image.exists():
        raise FileNotFoundError(
            f"Scene image not found: {image}"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if VIDEO_PROVIDER == "runway":
        print(f"\nGenerating real video with Runway: {output}\n")
        return _generate_with_runway(
            image=image,
            prompt=prompt,
            output=output,
            duration_seconds=duration_seconds,
        )

    if VIDEO_PROVIDER == "ffmpeg":
        print("\nUsing FFmpeg still-image fallback.\n")
        return create_motion_clip(
            image_path=str(image),
            output_path=str(output),
            duration=duration_seconds,
        )

    raise ValueError(
        f"Unsupported VIDEO_PROVIDER={VIDEO_PROVIDER!r}. "
        "Use 'runway' or 'ffmpeg'."
    )