import base64
import logging
import mimetypes
import random
import time
from contextlib import nullcontext
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import Event
from typing import Sequence
from urllib.parse import quote, urljoin, urlsplit

import httpx

from app.video.providers.base import (
    VideoGenerationCancelled,
    VideoGenerationProvider,
    VideoGenerationResult,
    VideoGenerationTimeout,
    VideoProviderCapabilities,
    VideoProviderError,
)


logger = logging.getLogger(__name__)

RUNWAY_API_BASE = "https://api.dev.runwayml.com"
RUNWAY_API_VERSION = "2024-11-06"
RUNWAY_PROMPT_MAX_CHARACTERS = 1000
_RUNWAY_MODELS = {
    "gen4.5": (2, 10),
    "grok_imagine_1_5": (1, 15),
}
_GEN45_RATIOS = {
    "16:9": "1280:720",
    "9:16": "720:1280",
    "4:3": "1104:832",
    "3:4": "832:1104",
    "1:1": "960:960",
    "21:9": "1584:672",
}
_GROK_RATIOS = ("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3")
_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
_MAX_DATA_URI_BYTES = 5 * 1024 * 1024
_TERMINAL_FAILURE_STATES = {"FAILED", "CANCELLED", "CANCELED"}


def _utf16_length(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _fit_runway_prompt(prompt: str) -> str:
    normalized = " ".join(prompt.split())
    if _utf16_length(normalized) <= RUNWAY_PROMPT_MAX_CHARACTERS:
        return normalized

    fitted: list[str] = []
    length = 0
    for character in normalized:
        character_length = _utf16_length(character)
        if length + character_length > RUNWAY_PROMPT_MAX_CHARACTERS - 3:
            break
        fitted.append(character)
        length += character_length

    compact = "".join(fitted).rstrip()
    if " " in compact:
        compact = compact.rsplit(" ", 1)[0].rstrip()
    return f"{compact}..."


def _image_data_uri(image_path: Path) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Runway image input not found: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in _ALLOWED_IMAGE_TYPES:
        raise ValueError(
            f"Unsupported image format for Runway: {path.suffix or 'unknown'}"
        )

    data_uri = (
        f"data:{mime_type};base64,"
        f"{base64.b64encode(path.read_bytes()).decode('ascii')}"
    )
    if len(data_uri.encode("utf-8")) > _MAX_DATA_URI_BYTES:
        raise ValueError("Runway image data URI exceeds the 5 MiB input limit.")
    return data_uri


class RunwayVideoGenerationProvider(VideoGenerationProvider):
    name = "runway"
    capabilities = VideoProviderCapabilities(
        text_to_video=False,
        image_to_video=True,
        reference_images=False,
        first_frame=True,
        last_frame=False,
        video_extension=False,
        supported_aspect_ratios=tuple(_GEN45_RATIOS),
    )

    def __init__(
        self,
        *,
        api_secret: str | None,
        model: str = "gen4.5",
        poll_interval_seconds: float = 5,
        timeout_seconds: float = 900,
        client: httpx.Client | None = None,
        api_base: str = RUNWAY_API_BASE,
    ) -> None:
        if not api_secret or not api_secret.strip():
            raise ValueError(
                "RUNWAYML_API_SECRET is required when VIDEO_PROVIDER=runway."
            )
        if model not in _RUNWAY_MODELS:
            allowed = ", ".join(_RUNWAY_MODELS)
            raise ValueError(
                f"Unsupported RUNWAY_MODEL={model!r}. Choose one of: {allowed}."
            )
        if poll_interval_seconds < 0:
            raise ValueError(
                "RUNWAY_POLL_INTERVAL_SECONDS must not be negative."
            )
        if timeout_seconds <= 0:
            raise ValueError("RUNWAY_TIMEOUT_SECONDS must be greater than zero.")

        self._api_secret = api_secret.strip()
        self._model = model
        self._poll_interval_seconds = poll_interval_seconds
        self._timeout_seconds = timeout_seconds
        self._client = client
        self._api_base = api_base.rstrip("/")
        self.capabilities = VideoProviderCapabilities(
            text_to_video=False,
            image_to_video=True,
            reference_images=False,
            first_frame=True,
            last_frame=False,
            video_extension=False,
            supported_aspect_ratios=(
                tuple(_GEN45_RATIOS)
                if model == "gen4.5"
                else _GROK_RATIOS
            ),
        )

    def generate_video(
        self,
        *,
        prompt: str,
        output_path: Path,
        duration_seconds: int | float | str | None = None,
        first_frame: Path | None = None,
        last_frame: Path | None = None,
        reference_images: Sequence[Path] = (),
        aspect_ratio: str = "16:9",
        cancel_event: Event | None = None,
    ) -> VideoGenerationResult:
        if not prompt.strip():
            raise ValueError("Video prompt must not be empty.")
        if first_frame is None:
            raise ValueError(
                "Runway image-to-video generation requires a first frame."
            )
        if last_frame is not None:
            raise ValueError(
                "Runway image-to-video does not support an explicit last frame."
            )
        if reference_images:
            raise ValueError(
                "Runway image-to-video does not support separate reference "
                "images; pass the continuity-adjusted first frame instead."
            )

        requested_duration = self._normalize_duration(duration_seconds)
        minimum, maximum = _RUNWAY_MODELS[self._model]
        if requested_duration > maximum:
            raise ValueError(
                f"RUNWAY_MODEL={self._model!r} supports scene durations from "
                f"{minimum} to {maximum} seconds; received "
                f"{requested_duration}. Split longer projects into scenes."
            )
        generation_duration = max(requested_duration, minimum)
        if aspect_ratio not in self.capabilities.supported_aspect_ratios:
            raise ValueError(
                f"RUNWAY_MODEL={self._model!r} does not support aspect ratio "
                f"{aspect_ratio!r} for image-to-video."
            )

        payload: dict[str, str | int] = {
            "model": self._model,
            "promptImage": _image_data_uri(Path(first_frame)),
            "promptText": _fit_runway_prompt(prompt),
            "duration": generation_duration,
        }
        if self._model == "gen4.5":
            payload["ratio"] = _GEN45_RATIOS[aspect_ratio]
        else:
            payload["resolution"] = "720p"

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self._timeout_seconds
        headers = {
            "Authorization": f"Bearer {self._api_secret}",
            "Content-Type": "application/json",
            "X-Runway-Version": RUNWAY_API_VERSION,
        }

        logger.info(
            "Runway image-to-video generation requested",
            extra={
                "provider": self.name,
                "model": self._model,
                "duration_seconds": requested_duration,
                "generation_duration_seconds": generation_duration,
                "aspect_ratio": aspect_ratio,
            },
        )

        client_context = (
            nullcontext(self._client)
            if self._client is not None
            else httpx.Client(timeout=min(60.0, self._timeout_seconds))
        )
        with client_context as client:
            assert client is not None
            task_id = self._submit(client, payload, headers)
            task = self._poll(client, task_id, headers, deadline, cancel_event)
            output_url = self._get_output_url(task)
            self._download_video(client, output_url, output, deadline)

        return VideoGenerationResult(
            video_path=output,
            provider=self.name,
            generation_id=task_id,
            duration_seconds=generation_duration,
            metadata={
                "model": self._model,
                "aspect_ratio": aspect_ratio,
                "requested_duration_seconds": requested_duration,
            },
        )

    @staticmethod
    def _normalize_duration(
        duration_seconds: int | float | str | None,
    ) -> int:
        if duration_seconds is None or isinstance(duration_seconds, bool):
            raise ValueError("Scene duration must be a positive whole number.")
        if not isinstance(duration_seconds, (int, float, str)):
            raise ValueError("Scene duration must be a positive whole number.")
        try:
            duration = Decimal(str(duration_seconds).strip())
        except InvalidOperation as exc:
            raise ValueError(
                "Scene duration must be a positive whole number."
            ) from exc
        if (
            not duration.is_finite()
            or duration <= 0
            or duration != duration.to_integral_value()
        ):
            raise ValueError("Scene duration must be a positive whole number.")
        return int(duration)

    def _submit(
        self,
        client: httpx.Client,
        payload: dict[str, str | int],
        headers: dict[str, str],
    ) -> str:
        try:
            response = client.post(
                f"{self._api_base}/v1/image_to_video",
                headers=headers,
                json=payload,
            )
        except httpx.RequestError as exc:
            raise VideoProviderError(
                "submit",
                f"network request failed: {type(exc).__name__}",
                retryable=True,
                provider=self.name,
            ) from exc

        body = self._response_json(response, "submit")
        self._raise_for_api_error(response, body, "submit")
        task_id = body.get("id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise VideoProviderError(
                "submit",
                "API response did not contain a task ID",
                retryable=False,
                provider=self.name,
            )
        logger.info(
            "Runway generation task submitted",
            extra={"provider": self.name, "generation_id": task_id},
        )
        return task_id

    def _poll(
        self,
        client: httpx.Client,
        task_id: str,
        headers: dict[str, str],
        deadline: float,
        cancel_event: Event | None,
    ) -> dict[str, Any]:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise VideoGenerationCancelled(provider=self.name)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VideoGenerationTimeout(
                    self._timeout_seconds,
                    provider=self.name,
                )
            jitter = random.uniform(
                0.0,
                min(1.0, self._poll_interval_seconds * 0.1),
            )
            wait_seconds = min(
                self._poll_interval_seconds + jitter,
                remaining,
            )
            if cancel_event is not None:
                if cancel_event.wait(wait_seconds):
                    raise VideoGenerationCancelled(provider=self.name)
            elif wait_seconds:
                time.sleep(wait_seconds)

            if time.monotonic() >= deadline:
                raise VideoGenerationTimeout(
                    self._timeout_seconds,
                    provider=self.name,
                )

            try:
                response = client.get(
                    f"{self._api_base}/v1/tasks/{quote(task_id, safe='')}",
                    headers=headers,
                    timeout=min(60.0, remaining),
                )
            except httpx.RequestError as exc:
                raise VideoProviderError(
                    "poll",
                    f"network request failed: {type(exc).__name__}",
                    retryable=True,
                    provider=self.name,
                ) from exc

            body = self._response_json(response, "poll")
            self._raise_for_api_error(response, body, "poll")
            status = body.get("status")
            if status == "SUCCEEDED":
                return body
            if status in _TERMINAL_FAILURE_STATES:
                reason = body.get("failure") or "generation task failed"
                failure_code = body.get("failureCode")
                raise VideoProviderError(
                    "generation",
                    f"{status.lower()}: {reason}"
                    + (
                        f" (failure code: {failure_code})"
                        if failure_code
                        else ""
                    ),
                    retryable=False,
                    provider=self.name,
                )
            if status not in {"PENDING", "THROTTLED", "RUNNING"}:
                raise VideoProviderError(
                    "poll",
                    f"API returned an unknown task status: {status!r}",
                    retryable=False,
                    provider=self.name,
                )

    def _download_video(
        self,
        client: httpx.Client,
        url: str,
        output: Path,
        deadline: float,
    ) -> None:
        temporary_output = output.with_name(
            f".{output.stem}.runway{output.suffix}"
        )
        current_url = url
        try:
            for redirect_count in range(4):
                parsed = urlsplit(current_url)
                if parsed.scheme != "https" or not parsed.hostname:
                    raise VideoProviderError(
                        "download",
                        "API returned a non-HTTPS video URL",
                        retryable=False,
                        provider=self.name,
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise VideoGenerationTimeout(
                        self._timeout_seconds,
                        provider=self.name,
                    )
                try:
                    response = client.get(
                        current_url,
                        timeout=min(60.0, remaining),
                        follow_redirects=False,
                    )
                except httpx.RequestError as exc:
                    raise VideoProviderError(
                        "download",
                        f"network request failed: {type(exc).__name__}",
                        retryable=True,
                        provider=self.name,
                    ) from exc

                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location or redirect_count == 3:
                        raise VideoProviderError(
                            "download",
                            "video download exceeded the redirect limit",
                            retryable=False,
                            provider=self.name,
                        )
                    current_url = urljoin(current_url, location)
                    continue
                if not response.is_success:
                    self._raise_for_api_error(response, {}, "download")
                if not response.content:
                    raise VideoProviderError(
                        "download",
                        "video download returned an empty response",
                        retryable=True,
                        provider=self.name,
                    )
                temporary_output.write_bytes(response.content)
                temporary_output.replace(output)
                return
        finally:
            temporary_output.unlink(missing_ok=True)

        raise VideoProviderError(
            "download",
            "video download did not complete",
            retryable=True,
            provider=self.name,
        )

    @staticmethod
    def _get_output_url(task: dict[str, Any]) -> str:
        outputs = task.get("output")
        if (
            not isinstance(outputs, list)
            or not outputs
            or not isinstance(outputs[0], str)
            or not outputs[0].strip()
        ):
            raise VideoProviderError(
                "generation",
                "successful task did not contain a video output URL",
                retryable=False,
                provider="runway",
            )
        return outputs[0]

    def _response_json(
        self,
        response: httpx.Response,
        operation: str,
    ) -> dict[str, Any]:
        try:
            body = response.json()
        except ValueError as exc:
            raise VideoProviderError(
                operation,
                "API returned malformed JSON",
                retryable=response.status_code >= 500,
                provider=self.name,
            ) from exc
        if not isinstance(body, dict):
            raise VideoProviderError(
                operation,
                "API returned a JSON value with an unexpected shape",
                retryable=False,
                provider=self.name,
            )
        return body

    def _raise_for_api_error(
        self,
        response: httpx.Response,
        body: dict[str, Any],
        operation: str,
    ) -> None:
        if response.is_success:
            return
        error = body.get("error")
        reason = (
            error.get("message")
            if isinstance(error, dict)
            else body.get("message")
        ) or f"HTTP {response.status_code}"
        raise VideoProviderError(
            operation,
            f"HTTP {response.status_code}: {reason}",
            retryable=response.status_code == 429 or response.status_code >= 500,
            provider=self.name,
        )
