import base64
import logging
import mimetypes
import time
from contextlib import nullcontext
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import Event
from typing import Any, Sequence
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

_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
_VIDEO_DURATION_CHOICES = (4, 6, 8)
_EXTENSION_GENERATION_DURATION_SECONDS = 8
_MAX_REFERENCE_IMAGES = 3
_MAX_EXTENSION_COUNT = 20
_EXTENSION_SECONDS = 7
_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


class VeoVideoGenerationProvider(VideoGenerationProvider):
    name = "veo"
    capabilities = VideoProviderCapabilities(
        text_to_video=True,
        image_to_video=True,
        reference_images=True,
        first_frame=True,
        last_frame=True,
        video_extension=True,
        supported_aspect_ratios=("16:9", "9:16"),
    )

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str = "veo-3.1-generate-preview",
        poll_interval_seconds: float = 5,
        timeout_seconds: float = 900,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is required when VIDEO_PROVIDER=veo."
            )
        if not model or not model.strip():
            raise ValueError("VEO_MODEL must not be empty.")
        if poll_interval_seconds < 0:
            raise ValueError("VEO_POLL_INTERVAL_SECONDS must not be negative.")
        if timeout_seconds <= 0:
            raise ValueError("VEO_TIMEOUT_SECONDS must be greater than zero.")

        self._api_key = api_key.strip()
        self._model = model.strip()
        self._poll_interval_seconds = poll_interval_seconds
        self._timeout_seconds = timeout_seconds
        self._client = client

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
        requested_duration = self._normalize_requested_duration(
            duration_seconds
        )
        if aspect_ratio not in self.capabilities.supported_aspect_ratios:
            raise ValueError(
                f"Veo does not support aspect ratio {aspect_ratio!r}; "
                "use '16:9' or '9:16'."
            )
        if last_frame is not None and first_frame is None:
            raise ValueError(
                "Veo last-frame interpolation requires a first frame."
            )
        if len(reference_images) > _MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"Veo supports at most {_MAX_REFERENCE_IMAGES} reference images."
            )

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        generation_duration = self._initial_duration(
            requested_duration,
            has_last_frame=last_frame is not None,
            has_reference_images=bool(reference_images),
        )
        extension_count = self._extension_count(
            requested_duration,
            generation_duration,
        )
        deadline = time.monotonic() + self._timeout_seconds

        logger.info(
            "Veo video generation requested",
            extra={
                "provider": self.name,
                "model": self._model,
                "duration_seconds": requested_duration,
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
            request = self._build_generation_request(
                prompt=prompt,
                duration_seconds=generation_duration,
                first_frame=first_frame,
                last_frame=last_frame,
                reference_images=reference_images,
                aspect_ratio=aspect_ratio,
            )
            operation = self._submit(client, request)
            operation_id = operation["name"]
            operation_ids = [operation_id]
            operation = self._poll(client, operation, deadline, cancel_event)
            video = self._get_generated_video(operation)
            generated_duration = generation_duration

            for extension_index in range(extension_count):
                extension_request = {
                    "instances": [
                        {
                            "prompt": (
                                "Continue seamlessly from the ending of the "
                                f"supplied video. {prompt}"
                            ),
                            "video": video,
                        }
                    ],
                    "parameters": {
                        "aspectRatio": aspect_ratio,
                        "durationSeconds": (
                            _EXTENSION_GENERATION_DURATION_SECONDS
                        ),
                        "resolution": "720p",
                    },
                }
                logger.info(
                    "Veo video extension requested",
                    extra={
                        "provider": self.name,
                        "model": self._model,
                        "extension_index": extension_index + 1,
                    },
                )
                extension_operation = self._submit(client, extension_request)
                operation_ids.append(extension_operation["name"])
                extension_operation = self._poll(
                    client,
                    extension_operation,
                    deadline,
                    cancel_event,
                )
                video = self._get_generated_video(extension_operation)
                generated_duration += _EXTENSION_SECONDS

            self._download_video(client, video, output)

        logger.info(
            "Veo video downloaded",
            extra={
                "provider": self.name,
                "generation_id": operation_id,
                "generated_duration_seconds": generated_duration,
                "output_path": str(output),
            },
        )
        return VideoGenerationResult(
            video_path=output,
            provider=self.name,
            generation_id=operation_id,
            duration_seconds=generated_duration,
            metadata={
                "model": self._model,
                "operation_ids": tuple(operation_ids),
                "requested_duration_seconds": requested_duration,
                "aspect_ratio": aspect_ratio,
            },
        )

    @staticmethod
    def _normalize_requested_duration(
        duration_seconds: int | float | str | None,
    ) -> int:
        if duration_seconds is None or isinstance(duration_seconds, bool):
            raise ValueError(
                "Requested scene duration must be a positive whole number "
                "of seconds."
            )
        if not isinstance(duration_seconds, (int, float, str)):
            raise ValueError(
                "Requested scene duration must be a positive whole number "
                "of seconds."
            )

        try:
            parsed_duration = Decimal(str(duration_seconds).strip())
        except InvalidOperation as exc:
            raise ValueError(
                "Requested scene duration must be a positive whole number "
                "of seconds."
            ) from exc

        if (
            not parsed_duration.is_finite()
            or parsed_duration <= 0
            or parsed_duration != parsed_duration.to_integral_value()
        ):
            raise ValueError(
                "Requested scene duration must be a positive whole number "
                "of seconds."
            )
        return int(parsed_duration)

    @staticmethod
    def _initial_duration(
        requested_seconds: int,
        *,
        has_last_frame: bool,
        has_reference_images: bool,
    ) -> int:
        if requested_seconds > 8 or has_last_frame or has_reference_images:
            return 8
        return next(
            duration
            for duration in _VIDEO_DURATION_CHOICES
            if duration >= requested_seconds
        )

    @staticmethod
    def _extension_count(requested_seconds: int, initial_seconds: int) -> int:
        if requested_seconds <= initial_seconds:
            return 0
        count = (
            requested_seconds - initial_seconds + _EXTENSION_SECONDS - 1
        ) // _EXTENSION_SECONDS
        if count > _MAX_EXTENSION_COUNT:
            maximum = 8 + _MAX_EXTENSION_COUNT * _EXTENSION_SECONDS
            raise ValueError(
                f"Veo supports generated videos up to {maximum} seconds "
                "through its documented extension workflow."
            )
        return count

    def _build_generation_request(
        self,
        *,
        prompt: str,
        duration_seconds: int,
        first_frame: Path | None,
        last_frame: Path | None,
        reference_images: Sequence[Path],
        aspect_ratio: str,
    ) -> dict[str, Any]:
        instance: dict[str, Any] = {"prompt": prompt}
        if first_frame is not None:
            instance["image"] = self._image_input(first_frame)
        if last_frame is not None:
            instance["lastFrame"] = self._image_input(last_frame)

        parameters: dict[str, Any] = {
            "aspectRatio": aspect_ratio,
            "durationSeconds": duration_seconds,
            "resolution": "720p",
        }
        if reference_images:
            parameters["referenceImages"] = [
                {
                    "image": self._image_input(image_path),
                    "referenceType": "asset",
                }
                for image_path in reference_images
            ]

        return {"instances": [instance], "parameters": parameters}

    @staticmethod
    def _image_input(image_path: Path) -> dict[str, Any]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Veo image input not found: {path}")

        mime_type, _ = mimetypes.guess_type(path.name)
        if mime_type not in _ALLOWED_IMAGE_TYPES:
            raise ValueError(
                f"Unsupported Veo image format: {path.suffix or 'unknown'}"
            )

        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "bytesBase64Encoded": encoded,
            "mimeType": mime_type,
        }

    def _submit(
        self,
        client: httpx.Client,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        logger.info(
            "Submitting Veo generation operation",
            extra={"provider": self.name, "model": self._model},
        )
        try:
            response = client.post(
                f"{_API_BASE}/models/{self._model}:predictLongRunning",
                headers=self._headers(),
                json=request,
            )
        except httpx.RequestError as exc:
            raise VideoProviderError(
                "submit",
                f"network request failed: {type(exc).__name__}",
                retryable=True,
                provider=self.name,
            ) from exc

        payload = self._response_json(response, "submit")
        self._raise_for_api_error(response, payload, "submit")
        operation_name = payload.get("name")
        if not isinstance(operation_name, str) or not operation_name.strip():
            raise VideoProviderError(
                "submit",
                "API response did not contain an operation name",
                retryable=False,
                provider=self.name,
            )
        if operation_name.startswith("/") or "://" in operation_name:
            raise VideoProviderError(
                "submit",
                "API response contained an invalid operation name",
                retryable=False,
                provider=self.name,
            )

        logger.info(
            "Veo generation operation created",
            extra={"provider": self.name, "generation_id": operation_name},
        )
        return payload

    def _poll(
        self,
        client: httpx.Client,
        operation: dict[str, Any],
        deadline: float,
        cancel_event: Event | None,
    ) -> dict[str, Any]:
        operation_name = operation["name"]
        operation_url = f"{_API_BASE}/{quote(operation_name, safe='/')}"

        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise VideoGenerationCancelled()

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VideoGenerationTimeout(self._timeout_seconds)
            wait_seconds = min(self._poll_interval_seconds, remaining)
            if cancel_event is not None:
                if cancel_event.wait(wait_seconds):
                    raise VideoGenerationCancelled()
            elif wait_seconds:
                time.sleep(wait_seconds)

            if time.monotonic() >= deadline:
                raise VideoGenerationTimeout(self._timeout_seconds)

            logger.info(
                "Polling Veo generation",
                extra={"provider": self.name, "generation_id": operation_name},
            )
            payload = self._get_json(
                client,
                operation_url,
                operation="poll",
            )
            done = payload.get("done")
            if not isinstance(done, bool):
                raise VideoProviderError(
                    "poll",
                    "API response did not contain a boolean done status",
                    retryable=False,
                    provider=self.name,
                )
            if done:
                self._raise_operation_error(payload)
                return payload

    def _get_json(
        self,
        client: httpx.Client,
        url: str,
        *,
        operation: str,
    ) -> dict[str, Any]:
        attempts = 3
        for attempt in range(attempts):
            try:
                response = client.get(url, headers=self._headers())
            except httpx.RequestError as exc:
                if attempt + 1 == attempts:
                    raise VideoProviderError(
                        operation,
                        f"network request failed: {type(exc).__name__}",
                        retryable=True,
                        provider=self.name,
                    ) from exc
                time.sleep(min(2**attempt, self._poll_interval_seconds or 1))
                continue

            payload = self._response_json(response, operation)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 < attempts:
                    time.sleep(
                        min(2**attempt, self._poll_interval_seconds or 1)
                    )
                    continue
            self._raise_for_api_error(response, payload, operation)
            return payload

        raise AssertionError("Unreachable retry loop")

    def _response_json(
        self,
        response: httpx.Response,
        operation: str,
    ) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise VideoProviderError(
                operation,
                "API returned malformed JSON",
                retryable=response.status_code >= 500,
                provider=self.name,
            ) from exc

        if not isinstance(payload, dict):
            raise VideoProviderError(
                operation,
                "API returned a JSON value with an unexpected shape",
                retryable=False,
                provider=self.name,
            )
        return payload

    def _raise_for_api_error(
        self,
        response: httpx.Response,
        payload: dict[str, Any],
        operation: str,
    ) -> None:
        if response.is_success:
            return

        error = payload.get("error")
        reason = (
            error.get("message")
            if isinstance(error, dict)
            else f"HTTP {response.status_code}"
        )
        retryable = response.status_code == 429 or response.status_code >= 500
        raise VideoProviderError(
            operation,
            f"HTTP {response.status_code}: {reason}",
            retryable=retryable,
            provider=self.name,
        )

    def _raise_operation_error(self, operation: dict[str, Any]) -> None:
        error = operation.get("error")
        if error is None:
            return
        if not isinstance(error, dict):
            raise VideoProviderError(
                "generation",
                "API reported a failure with an invalid error response",
                retryable=False,
                provider=self.name,
            )

        code = error.get("code")
        reason = error.get("message") or "generation operation failed"
        retryable = code == 429 or code in {500, 502, 503, 504}
        logger.error(
            "Veo generation failed",
            extra={
                "provider": self.name,
                "generation_id": operation.get("name"),
                "reason": reason,
                "retryable": retryable,
            },
        )
        raise VideoProviderError(
            "generation",
            str(reason),
            retryable=retryable,
            provider=self.name,
        )

    def _get_generated_video(self, operation: dict[str, Any]) -> dict[str, Any]:
        response = operation.get("response")
        video_response = (
            response.get("generateVideoResponse")
            if isinstance(response, dict)
            else None
        )
        samples = (
            video_response.get("generatedSamples")
            if isinstance(video_response, dict)
            else None
        )
        if not isinstance(samples, list) or not samples:
            filtered_count = (
                video_response.get("raiMediaFilteredCount", 0)
                if isinstance(video_response, dict)
                else 0
            )
            reason = (
                "generation was rejected by Veo's safety filters"
                if filtered_count
                else "completed operation contained no generated video"
            )
            raise VideoProviderError(
                "generation",
                reason,
                retryable=False,
                provider=self.name,
            )

        video_sample = samples[0]
        video = (
            video_sample.get("video")
            if isinstance(video_sample, dict)
            else None
        )
        uri = video.get("uri") if isinstance(video, dict) else None
        mime_type = video.get("mimeType") if isinstance(video, dict) else None
        if not isinstance(uri, str) or not uri:
            raise VideoProviderError(
                "generation",
                "API response did not contain a video URI",
                retryable=False,
                provider=self.name,
            )
        if mime_type is not None and (
            not isinstance(mime_type, str) or not mime_type.startswith("video/")
        ):
            raise VideoProviderError(
                "generation",
                "API response did not contain a valid video MIME type",
                retryable=False,
                provider=self.name,
            )
        return video

    def _download_video(
        self,
        client: httpx.Client,
        video: dict[str, Any],
        output: Path,
    ) -> None:
        current_url = video["uri"]
        for redirect_count in range(4):
            self._validate_download_url(current_url)
            response = self._get_download_response(client, current_url)

            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise VideoProviderError(
                        "download",
                        "video download redirected without a location",
                        retryable=False,
                        provider=self.name,
                    )
                if redirect_count == 3:
                    raise VideoProviderError(
                        "download",
                        "video download exceeded the redirect limit",
                        retryable=False,
                        provider=self.name,
                    )
                current_url = urljoin(current_url, location)
                continue

            if not response.is_success:
                payload = self._response_json(response, "download")
                self._raise_for_api_error(response, payload, "download")
            if not response.content:
                raise VideoProviderError(
                    "download",
                    "video download returned an empty response",
                    retryable=True,
                    provider=self.name,
                )
            output.write_bytes(response.content)
            return

    def _get_download_response(
        self,
        client: httpx.Client,
        url: str,
    ) -> httpx.Response:
        host = urlsplit(url).hostname
        headers = (
            self._headers()
            if host == "generativelanguage.googleapis.com"
            else {}
        )
        attempts = 3
        for attempt in range(attempts):
            try:
                response = client.get(
                    url,
                    headers=headers,
                    follow_redirects=False,
                )
            except httpx.RequestError as exc:
                if attempt + 1 == attempts:
                    raise VideoProviderError(
                        "download",
                        f"network request failed: {type(exc).__name__}",
                        retryable=True,
                        provider=self.name,
                    ) from exc
                time.sleep(min(2**attempt, self._poll_interval_seconds or 1))
                continue

            if (
                response.status_code == 429 or response.status_code >= 500
            ) and attempt + 1 < attempts:
                time.sleep(min(2**attempt, self._poll_interval_seconds or 1))
                continue
            return response

        raise AssertionError("Unreachable download retry loop")

    @staticmethod
    def _validate_download_url(url: str) -> None:
        parsed = urlsplit(url)
        host = parsed.hostname or ""
        trusted_host = (
            host.endswith(".googleapis.com")
            or host == "googleapis.com"
            or host.endswith(".googleusercontent.com")
            or host == "googleusercontent.com"
        )
        if parsed.scheme != "https" or not trusted_host:
            raise VideoProviderError(
                "download",
                "API returned an untrusted video download URL",
                retryable=False,
                provider="veo",
            )

    def _headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }
