import base64
import json
from pathlib import Path
from threading import Event

import httpx
import pytest

from app.video.providers import (
    VideoGenerationCancelled,
    VideoGenerationTimeout,
    VideoProviderError,
    VeoVideoGenerationProvider,
)


def _video_operation(
    operation_name: str,
    video_id: str,
    *,
    done: bool = True,
) -> dict:
    return {
        "name": operation_name,
        "done": done,
        "response": {
            "generateVideoResponse": {
                "generatedSamples": [
                    {
                        "video": {
                            "uri": (
                                "https://generativelanguage.googleapis.com/"
                                f"download/{video_id}"
                            ),
                            "mimeType": "video/mp4",
                        }
                    }
                ]
            }
        },
    }


def _provider(
    client: httpx.Client,
    *,
    poll_interval_seconds: float = 0,
    timeout_seconds: float = 5,
) -> VeoVideoGenerationProvider:
    return VeoVideoGenerationProvider(
        api_key="test-api-key",
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        client=client,
    )


def _unexpected_api_request(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"Unexpected API request: {request.url}")


def test_provider_requires_api_key():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        VeoVideoGenerationProvider(api_key=None)


@pytest.mark.parametrize(
    ("poll_interval_seconds", "timeout_seconds"),
    [(-1, 10), (1, 0)],
)
def test_provider_rejects_invalid_polling_configuration(
    poll_interval_seconds,
    timeout_seconds,
):
    with pytest.raises(ValueError):
        VeoVideoGenerationProvider(
            api_key="test-api-key",
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
        )


def test_request_poll_and_download_lifecycle(tmp_path):
    image_path = tmp_path / "first.png"
    image_path.write_bytes(b"png bytes")
    output_path = tmp_path / "scene.mp4"
    requests = []
    poll_count = 0

    def handle_request(request):
        nonlocal poll_count
        requests.append(request)
        if request.method == "POST":
            assert request.headers["x-goog-api-key"] == "test-api-key"
            return httpx.Response(
                200,
                json={"name": "operations/generation-1"},
            )
        if request.url.path.endswith("/operations/generation-1"):
            poll_count += 1
            if poll_count == 1:
                return httpx.Response(
                    200,
                    json={"name": "operations/generation-1", "done": False},
                )
            return httpx.Response(
                200,
                json=_video_operation(
                    "operations/generation-1",
                    "clip-1",
                ),
            )
        if request.url.path.endswith("/download/clip-1"):
            return httpx.Response(200, content=b"mock mp4 data")
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        result = _provider(client).generate_video(
            prompt="A character walks through a sunlit room.",
            output_path=output_path,
            duration_seconds=5,
            first_frame=image_path,
            aspect_ratio="16:9",
        )

    request_body = json.loads(requests[0].content)
    instance = request_body["instances"][0]
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    assert instance["prompt"] == "A character walks through a sunlit room."
    assert instance["image"] == {
        "bytesBase64Encoded": encoded_image,
        "mimeType": "image/png",
    }
    assert request_body["parameters"] == {
        "aspectRatio": "16:9",
        "durationSeconds": 6,
        "resolution": "720p",
    }
    assert isinstance(request_body["parameters"]["durationSeconds"], int)
    assert b'"durationSeconds":6' in requests[0].content
    assert poll_count == 2
    assert output_path.read_bytes() == b"mock mp4 data"
    assert result.provider == "veo"
    assert result.generation_id == "operations/generation-1"
    assert result.duration_seconds == 6


def test_reference_and_last_frame_inputs_use_eight_seconds(tmp_path):
    first_frame = tmp_path / "first.png"
    last_frame = tmp_path / "last.png"
    reference = tmp_path / "character.jpg"
    for image in (first_frame, last_frame, reference):
        image.write_bytes(b"image")
    request_body = {}

    def handle_request(request):
        nonlocal request_body
        if request.method == "POST":
            request_body = json.loads(request.content)
            return httpx.Response(
                200,
                json={"name": "operations/generation-2"},
            )
        if request.url.path.endswith("/operations/generation-2"):
            return httpx.Response(
                200,
                json=_video_operation(
                    "operations/generation-2",
                    "clip-2",
                ),
            )
        if request.url.path.endswith("/download/clip-2"):
            return httpx.Response(200, content=b"mock mp4 data")
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    output_path = tmp_path / "scene.mp4"
    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        result = _provider(client).generate_video(
            prompt="The same character enters.",
            output_path=output_path,
            duration_seconds=5,
            first_frame=first_frame,
            last_frame=last_frame,
            reference_images=(reference,),
            aspect_ratio="9:16",
        )

    instance = request_body["instances"][0]
    assert instance["lastFrame"]["mimeType"] == "image/png"
    assert "bytesBase64Encoded" in instance["lastFrame"]
    assert request_body["parameters"]["referenceImages"][0][
        "referenceType"
    ] == "asset"
    assert (
        request_body["parameters"]["referenceImages"][0]["image"][
            "mimeType"
        ]
        == "image/jpeg"
    )
    assert "bytesBase64Encoded" in request_body["parameters"][
        "referenceImages"
    ][0]["image"]
    assert request_body["parameters"]["durationSeconds"] == 8
    assert isinstance(request_body["parameters"]["durationSeconds"], int)
    assert request_body["parameters"]["aspectRatio"] == "9:16"
    assert result.duration_seconds == 8


def test_ten_second_scene_uses_veo_extension(tmp_path):
    post_count = 0
    operation_responses = {
        "/operations/base": _video_operation(
            "operations/base",
            "base-video",
        ),
        "/operations/extension": _video_operation(
            "operations/extension",
            "extended-video",
        ),
    }
    requests = []

    def handle_request(request):
        nonlocal post_count
        requests.append(request)
        if request.method == "POST":
            post_count += 1
            if post_count == 1:
                return httpx.Response(
                    200,
                    json={"name": "operations/base"},
                )
            extension_payload = json.loads(request.content)
            assert extension_payload["instances"][0]["video"]["uri"].endswith(
                "/download/base-video"
            )
            extension_duration = extension_payload["parameters"][
                "durationSeconds"
            ]
            assert extension_duration == 8
            assert isinstance(extension_duration, int)
            assert b'"durationSeconds":8' in request.content
            return httpx.Response(
                200,
                json={"name": "operations/extension"},
            )
        for suffix, operation in operation_responses.items():
            if request.url.path.endswith(suffix):
                return httpx.Response(200, json=operation)
        if request.url.path.endswith("/download/extended-video"):
            return httpx.Response(200, content=b"extended mp4 data")
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    output_path = tmp_path / "extended.mp4"
    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        result = _provider(client).generate_video(
            prompt="Continue walking toward the door.",
            output_path=output_path,
            duration_seconds=10,
        )

    extension_request = json.loads(
        next(
            request.content
            for request in requests
            if request.method == "POST"
            and b'"video":' in request.content
        )
    )
    assert "Continue seamlessly" in extension_request["instances"][0]["prompt"]
    assert result.duration_seconds == 15
    assert result.metadata["operation_ids"] == (
        "operations/base",
        "operations/extension",
    )
    assert output_path.read_bytes() == b"extended mp4 data"


@pytest.mark.parametrize(
    ("requested_duration", "expected_generation_duration"),
    [
        (4, 4),
        (5, 6),
        (6, 6),
        (7, 8),
        (8, 8),
        ("5", 6),
        (5.0, 6),
    ],
)
def test_requested_duration_maps_to_supported_generation_duration(
    requested_duration,
    expected_generation_duration,
):
    normalized = VeoVideoGenerationProvider._normalize_requested_duration(
        requested_duration
    )
    generated = VeoVideoGenerationProvider._initial_duration(
        normalized,
        has_last_frame=False,
        has_reference_images=False,
    )

    assert generated == expected_generation_duration
    assert generated in {4, 6, 8}


@pytest.mark.parametrize(
    "requested_duration",
    [None, 0, -1, "not-a-duration", "5.5", 5.5, True, float("inf")],
)
def test_invalid_or_missing_requested_duration_is_rejected(
    requested_duration,
    tmp_path,
):
    with httpx.Client(
        transport=httpx.MockTransport(_unexpected_api_request)
    ) as client:
        with pytest.raises(ValueError, match="positive whole number"):
            _provider(client).generate_video(
                prompt="A test clip.",
                output_path=tmp_path / "unused.mp4",
                duration_seconds=requested_duration,
            )


def test_missing_requested_duration_is_rejected(tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(_unexpected_api_request)
    ) as client:
        with pytest.raises(ValueError, match="positive whole number"):
            _provider(client).generate_video(
                prompt="A test clip.",
                output_path=tmp_path / "unused.mp4",
            )


def test_scene_target_beyond_veo_extension_limit_is_rejected(tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(_unexpected_api_request)
    ) as client:
        with pytest.raises(ValueError, match="148 seconds"):
            _provider(client).generate_video(
                prompt="A test clip.",
                output_path=tmp_path / "unused.mp4",
                duration_seconds=149,
            )


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(401, False), (403, False), (429, True), (503, True)],
)
def test_http_errors_are_classified(status_code, retryable):
    def handle_request(_request):
        del _request
        return httpx.Response(
            status_code,
            json={"error": {"message": "request rejected"}},
        )

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        with pytest.raises(VideoProviderError) as error:
            _provider(client).generate_video(
                prompt="A test clip.",
                output_path=Path("unused.mp4"),
                duration_seconds=4,
            )

    assert error.value.retryable is retryable
    assert "request rejected" in error.value.reason


def test_failed_generation_operation_surfaces_api_error():
    def handle_request(request):
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"name": "operations/failed"},
            )
        return httpx.Response(
            200,
            json={
                "name": "operations/failed",
                "done": True,
                "error": {"code": 400, "message": "safety policy rejection"},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        with pytest.raises(VideoProviderError, match="safety policy rejection"):
            _provider(client).generate_video(
                prompt="A test clip.",
                output_path=Path("unused.mp4"),
                duration_seconds=4,
            )


def test_timeout_stops_polling():
    def handle_request(request):
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"name": "operations/pending"},
            )
        return httpx.Response(
            200,
            json={"name": "operations/pending", "done": False},
        )

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        with pytest.raises(VideoGenerationTimeout):
            _provider(
                client,
                poll_interval_seconds=0.001,
                timeout_seconds=0.005,
            ).generate_video(
                prompt="A test clip.",
                output_path=Path("unused.mp4"),
                duration_seconds=4,
            )


def test_cancel_event_stops_polling():
    cancel_event = Event()
    cancel_event.set()

    def handle_request(_request):
        del _request
        return httpx.Response(
            200,
            json={"name": "operations/pending"},
        )

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        with pytest.raises(VideoGenerationCancelled):
            _provider(client).generate_video(
                prompt="A test clip.",
                output_path=Path("unused.mp4"),
                duration_seconds=4,
                cancel_event=cancel_event,
            )


def test_invalid_operation_response_is_rejected():
    def handle_request(_request):
        del _request
        return httpx.Response(200, json={"unexpected": "shape"})

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        with pytest.raises(VideoProviderError, match="operation name"):
            _provider(client).generate_video(
                prompt="A test clip.",
                output_path=Path("unused.mp4"),
                duration_seconds=4,
            )


def test_download_failure_is_reported():
    def handle_request(request):
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"name": "operations/download-failure"},
            )
        if request.url.path.endswith("/operations/download-failure"):
            return httpx.Response(
                200,
                json=_video_operation(
                    "operations/download-failure",
                    "download-failure",
                ),
            )
        return httpx.Response(
            503,
            json={"error": {"message": "storage temporarily unavailable"}},
        )

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        with pytest.raises(VideoProviderError, match="storage temporarily"):
            _provider(client).generate_video(
                prompt="A test clip.",
                output_path=Path("unused.mp4"),
                duration_seconds=4,
            )


def test_download_retries_transient_server_error(tmp_path):
    download_attempts = 0

    def handle_request(request):
        nonlocal download_attempts
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"name": "operations/retry-download"},
            )
        if request.url.path.endswith("/operations/retry-download"):
            return httpx.Response(
                200,
                json=_video_operation(
                    "operations/retry-download",
                    "retry-download",
                ),
            )
        download_attempts += 1
        if download_attempts == 1:
            return httpx.Response(
                503,
                json={"error": {"message": "temporary storage error"}},
            )
        return httpx.Response(200, content=b"retried mp4 data")

    output_path = tmp_path / "retried.mp4"
    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        _provider(client).generate_video(
            prompt="A test clip.",
            output_path=output_path,
            duration_seconds=4,
        )

    assert download_attempts == 2
    assert output_path.read_bytes() == b"retried mp4 data"


def test_download_rejects_untrusted_urls():
    with pytest.raises(VideoProviderError, match="untrusted"):
        VeoVideoGenerationProvider._validate_download_url(
            "https://example.invalid/video.mp4"
        )
