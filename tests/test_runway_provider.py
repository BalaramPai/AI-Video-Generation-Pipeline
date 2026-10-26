import base64
import json
from pathlib import Path
from threading import Event
import time

import httpx
import pytest

from app.video.providers import (
    RunwayVideoGenerationProvider,
    VideoGenerationCancelled,
    VideoGenerationTimeout,
    VideoProviderError,
)
from app.video.providers.runway import (
    RUNWAY_API_VERSION,
    RUNWAY_MODEL_PROMPT_MAX_CHARACTERS,
    RUNWAY_PROMPT_MAX_CHARACTERS,
    _fit_runway_prompt,
)


def _provider(
    client: httpx.Client,
    *,
    model: str = "gen4.5",
    poll_interval_seconds: float = 0,
    timeout_seconds: float = 5,
) -> RunwayVideoGenerationProvider:
    return RunwayVideoGenerationProvider(
        api_secret="test-runway-secret",
        model=model,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        client=client,
    )


def test_gen45_request_task_poll_and_download_lifecycle(tmp_path):
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"png image bytes")
    output_path = tmp_path / "scene.mp4"
    requests = []
    poll_count = 0

    def handle_request(request):
        nonlocal poll_count
        requests.append(request)
        if request.method == "POST":
            assert request.headers["Authorization"] == (
                "Bearer test-runway-secret"
            )
            assert request.headers["X-Runway-Version"] == RUNWAY_API_VERSION
            return httpx.Response(200, json={"id": "task-123"})
        if request.url.path == "/v1/tasks/task-123":
            poll_count += 1
            if poll_count == 1:
                return httpx.Response(
                    200,
                    json={"id": "task-123", "status": "RUNNING"},
                )
            return httpx.Response(
                200,
                json={
                    "id": "task-123",
                    "status": "SUCCEEDED",
                    "output": ["https://cdn.runwayml.com/clip.mp4"],
                },
            )
        if request.url.host == "cdn.runwayml.com":
            assert "Authorization" not in request.headers
            return httpx.Response(200, content=b"mock mp4 bytes")
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
    expected_image = (
        "data:image/png;base64,"
        + base64.b64encode(image_path.read_bytes()).decode("ascii")
    )
    assert request_body == {
        "model": "gen4.5",
        "promptImage": expected_image,
        "promptText": "A character walks through a sunlit room.",
        "duration": 5,
        "ratio": "1280:720",
    }
    assert poll_count == 2
    assert output_path.read_bytes() == b"mock mp4 bytes"
    assert result.provider == "runway"
    assert result.generation_id == "task-123"
    assert result.duration_seconds == 5


def test_grok_imagine_uses_model_specific_image_payload(tmp_path):
    image_path = tmp_path / "portrait.png"
    image_path.write_bytes(b"png image bytes")
    output_path = tmp_path / "grok.mp4"
    payload = {}

    def handle_request(request):
        if request.method == "POST":
            payload.update(json.loads(request.content))
            return httpx.Response(200, json={"id": "grok-task"})
        if request.url.path == "/v1/tasks/grok-task":
            return httpx.Response(
                200,
                json={
                    "id": "grok-task",
                    "status": "SUCCEEDED",
                    "output": ["https://cdn.runwayml.com/grok.mp4"],
                },
            )
        return httpx.Response(200, content=b"grok video")

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        result = _provider(
            client,
            model="grok_imagine_1_5",
        ).generate_video(
            prompt="Continue the previous scene.",
            output_path=output_path,
            duration_seconds=12,
            first_frame=image_path,
            aspect_ratio="9:16",
        )

    assert payload["model"] == "grok_imagine_1_5"
    assert payload["duration"] == 12
    assert payload["resolution"] == "720p"
    assert "ratio" not in payload
    assert payload["promptText"] == "Continue the previous scene."
    assert result.metadata["model"] == "grok_imagine_1_5"


@pytest.mark.parametrize(
    ("model", "duration", "message"),
    [
        ("gen4.5", 11, "2 to 10 seconds"),
        ("grok_imagine_1_5", 16, "1 to 15 seconds"),
    ],
)
def test_model_duration_constraints_reject_before_api_request(
    model,
    duration,
    message,
    tmp_path,
):
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"image")
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: pytest.fail(f"Unexpected request: {request.url}")
        )
    ) as client:
        with pytest.raises(ValueError, match=message):
            _provider(client, model=model).generate_video(
                prompt="A test clip.",
                output_path=tmp_path / "scene.mp4",
                duration_seconds=duration,
                first_frame=image_path,
            )


def test_gen45_minimum_generation_is_trimmed_to_one_second_target(tmp_path):
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"image")
    output_path = tmp_path / "scene.mp4"
    payload = {}

    def handle_request(request):
        if request.method == "POST":
            payload.update(json.loads(request.content))
            return httpx.Response(200, json={"id": "short-task"})
        if request.url.path == "/v1/tasks/short-task":
            return httpx.Response(
                200,
                json={
                    "id": "short-task",
                    "status": "SUCCEEDED",
                    "output": ["https://cdn.runwayml.com/short.mp4"],
                },
            )
        return httpx.Response(200, content=b"two second clip")

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        result = _provider(client).generate_video(
            prompt="A one-second shot.",
            output_path=output_path,
            duration_seconds=1,
            first_frame=image_path,
        )

    assert payload["duration"] == 2
    assert result.duration_seconds == 2
    assert result.metadata["requested_duration_seconds"] == 1


def test_runway_rejects_missing_key_and_unknown_model():
    with pytest.raises(ValueError, match="RUNWAYML_API_SECRET"):
        RunwayVideoGenerationProvider(api_secret=None)
    with pytest.raises(ValueError, match="grok_imagine_1_5"):
        RunwayVideoGenerationProvider(
            api_secret="test",
            model="Grok Imagine 1.5",
        )


def test_runway_prompt_limit_counts_utf16_code_units():
    fitted = _fit_runway_prompt(" 🎬 scene continuity ")
    assert fitted == "🎬 scene continuity"
    assert len(fitted.encode("utf-16-le")) // 2 < RUNWAY_PROMPT_MAX_CHARACTERS

    overlong = "🎬" * (RUNWAY_PROMPT_MAX_CHARACTERS // 2 + 1)
    with pytest.raises(ValueError, match="UTF-16 characters"):
        _fit_runway_prompt(overlong)

    grok_prompt = "x" * 2000
    assert _fit_runway_prompt(
        grok_prompt,
        "grok_imagine_1_5",
    ) == grok_prompt
    assert RUNWAY_MODEL_PROMPT_MAX_CHARACTERS["grok_imagine_1_5"] == 2500


def test_runway_rejects_overlong_prompt_before_submitting_task(tmp_path):
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"mock image bytes")
    transport = httpx.MockTransport(
        lambda request: pytest.fail(
            f"Overlong prompt must not be submitted: {request.url}"
        )
    )

    with httpx.Client(transport=transport) as client:
        with pytest.raises(ValueError, match="will not be truncated"):
            _provider(client).generate_video(
                prompt="x" * (RUNWAY_PROMPT_MAX_CHARACTERS + 1),
                output_path=tmp_path / "scene.mp4",
                duration_seconds=5,
                first_frame=image_path,
            )


def test_runway_rejects_separate_reference_images_without_api_request(tmp_path):
    first_frame = tmp_path / "first.png"
    reference = tmp_path / "reference.png"
    first_frame.write_bytes(b"first")
    reference.write_bytes(b"reference")

    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: pytest.fail(f"Unexpected request: {request.url}")
        )
    ) as client:
        with pytest.raises(ValueError, match="does not support separate"):
            _provider(client).generate_video(
                prompt="Continue the same scene.",
                output_path=tmp_path / "scene.mp4",
                duration_seconds=5,
                first_frame=first_frame,
                reference_images=(reference,),
            )


def test_runway_failure_status_is_reported(tmp_path):
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"image")

    def handle_request(request):
        if request.method == "POST":
            return httpx.Response(200, json={"id": "failed-task"})
        return httpx.Response(
            200,
            json={
                "id": "failed-task",
                "status": "FAILED",
                "failure": "safety policy rejection",
                "failureCode": "CONTENT_MODERATION",
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        with pytest.raises(VideoProviderError, match="safety policy rejection"):
            _provider(client).generate_video(
                prompt="A test clip.",
                output_path=Path("unused.mp4"),
                duration_seconds=5,
                first_frame=image_path,
            )


def test_runway_polling_can_be_cancelled(tmp_path):
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"image")
    cancel_event = Event()
    cancel_event.set()

    def handle_request(request):
        if request.method == "POST":
            return httpx.Response(200, json={"id": "pending-task"})
        pytest.fail(f"Unexpected poll after cancellation: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        with pytest.raises(VideoGenerationCancelled):
            _provider(client).generate_video(
                prompt="A test clip.",
                output_path=Path("unused.mp4"),
                duration_seconds=5,
                first_frame=image_path,
                cancel_event=cancel_event,
            )


def test_interrupted_runway_task_resumes_without_resubmitting(tmp_path):
    image_path = tmp_path / "scene.png"
    original_image = b"original image"
    image_path.write_bytes(original_image)
    output_path = tmp_path / "scene.mp4"
    requests = []
    interrupt_poll = True

    def handle_request(request):
        nonlocal interrupt_poll
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"id": "resumable-task"})
        if request.url.path == "/v1/tasks/resumable-task":
            if interrupt_poll:
                interrupt_poll = False
                raise KeyboardInterrupt
            return httpx.Response(
                200,
                json={
                    "id": "resumable-task",
                    "status": "SUCCEEDED",
                    "output": ["https://cdn.runwayml.com/resumed.mp4"],
                },
            )
        if request.url.host == "cdn.runwayml.com":
            return httpx.Response(200, content=b"resumed video")
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        provider = _provider(client)
        with pytest.raises(KeyboardInterrupt):
            provider.generate_video(
                prompt="Continue the same filter demonstration.",
                output_path=output_path,
                duration_seconds=5,
                first_frame=image_path,
                aspect_ratio="16:9",
            )

        task_state_path = tmp_path / "scene.runway-task.json"
        assert task_state_path.is_file()
        assert (tmp_path / "scene.runway-input.png").read_bytes() == original_image

        image_path.write_bytes(b"regenerated image")
        result = provider.generate_video(
            prompt="Continue the same filter demonstration.",
            output_path=output_path,
            duration_seconds=5,
            first_frame=image_path,
            aspect_ratio="16:9",
        )

    assert sum(request.method == "POST" for request in requests) == 1
    assert image_path.read_bytes() == original_image
    assert output_path.read_bytes() == b"resumed video"
    assert result.generation_id == "resumable-task"
    assert not task_state_path.exists()
    assert not (tmp_path / "scene.runway-input.png").exists()


def test_runway_does_not_resubmit_when_saved_request_changed(tmp_path):
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"image")
    output_path = tmp_path / "scene.mp4"

    def handle_request(request):
        if request.method == "POST":
            return httpx.Response(200, json={"id": "pending-task"})
        raise KeyboardInterrupt

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        provider = _provider(client)
        with pytest.raises(KeyboardInterrupt):
            provider.generate_video(
                prompt="Original prompt.",
                output_path=output_path,
                duration_seconds=5,
                first_frame=image_path,
            )

        with pytest.raises(VideoProviderError, match="avoid.*duplicate"):
            provider.generate_video(
                prompt="Changed prompt.",
                output_path=output_path,
                duration_seconds=5,
                first_frame=image_path,
            )


def test_runway_polling_times_out_and_keeps_resumable_task(
    tmp_path,
    capsys,
):
    image_path = tmp_path / "scene.png"
    image_path.write_bytes(b"image")
    output_path = tmp_path / "unused.mp4"

    def handle_request(request):
        if request.method == "POST":
            return httpx.Response(200, json={"id": "pending-task"})
        return httpx.Response(
            200,
            json={"id": "pending-task", "status": "RUNNING"},
        )

    with httpx.Client(transport=httpx.MockTransport(handle_request)) as client:
        with pytest.raises(VideoGenerationTimeout):
            _provider(
                client,
                poll_interval_seconds=0.001,
                timeout_seconds=0.005,
            ).generate_video(
                prompt="A test clip.",
                output_path=output_path,
                duration_seconds=5,
                first_frame=image_path,
            )

    assert (tmp_path / "unused.runway-task.json").is_file()
    assert "rerun with the same storyboard" in capsys.readouterr().out


def test_runway_rejects_non_https_output_url(tmp_path):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: pytest.fail(f"Unexpected request: {request.url}")
        )
    ) as client:
        provider = _provider(client)
        with pytest.raises(VideoProviderError, match="non-HTTPS"):
            provider._download_video(
                client,
                "http://cdn.runwayml.com/clip.mp4",
                tmp_path / "video.mp4",
                time.monotonic() + 5,
            )
