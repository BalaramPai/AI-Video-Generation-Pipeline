from app.config import (
    RUNWAY_API_SECRET,
    RUNWAY_MODEL,
    RUNWAY_POLL_INTERVAL_SECONDS,
    RUNWAY_TIMEOUT_SECONDS,
    VIDEO_PROVIDER,
    GEMINI_API_KEY,
    VEO_MODEL,
    VEO_POLL_INTERVAL_SECONDS,
    VEO_TIMEOUT_SECONDS,
)
from app.video.providers.base import VideoGenerationProvider
from app.video.providers.runway import RunwayVideoGenerationProvider
from app.video.providers.veo import VeoVideoGenerationProvider


def create_video_provider(
    provider_name: str | None = None,
) -> VideoGenerationProvider:
    selected_provider = (
        VIDEO_PROVIDER if provider_name is None else provider_name
    ).strip().lower()
    if selected_provider == "runway":
        return RunwayVideoGenerationProvider(
            api_secret=RUNWAY_API_SECRET,
            model=RUNWAY_MODEL,
            poll_interval_seconds=RUNWAY_POLL_INTERVAL_SECONDS,
            timeout_seconds=RUNWAY_TIMEOUT_SECONDS,
        )
    if selected_provider == "veo":
        return VeoVideoGenerationProvider(
            api_key=GEMINI_API_KEY,
            model=VEO_MODEL,
            poll_interval_seconds=VEO_POLL_INTERVAL_SECONDS,
            timeout_seconds=VEO_TIMEOUT_SECONDS,
        )
    raise ValueError(
        f"Unsupported VIDEO_PROVIDER={selected_provider!r}. "
        "Use 'runway', 'veo', or 'ffmpeg'."
    )
