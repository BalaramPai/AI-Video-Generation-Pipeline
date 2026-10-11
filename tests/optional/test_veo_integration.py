import os

import pytest

from app.config import GEMINI_API_KEY, VEO_MODEL
from app.video.providers import VeoVideoGenerationProvider


pytestmark = pytest.mark.skipif(
    os.getenv("VEO_INTEGRATION_TEST") != "1" or not GEMINI_API_KEY,
    reason=(
        "Set GEMINI_API_KEY and VEO_INTEGRATION_TEST=1 to run paid Veo "
        "integration tests."
    ),
)


def test_live_veo_generation(tmp_path):
    provider = VeoVideoGenerationProvider(
        api_key=GEMINI_API_KEY,
        model=VEO_MODEL,
    )
    result = provider.generate_video(
        prompt="A cinematic shot of sunlight moving across a quiet ocean.",
        output_path=tmp_path / "veo_integration.mp4",
        duration_seconds=4,
    )

    assert result.video_path.is_file()
    assert result.video_path.stat().st_size > 0
