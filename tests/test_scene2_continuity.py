from app.assets.image_generator import (
    _token_count,
    build_continuity_image_prompt,
)
from tests.test_image_prompt import _circuit_storyboard


def test_scene2_prompt_preserves_reference_state_and_action():
    storyboard = _circuit_storyboard()
    scene = storyboard.scenes[1]
    reference_frames = ["previous-final-frame.png"]

    prompt = build_continuity_image_prompt(
        storyboard=storyboard,
        scene=scene,
        reference_frames=reference_frames,
    )

    assert "same battery, wire, open switch, unlit bulb" in prompt.lower()
    assert "close switch; bulb lights" in prompt.lower()
    assert "preserve reference composition" in prompt.lower()
    assert _token_count(prompt) <= 75
