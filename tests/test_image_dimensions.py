import pytest

from app.assets.image_generator import _image_dimensions


@pytest.mark.parametrize(
    ("ratio", "dimensions"),
    [
        ("16:9", (768, 432)),
        ("9:16", (432, 768)),
        ("4:3", (576, 432)),
        ("3:4", (432, 576)),
        ("1:1", (512, 512)),
        ("3:2", (768, 512)),
        ("2:3", (512, 768)),
    ],
)
def test_image_dimensions_match_requested_aspect_ratio(ratio, dimensions):
    assert _image_dimensions(ratio) == dimensions


def test_image_dimensions_reject_unsupported_aspect_ratio():
    with pytest.raises(ValueError, match="Unsupported image aspect ratio"):
        _image_dimensions("7:5")
