from pathlib import Path


def build_continuity_reference(
    frame_paths: list[str],
) -> list[str]:

    if not frame_paths:
        raise ValueError(
            "At least one reference frame is required."
        )

    for frame in frame_paths:
        if not Path(frame).exists():
            raise FileNotFoundError(
                f"Reference frame not found: {frame}"
            )

    return frame_paths


def get_primary_reference_frame(
    frame_paths: list[str],
) -> str:

    references = build_continuity_reference(frame_paths)

    # The final frame represents the latest visual state.
    return references[-1]