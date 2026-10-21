import json
from pathlib import Path
from typing import Any

from app.storyboard.generator import validate_storyboard_continuity
from app.storyboard.models import Storyboard


def parse_storyboard_json(raw_json: str) -> Storyboard:
    try:
        data: Any = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Storyboard input is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})."
        ) from exc

    if not isinstance(data, dict):
        raise ValueError("Storyboard JSON must contain a top-level object.")

    storyboard = Storyboard.model_validate(data)
    validate_storyboard_continuity(storyboard)
    return storyboard


def load_storyboard_file(file_path: str | Path) -> Storyboard:
    path = Path(file_path)
    try:
        raw_json = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise OSError(f"Could not read storyboard file {path}: {exc}") from exc
    return parse_storyboard_json(raw_json)
