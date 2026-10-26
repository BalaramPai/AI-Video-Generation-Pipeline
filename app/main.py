import argparse
import sys
from pathlib import Path

from app.storyboard.io import load_storyboard_file, parse_storyboard_json
from app.storyboard.models import Scene, Storyboard
from app.video.pipeline import (
    ImageReviewDeclined,
    generate_video_from_storyboard,
    preflight_storyboard,
    PreflightValidationError,
)


def _read_pasted_storyboard() -> Storyboard:
    print(
        "Paste the complete storyboard JSON below. "
        "When finished, enter END on its own line:"
    )
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)

    if not lines or not any(line.strip() for line in lines):
        raise ValueError("No storyboard JSON was provided.")
    return parse_storyboard_json("\n".join(lines))


def _load_input_storyboard(file_argument: str | None) -> Storyboard:
    if file_argument == "-":
        return parse_storyboard_json(sys.stdin.read())
    if file_argument:
        return load_storyboard_file(file_argument)

    source = input(
        "Storyboard JSON file path (or press Enter to paste JSON): "
    ).strip()
    if source:
        return load_storyboard_file(source)
    return _read_pasted_storyboard()


def _review_scene_image(scene: Scene, image_path: Path) -> bool:
    print(
        f"\nReview the first-frame image for {scene.id}: {image_path}\n"
        "Open the image and confirm it is the composition you want to send "
        "to the video provider."
    )
    answer = input("Generate this scene's video? [Y/n] ").strip().lower()
    return answer in {"", "y", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a video from an existing storyboard JSON."
    )
    parser.add_argument(
        "storyboard",
        nargs="?",
        help=(
            "Path to storyboard JSON. Use '-' to read JSON from standard "
            "input; omit to choose a file or paste interactively."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for scene assets and final video (default: outputs).",
    )
    parser.add_argument(
        "--review-images",
        action="store_true",
        help=(
            "Pause before each new video request so you can inspect its "
            "first-frame image. Existing Runway tasks resume without review."
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Validate storyboard references and compile all image/video "
            "prompts without generating images or videos."
        ),
    )
    args = parser.parse_args()

    try:
        storyboard = _load_input_storyboard(args.storyboard)
    except (OSError, ValueError) as exc:
        if args.preflight:
            print(f"Preflight failed: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        raise
    if args.preflight:
        try:
            report = preflight_storyboard(storyboard)
        except PreflightValidationError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
        scene_label = (
            "scene" if len(report.motion_prompts) == 1 else "scenes"
        )
        print(
            "Preflight passed: "
            f"{len(report.motion_prompts)} {scene_label}; initial image prompt "
            f"{report.initial_image_prompt_tokens}/75 tokens."
        )
        for scene, diagnostics in zip(
            storyboard.scenes,
            report.motion_prompt_diagnostics,
        ):
            count = diagnostics["video_prompt_character_count"]
            limit = diagnostics["video_prompt_character_limit"]
            unit = diagnostics["video_prompt_character_count_unit"]
            if limit is None:
                print(
                    f"  {scene.id}: motion prompt {count} ({unit}; "
                    "provider limit not declared)."
                )
            else:
                print(
                    f"  {scene.id}: motion prompt {count}/{limit} {unit}."
                )
        return
    try:
        output_path = generate_video_from_storyboard(
            storyboard,
            output_dir=Path(args.output_dir),
            image_review_callback=(
                _review_scene_image if args.review_images else None
            ),
        )
    except PreflightValidationError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    except ImageReviewDeclined as exc:
        print(f"\n{exc}", file=sys.stderr)
        return
    except KeyboardInterrupt:
        task_states = sorted(
            Path(args.output_dir).glob("*.runway-task.json")
        )
        print(
            "\nGeneration interrupted. A submitted Runway task may continue "
            "remotely.",
            file=sys.stderr,
        )
        if task_states:
            print(
                "Saved task state found. Rerun with the same storyboard and "
                "output directory to resume without submitting another "
                "generation request:",
                file=sys.stderr,
            )
            for task_state in task_states:
                print(f"  {task_state}", file=sys.stderr)
        else:
            print(
                "No resumable task state was found; check your Runway task "
                "history before retrying to avoid duplicate generation.",
                file=sys.stderr,
            )
        return
    print(f"\nFinal video saved to: {output_path}")


if __name__ == "__main__":
    main()
