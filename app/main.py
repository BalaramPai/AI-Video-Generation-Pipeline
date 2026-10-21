import argparse
import sys
from pathlib import Path

from app.storyboard.io import load_storyboard_file, parse_storyboard_json
from app.storyboard.models import Storyboard
from app.video.pipeline import generate_video_from_storyboard


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
    args = parser.parse_args()

    storyboard = _load_input_storyboard(args.storyboard)
    try:
        output_path = generate_video_from_storyboard(
            storyboard,
            output_dir=Path(args.output_dir),
        )
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
