import json

from app.storyboard.generator import generate_storyboard


def main():
    idea = input("\nEnter your video idea:\n> ").strip()

    if not idea:
        raise ValueError("Video idea cannot be empty.")

    print("\nGenerating storyboard...\n")

    storyboard = generate_storyboard(idea)

    storyboard_data = storyboard.model_dump()

    print(json.dumps(
        storyboard_data,
        indent=2,
        ensure_ascii=False,
    ))

    output_path = "outputs/storyboard.json"

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            storyboard_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nStoryboard saved to: {output_path}")


if __name__ == "__main__":
    main()