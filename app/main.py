import json
from pathlib import Path

from app.assets.image_generator import (
    build_continuity_image_prompt,
    build_scene_image_prompt,
    generate_scene_image,
    generate_scene_image_from_reference,
)
from app.storyboard.generator import generate_storyboard
from app.video.composer import concatenate_clips
from app.video.continuity import build_continuity_reference
from app.video.frame_extractor import extract_last_frames
from app.video.generator import generate_scene_video


OUTPUT_DIR = Path("outputs")


def main():
    idea = input("\nEnter your video idea:\n> ").strip()

    if not idea:
        raise ValueError("Video idea cannot be empty.")

    print("\nGenerating storyboard...\n")

    storyboard = generate_storyboard(idea)

    storyboard_data = storyboard.model_dump()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(json.dumps(
        storyboard_data,
        indent=2,
        ensure_ascii=False,
    ))

    output_path = OUTPUT_DIR / "storyboard.json"

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

    clip_paths: list[str] = []
    previous_video: str | None = None

    for index, scene in enumerate(storyboard.scenes):
        scene_number = index + 1
        image_path = OUTPUT_DIR / f"scene_{scene_number:03d}.png"
        video_path = OUTPUT_DIR / f"scene_{scene_number:03d}.mp4"

        if index == 0:
            image_prompt = build_scene_image_prompt(storyboard, scene)
            generate_scene_image(
                prompt=image_prompt,
                output_path=str(image_path),
            )
        else:
            if previous_video is None:
                raise RuntimeError(
                    "A previous scene video is required for continuity."
                )

            reference_dir = OUTPUT_DIR / (
                f"scene_{index:03d}_end_frames"
            )
            reference_frames = build_continuity_reference(
                extract_last_frames(
                    video_path=previous_video,
                    output_dir=str(reference_dir),
                )
            )
            image_prompt = build_continuity_image_prompt(
                storyboard=storyboard,
                scene=scene,
                reference_frames=reference_frames,
            )
            generate_scene_image_from_reference(
                prompt=image_prompt,
                reference_image_path=reference_frames[-1],
                output_path=str(image_path),
            )

        motion_prompt = (
            f"{scene.action}. "
            f"{scene.visual_description} "
            f"Natural realistic movement, continuous action, "
            f"{scene.camera.movement}, {scene.camera.shot_type}, "
            f"{scene.camera.angle}. "
            "Keep the characters, clothing, location, lighting, and props "
            "consistent with the input image."
        )
        previous_video = generate_scene_video(
            image_path=str(image_path),
            prompt=motion_prompt,
            output_path=str(video_path),
            duration_seconds=scene.duration_seconds,
        )
        clip_paths.append(previous_video)

    final_path = OUTPUT_DIR / "final_video.mp4"
    concatenate_clips(
        clip_paths=clip_paths,
        output_path=str(final_path),
    )
    print(f"\nFinal video saved to: {final_path}")


if __name__ == "__main__":
    main()