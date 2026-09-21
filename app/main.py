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


def _build_motion_prompt(storyboard, scene) -> str:
    location = next(
        location for location in storyboard.locations
        if location.id == scene.location_id
    )
    characters = [
        character for character in storyboard.characters
        if character.id in scene.character_ids
    ]
    character_lock = "; ".join(
        (
            f"{character.name}: age {character.appearance.age}, "
            f"{character.appearance.hair}, "
            f"{character.appearance.facial_features}, "
            f"{character.appearance.clothing}"
        )
        for character in characters
    )
    prop_lock = ", ".join(
        ", ".join(
            detail for detail in
            (prop.description, prop.continuity_anchor) if detail
        )
        for prop in storyboard.props
        if prop.id in scene.prop_ids
    )
    negative = scene.negative_prompt or (
        "No identity changes, face morphing, duplicate people, extra limbs, "
        "warped hands, costume changes, prop teleportation, text, logos, "
        "flicker, melting objects, or sudden camera cuts."
    )
    return (
        f"Continuous {scene.duration_seconds}s shot. "
        f"Location: {location.name}; light: {location.lighting}. "
        f"Identity lock: {character_lock}. "
        f"Props: {prop_lock or 'none'}. "
        f"Start: {scene.continuity.required_start_state}. "
        f"Blocking: {scene.blocking}. "
        f"Action: {scene.action}. "
        f"End: {scene.continuity.ending_state}. "
        f"Camera: {scene.camera.shot_type}, {scene.camera.angle}, "
        f"{scene.camera.movement}, {scene.camera.lens or 'natural lens'}. "
        f"Preserve faces, proportions, wardrobe, lighting, composition, "
        f"and prop positions; animate only this action naturally. "
        f"Avoid: {negative}"
    )


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
                negative_prompt=scene.negative_prompt,
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

        motion_prompt = _build_motion_prompt(storyboard, scene)
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