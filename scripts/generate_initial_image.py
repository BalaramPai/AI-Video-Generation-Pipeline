import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.storyboard.io import load_storyboard_file
from app.assets.image_generator import (
    build_initial_scene_image_prompt,
    build_initial_scene_negative_prompt,
    generate_scene_image,
)

storyboard = load_storyboard_file("outputs/storyboard.json")
scene = storyboard.scenes[0]

prompt = build_initial_scene_image_prompt(storyboard, scene)
negative_prompt = build_initial_scene_negative_prompt(storyboard, scene)

print("\n=== INITIAL LUNAR IMAGE ===")
print(f"Scene: {scene.id}")
print(f"Positive prompt: {prompt}")
print(f"Negative prompt: {negative_prompt}")

output = Path("outputs/lunar_demo/scene_001.png")
metadata = Path("outputs/lunar_demo/scene_001_image_generation.json")

generate_scene_image(
    prompt=prompt,
    negative_prompt=negative_prompt,
    output_path=str(output),
    aspect_ratio=storyboard.project.aspect_ratio,
    metadata_path=metadata,
)

print(f"\nImage: {output.resolve()}")
print(f"Metadata: {metadata.resolve()}")
