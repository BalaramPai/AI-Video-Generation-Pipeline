# AI Video Generation Pipeline

This pipeline turns a video idea into a storyboard, generates scene images,
animates each image with Runway image-to-video, preserves continuity between
scenes, and concatenates the generated clips into one MP4.

## Setup

1. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

2. Install and start Ollama, then pull the configured storyboard model:

   ```powershell
   ollama pull llama3.1:8b
   ```

3. Copy `.env.example` to `.env`.

4. Add a Runway API secret to `.env`:

   ```text
   RUNWAYML_API_SECRET=your_runway_secret
   ```

   The secret is used only for the image-to-video requests and must not be
   committed.

## Generate a real video

Run from the project directory:

```powershell
python -m app.main
```

The pipeline writes the storyboard, scene images, scene videos, continuity
frames, and the final video to `outputs/`.

Runway generates actual motion from each scene image using the scene action,
visual description, camera movement, and continuity instructions. The final
video is assembled only after all scene clips have been generated.

## Continuity and artifact controls

The storyboard is a production contract, not just a scene list. Each
character can carry immutable eye color, distinctive-feature, and wardrobe
anchors; each location and prop can carry a visual continuity anchor; and each
scene can specify blocking, shot intent, visual continuity, and a negative
prompt. The storyboard prompt instructs Ollama to keep these facts stable and
the validator rejects broken scene references, duplicate IDs, and duplicate
character or prop assignments.

Image prompts include every character in the scene, the scene action, and the
camera framing. They are compacted against the Stable Diffusion 1.5 CLIP
limit instead of being silently truncated. Continuation images use the last
frame as an img2img reference with a low denoise strength so identity and
composition are preserved rather than redrawn. Generated motion prompts
repeat the start state, blocking, prop state, and end state and explicitly
exclude identity morphing, extra limbs, duplicate people, costume changes,
prop teleportation, flicker, and sudden cuts.

These controls reduce, but cannot mathematically eliminate, artifacts from
the underlying image-to-video model. For the most reliable results, keep
each scene to one primary action, one camera movement, and a small number of
characters and interacting props.

## Local fallback

If no hosted video provider is available, set:

```text
VIDEO_PROVIDER=ffmpeg
```

This fallback creates a camera effect from still images; it is not AI-generated
motion and is intended only for offline testing.
