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

## Local fallback

If no hosted video provider is available, set:

```text
VIDEO_PROVIDER=ffmpeg
```

This fallback creates a camera effect from still images; it is not AI-generated
motion and is intended only for offline testing.
