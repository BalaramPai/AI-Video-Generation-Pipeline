# AI Video Generation Pipeline

This pipeline turns a video idea into a storyboard, generates scene images
with Stable Diffusion, animates them with a selectable Runway video model,
preserves visual continuity between scenes, and joins the generated clips
into one MP4.

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

4. Add your Runway API secret to `.env`:

   ```text
   RUNWAYML_API_SECRET=your_runway_api_secret
   ```

   The secret is sent only to Runway's API and must not be committed.

5. Choose the provider and model in `.env`:

   ```text
   VIDEO_PROVIDER=runway
   RUNWAY_MODEL=gen4.5
   RUNWAY_POLL_INTERVAL_SECONDS=5
   RUNWAY_TIMEOUT_SECONDS=900
   ```

   To use Grok Imagine Video 1.5 through Runway instead, set
   `RUNWAY_MODEL=grok_imagine_1_5`. These are Runway API model IDs; no code
   changes are needed to switch between them. Both use the same Runway secret.

## Generate a real video

Run from the project directory:

```powershell
python -m app.main
```

The pipeline writes the storyboard, scene images, scene videos, continuity
frames, and the final video to `outputs/`.

Each scene image is submitted to Runway's image-to-video API with the selected
model. The pipeline polls the asynchronous task until it succeeds, fails, is
cancelled, or times out, then downloads the resulting MP4 into `outputs/`.
The final video is assembled only after all scene clips have been generated.

The storyboard may use any positive whole-number project duration. It is
normalized into exact scene durations, with each scene at most 10 seconds so
the same storyboard works with either Runway model. The model-specific limits
are validated before requests are made: Gen-4.5 supports 2–10 seconds per
clip, and Grok Imagine Video 1.5 supports 1–15 seconds. If a Gen-4.5 scene is
only one second, the provider generates the minimum two seconds and FFmpeg
trims it to the storyboard target. Include your desired runtime in the video
idea to have the storyboard target that exact length. Longer projects are
made from more scenes and concatenated; there is no single-task long-video
limit in the pipeline.

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

These controls reduce, but cannot eliminate, artifacts from the underlying
video model. Each generated scene image is supplied as the video's starting
frame; for later scenes the Stable Diffusion img2img pipeline first uses
frames extracted from the previous clip. Runway image-to-video supports a
starting image but not a separate last frame or reference-image list; the
continuity-adjusted starting image is used instead. Keep each scene to one
primary action, one camera movement, and a small number of characters and
interacting props.

The optional `VIDEO_PROVIDER=veo` setting keeps the Google Veo provider
available; it requires `GEMINI_API_KEY` and uses the `VEO_*` settings in
`.env.example`. Use `VIDEO_PROVIDER=ffmpeg` only for the local still-image
motion fallback; it is not AI-generated motion.

## Local fallback

FFmpeg must be installed and available on `PATH`, configured with
`FFMPEG_PATH`, or installed in the standard Windows WinGet location. The join
step re-encodes the completed sequence as MP4/H.264 with optional AAC audio
for reliable playback.

To use the local still-image fallback instead of hosted generation, set:

```text
VIDEO_PROVIDER=ffmpeg
```

This fallback creates a camera effect from still images; it is not
AI-generated motion and is intended only for offline testing.

## Tests

Run the normal mocked unit suite with:

```powershell
pytest
```

An optional live-generation test is kept under `tests/optional/` and is skipped
unless both `GEMINI_API_KEY` and `VEO_INTEGRATION_TEST=1` are set. It generates
a short clip, which may incur API cost:

```powershell
$env:VEO_INTEGRATION_TEST = "1"
pytest tests/optional/test_veo_integration.py
```

Runway model IDs, request fields, duration limits, and task lifecycle follow
the official [Runway API reference](https://docs.dev.runwayml.com/api/) and
[OpenAPI schema](https://docs.dev.runwayml.com/openapi.json).
