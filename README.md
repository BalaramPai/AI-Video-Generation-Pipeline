# AI Video Generation Pipeline

This pipeline takes an existing storyboard JSON, generates scene images with
Stable Diffusion, animates them with a selectable video model, preserves visual
continuity between scenes, and joins the generated clips into one MP4.

## Setup

1. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env`.

3. Add your Runway API secret to `.env`:

   ```text
   RUNWAYML_API_SECRET=your_runway_api_secret
   ```

   The secret is sent only to Runway's API and must not be committed.

      Img2img continuity strength defaults to `0.30`. It can be varied between
      `0.0` and `1.0` for controlled experiments without changing code:

      ```text
      IMAGE_TO_IMAGE_STRENGTH=0.30
      ```

4. Choose the provider and model in `.env`:

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

Run from the project directory and provide a storyboard JSON file:

```powershell
python -m app.main path\to\storyboard.json
```

To paste JSON interactively instead, run without a path, press Enter at the
file-path prompt, paste the complete JSON, then enter `END` on its own line:

```powershell
python -m app.main
```

For scripts or another application, JSON can be piped through standard input:

```powershell
Get-Content -Raw path\to\storyboard.json | python -m app.main -
```

The input is parsed and validated as a storyboard; this pipeline does not call
Ollama or create, repair, or redistribute the storyboard. Scene durations must
sum to the project duration, and each generated scene must be at most 10
seconds. The provider validates its own model-specific duration limits before
making requests. Scene images, scene videos, continuity frames, a copy of the
validated storyboard, and the final MP4 are written to `outputs/` by default.
Use `--output-dir path\to\output` to select another output directory.

Runway task IDs and the submitted first-frame image are saved beside each
scene video while generation is pending. If polling times out or you interrupt
the CLI, rerunning with the same storyboard and output directory resumes that
task instead of submitting another paid request. If the model, prompt,
duration, or aspect ratio changed, the pipeline refuses to submit until the
saved task is resolved; inspect the remote task before manually removing its
`scene_NNN.runway-task.json` state file.

Other applications can call the generation stage directly after constructing
a validated `Storyboard` model:

```python
from app.storyboard.io import load_storyboard_file
from app.video.pipeline import generate_video_from_storyboard

storyboard = load_storyboard_file("path/to/storyboard.json")
video_path = generate_video_from_storyboard(storyboard, output_dir="outputs")
```

Before joining, each scene is normalized to the measured frame rate of the
first generated clip and to a calculated frame count. Fractional frame
boundaries use exact rational arithmetic and round to the nearest frame, with
half-frame ties rounded up; each scene receives the difference between its
rounded cumulative end and start boundaries. The final clip is checked against
the sum of those scene frame counts, avoiding per-scene rounding drift and
unnecessary re-encoding during concatenation.

## Continuity and artifact controls

The storyboard is a production contract, not just a scene list. Each
character can carry immutable eye color, distinctive-feature, and wardrobe
anchors; each location and prop can carry a visual continuity anchor; and each
scene can specify blocking, shot intent, visual continuity, and a negative
prompt. The validator rejects broken scene references, duplicate IDs, and
duplicate character or prop assignments. Scenes without visible characters
are supported; their continuity subject state and visual description drive
the image and motion prompts.

Image prompts include every character in the scene, the scene action, and the
camera framing. They are compacted against a conservative Stable Diffusion
1.5 CLIP limit (75 tokenizer IDs including special tokens) instead of allowing
silent model-side truncation. If required details exceed that limit, the
builder fairly shortens text across the subject, start state, action, layout,
and props at word boundaries, and logs which fields were shortened. Short
storyboard descriptions still produce the best prompts; optional style,
location, and camera details are omitted only when necessary. Continuation
images use the last frame as an img2img reference with a low denoise strength
so identity and composition are preserved rather than redrawn. Generated
motion prompts repeat the start state, blocking, prop state, and end state and
explicitly exclude identity morphing, extra limbs, duplicate people, costume
changes, prop teleportation, flicker, and sudden cuts.

Runway motion prompts use the model-specific limits from the Runway API
schema: 1,000 UTF-16 code units for Gen-4.5 and 2,500 for Grok Imagine 1.5.
The compiler keeps the subject, starting state, action, ending state, and
camera instructions complete; it prioritizes complete negative clauses and
optional visual/context details after those fields. Omitted optional sections
and negative constraints are reported. If the required core alone is too
long, generation fails with a request to shorten the storyboard rather than
truncating an essential instruction.

Negative prompts are also kept within the model limit. Built-in quality and
character safeguards take priority; comma- or semicolon-separated storyboard
exclusions are included while space remains, and any omitted exclusions are
reported in a warning.

For each scene, the output directory also contains
`scene_NNN_image_prompt.txt`, `scene_NNN_video_prompt.txt`, and
`scene_NNN_generation.json`. These preserve the actual prompts, provider/model,
requested and measured timing, continuity references, and any generation
error, making it possible to inspect what the models received without
regenerating. Storyboard dialogue is retained as metadata; this pipeline does
not synthesize narration or render deterministic text labels/overlays.

These controls reduce, but cannot eliminate, artifacts from the underlying
video model. Each generated scene image is supplied as the video's starting
frame; for later scenes the Stable Diffusion img2img pipeline first uses
verified decoded frames extracted from the normalized previous scene video.
The extractor clears old frame files, validates image decoding and dimensions,
and falls back to the nearest earlier valid frame if the clip's final frame
cannot be used. If no valid reference can be extracted, generation stops.
Runway image-to-video supports a starting image but not a separate last frame
or reference-image list; the extracted frame is passed through img2img, and
that generated continuity image is uploaded as Runway's first-frame input.
Runway's API request does not contain a separate reference-image field. Keep
each scene to one primary action, one camera movement, and a small number of
characters and interacting props.

The current Runway Gen-4.5 and Grok Imagine 1.5 integration uses the
image-to-video endpoint and cannot take a preceding video as temporal
continuation input. The current Runway OpenAPI schema has a separate
video-to-video `extend` mode for `gemini_omni_flash_1.1`; that is a different
model and API operation and is not wired into this pipeline. Do not describe
the current Gen-4.5/Grok scene-by-scene workflow as provider-level temporal
continuation.

Each later scene's `scene_NNN_generation.json` also records the previous and
next scene IDs, source video timing, extracted candidate frame indexes and
timestamps, selected reference, img2img input/output, provider input image,
effective img2img strength, and transition status. Candidate timestamps are
calculated from decoded frame indexes and the normalized constant frame rate.
The final MP4 currently uses hard cuts; no crossfade is applied automatically.
This keeps transitions from hiding incompatible poses or motion.

### Evaluating continuity changes

The automated suite verifies file handoffs and timing, not perceived motion
quality. For a visual comparison, use the same storyboard, provider, model,
scene durations, and prompt text in separate output directories. Change only
one setting at a time (for example, `IMAGE_TO_IMAGE_STRENGTH=0.20`, `0.30`,
or `0.40`), and expect each Runway run to incur generation cost. The pipeline
does not launch these paid comparisons automatically.

For every scene boundary, inspect the last two seconds of the preceding clip
and the first two seconds of the next clip, then watch the complete output.
Record the storyboard version, provider/model, strength, selected reference
frame, prompt files, approximate cost, identity/pose changes, motion
direction, camera and lighting changes, flicker, and final duration. A still
frame that resembles the boundary does not prove that motion continues
smoothly.

The code retains an optional `VIDEO_PROVIDER=veo` integration; it requires
`GEMINI_API_KEY` and uses the `VEO_*` settings in `.env.example`. The current
Google Gemini video documentation states that the Veo 3.1 Preview in the
Gemini API has been shut down, so this older configured preview integration
may no longer be available. Check current Google documentation before using
it. Use
`VIDEO_PROVIDER=ffmpeg` only for the local still-image motion fallback; it is
not AI-generated motion.

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
