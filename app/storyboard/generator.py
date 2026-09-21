# from openai import OpenAI

# from app.config import OPENAI_API_KEY, LLM_MODEL
# from app.prompts.storyboard_prompt import STORYBOARD_SYSTEM_PROMPT
# from app.storyboard.models import Storyboard


# client = OpenAI(api_key=OPENAI_API_KEY)


# def generate_storyboard(idea: str) -> Storyboard:
#     response = client.responses.parse(
#         model=LLM_MODEL,
#         instructions=STORYBOARD_SYSTEM_PROMPT,
#         input=idea,
#         text_format=Storyboard,
#     )

#     storyboard = response.output_parsed

#     if storyboard is None:
#         raise RuntimeError(
#             "The LLM did not return a valid storyboard."
#         )

#     return storyboard

# import json

# import ollama

# from app.prompts.storyboard_prompt import STORYBOARD_SYSTEM_PROMPT
# from app.storyboard.models import Storyboard


# OLLAMA_MODEL = "llama3.1:8b"


# def generate_storyboard(idea: str) -> Storyboard:
#     response = ollama.chat(
#         model=OLLAMA_MODEL,
#         messages=[
#             {
#                 "role": "system",
#                 "content": STORYBOARD_SYSTEM_PROMPT,
#             },
#             {
#                 "role": "user",
#                 "content": idea,
#             },
#         ],
#         format=Storyboard.model_json_schema(),
#         options={
#             "temperature": 0.2,
#         },
#     )

#     raw_content = response["message"]["content"]

#     try:
#         data = json.loads(raw_content)
#     except json.JSONDecodeError as exc:
#         raise RuntimeError(
#             f"Ollama returned invalid JSON:\n{raw_content}"
#         ) from exc

#     try:
#         return Storyboard.model_validate(data)
#     except Exception as exc:
#         raise RuntimeError(
#             f"Storyboard failed Pydantic validation:\n{exc}"
#         ) from exc


import json
import re

import ollama

from app.config import OLLAMA_MODEL
from app.prompts.storyboard_prompt import STORYBOARD_SYSTEM_PROMPT
from app.storyboard.models import Storyboard


def generate_storyboard(idea: str) -> Storyboard:
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": STORYBOARD_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": idea,
                },
            ],
            format=Storyboard.model_json_schema(),
            options={
                "temperature": 0.2,
            },
        )
    except (ConnectionError, ollama.ResponseError) as exc:
        raise RuntimeError(
            "Ollama is not available. Start the Ollama application, then "
            f"run `ollama pull {OLLAMA_MODEL}` before retrying."
        ) from exc

    raw_content = response["message"]["content"]

    try:
        data = json.loads(raw_content)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Ollama returned invalid JSON:\n{raw_content}"
        ) from exc

    # --------------------------------------------------
    # REPAIR DETERMINISTIC CONTINUITY FIELDS
    # --------------------------------------------------

    repair_storyboard_continuity(data)

    # --------------------------------------------------
    # VALIDATE WITH PYDANTIC
    # --------------------------------------------------

    storyboard = Storyboard.model_validate(data)

    validate_storyboard_continuity(storyboard)

    return storyboard


def repair_storyboard_continuity(data: dict) -> None:
    """Normalize deterministic continuity fields returned by the LLM."""
    scenes = data.get("scenes", [])

    for index, scene in enumerate(scenes):
        continuity = scene.setdefault("continuity", {})
        visual_description = scene.get("visual_description", "")
        action = scene.get("action", "")

        if index == 0:
            continuity["previous_scene_id"] = None
        else:
            previous_scene = scenes[index - 1]
            previous_scene_id = previous_scene["id"]
            continuity["previous_scene_id"] = previous_scene_id

            required_start = continuity.get("required_start_state") or ""
            symbolic_state = re.fullmatch(
                r"(?:previous_scene|" + re.escape(previous_scene_id)
                + r")\.ending_state",
                required_start.strip(),
                flags=re.IGNORECASE,
            )

            if symbolic_state:
                continuity["required_start_state"] = (
                    previous_scene["continuity"]["ending_state"]
                )

        ending_state = continuity.get("ending_state")
        if ending_state is None or str(ending_state).strip().lower() in {
            "",
            "null",
            "none",
        }:
            continuity["required_start_state"] = (
                continuity.get("required_start_state")
                or action
            )
            continuity["ending_state"] = (
                visual_description or action
            )

        # Optional detail fields are still made explicit in the persisted
        # storyboard so downstream prompt builders have a stable shape.
        scene.setdefault("blocking", "")
        scene.setdefault("shot_intent", "")
        scene.setdefault("visual_continuity", "")
        scene.setdefault("negative_prompt", "")

def validate_storyboard_continuity(storyboard: Storyboard) -> None:

    scenes = storyboard.scenes

    if not scenes:
        raise ValueError("Storyboard must contain at least one scene.")

    scene_ids = {scene.id for scene in scenes}
    if len(scene_ids) != len(scenes):
        raise ValueError("Scene IDs must be unique.")

    location_ids = {
        location.id
        for location in storyboard.locations
    }

    character_ids = {
        character.id
        for character in storyboard.characters
    }

    prop_ids = {
        prop.id
        for prop in storyboard.props
    }

    if len(location_ids) != len(storyboard.locations):
        raise ValueError("Location IDs must be unique.")
    if len(character_ids) != len(storyboard.characters):
        raise ValueError("Character IDs must be unique.")
    if len(prop_ids) != len(storyboard.props):
        raise ValueError("Prop IDs must be unique.")

    for index, scene in enumerate(scenes):

        # ---------------------------------------------
        # SCENE ID
        # ---------------------------------------------

        if not scene.id:
            raise ValueError("Every scene must have an ID.")

        # ---------------------------------------------
        # PREVIOUS SCENE
        # ---------------------------------------------

        if index == 0:

            if scene.continuity.previous_scene_id is not None:
                raise ValueError(
                    "First scene must have previous_scene_id=None."
                )

        else:

            previous_scene = scenes[index - 1]

            if scene.continuity.previous_scene_id != previous_scene.id:
                raise ValueError(
                    f"{scene.id} must reference the previous "
                    f"scene {previous_scene.id}."
                )

        # ---------------------------------------------
        # LOCATION
        # ---------------------------------------------

        if scene.location_id not in location_ids:
            raise ValueError(
                f"{scene.id} references unknown location "
                f"{scene.location_id}."
            )

        if len(scene.character_ids) != len(set(scene.character_ids)):
            raise ValueError(
                f"{scene.id} contains duplicate character IDs."
            )
        if len(scene.prop_ids) != len(set(scene.prop_ids)):
            raise ValueError(
                f"{scene.id} contains duplicate prop IDs."
            )

        # ---------------------------------------------
        # CHARACTERS
        # ---------------------------------------------

        for character_id in scene.character_ids:

            if character_id not in character_ids:
                raise ValueError(
                    f"{scene.id} references unknown character "
                    f"{character_id}."
                )

        # ---------------------------------------------
        # PROPS
        # ---------------------------------------------

        for prop_id in scene.prop_ids:

            if prop_id not in prop_ids:
                raise ValueError(
                    f"{scene.id} references unknown prop "
                    f"{prop_id}."
                )

        # ---------------------------------------------
        # CONTINUITY STATE
        # ---------------------------------------------

        required_start = scene.continuity.required_start_state

        if index == 0:

            if required_start.lower() in {
                "null",
                "none",
                "",
            }:
                pass

        else:

            previous_scene = scenes[index - 1]

            if required_start.lower() in {
                "null",
                "none",
                "",
            }:
                raise ValueError(
                    f"{scene.id} must have a real "
                    "required_start_state."
                )

            if previous_scene.continuity.ending_state.lower() in {
                "null",
                "none",
                "",
            }:
                raise ValueError(
                    f"{previous_scene.id} must have a real "
                    "ending_state."
                )

            # Reject references like:
            # SCENE_001.ending_state
            if ".ending_state" in required_start:
                raise ValueError(
                    f"{scene.id} contains a symbolic "
                    "required_start_state instead of "
                    "the actual physical state."
                )

        # ---------------------------------------------
        # ENDING STATE
        # ---------------------------------------------

        if scene.continuity.ending_state.lower() in {
            "null",
            "none",
            "",
        }:
            raise ValueError(
                f"{scene.id} must have a real ending_state."
            )

        # ---------------------------------------------
        # CHARACTER STATE
        # ---------------------------------------------

        if scene.continuity.character_state.lower() in {
            "null",
            "none",
            "",
        }:
            raise ValueError(
                f"{scene.id} must have a real character_state."
            )

        # ---------------------------------------------
        # ENVIRONMENT STATE
        # ---------------------------------------------

        if scene.continuity.environment_state.lower() in {
            "null",
            "none",
            "",
        }:
            raise ValueError(
                f"{scene.id} must have a real environment_state."
            )
