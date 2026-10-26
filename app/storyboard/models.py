# from typing import List, Optional

# from pydantic import BaseModel, Field


# class Project(BaseModel):
#     title: str
#     description: str
#     duration_seconds: int = Field(gt=0)
#     aspect_ratio: str = "16:9"
#     visual_style: str


# class CharacterAppearance(BaseModel):
#     age: Optional[int] = None
#     hair: str
#     clothing: str
#     facial_features: str
#     physical_description: str


# class Character(BaseModel):
#     id: str
#     name: str
#     appearance: CharacterAppearance


# class Location(BaseModel):
#     id: str
#     name: str
#     description: str
#     lighting: str
#     visual_style: str


# class Prop(BaseModel):
#     id: str
#     name: str
#     description: str


# class Camera(BaseModel):
#     shot_type: str
#     angle: str
#     movement: str
#     lens: Optional[str] = None


# class Continuity(BaseModel):
#     previous_scene_id: Optional[str] = None
#     character_state: str
#     environment_state: str
#     required_start_state: str
#     ending_state: str


# class Scene(BaseModel):
#     id: str
#     duration_seconds: int = Field(gt=0)

#     location_id: str
#     character_ids: List[str]
#     prop_ids: List[str]

#     action: str
#     dialogue: Optional[str] = None
#     visual_description: str

#     camera: Camera
#     continuity: Continuity


# class Storyboard(BaseModel):
#     project: Project
#     characters: List[Character]
#     locations: List[Location]
#     props: List[Prop]
#     scenes: List[Scene]

from typing import Any, List, Optional

from pydantic import BaseModel, Field, model_validator


class Project(BaseModel):
    title: str
    description: str
    duration_seconds: int = Field(gt=0)
    aspect_ratio: str = "16:9"
    visual_style: str


class CharacterAppearance(BaseModel):
    age: int = Field(gt=0)
    hair: str
    clothing: str
    facial_features: str
    physical_description: str
    eye_color: str = ""
    distinctive_features: str = ""
    wardrobe_anchor: str = ""


class Character(BaseModel):
    id: str
    name: str
    appearance: CharacterAppearance
    role: str = ""


class Location(BaseModel):
    id: str
    name: str
    description: str
    lighting: str
    visual_style: str
    continuity_anchor: str = ""


class Prop(BaseModel):
    id: str
    name: str
    description: str
    continuity_anchor: str = ""


class Camera(BaseModel):
    shot_type: str
    angle: str
    movement: str
    lens: Optional[str] = None


class Continuity(BaseModel):
    previous_scene_id: Optional[str] = None
    character_state: str
    environment_state: str
    required_start_state: str
    ending_state: str


class Scene(BaseModel):
    id: str
    previous_scene_id: Optional[str]
    duration_seconds: int = Field(gt=0)

    location_id: str
    character_ids: List[str]
    prop_ids: List[str]

    action: str
    dialogue: Optional[str] = None
    visual_description: str

    camera: Camera
    continuity: Continuity
    blocking: str = ""
    shot_intent: str = ""
    visual_continuity: str = ""
    negative_prompt: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_previous_scene_id(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        normalized = dict(values)
        continuity = normalized.get("continuity")
        if isinstance(continuity, Continuity):
            nested_present = (
                "previous_scene_id" in continuity.model_fields_set
            )
            nested_link = continuity.previous_scene_id
            continuity_values = continuity.model_dump()
        elif isinstance(continuity, dict):
            nested_present = "previous_scene_id" in continuity
            nested_link = continuity.get("previous_scene_id")
            continuity_values = dict(continuity)
        else:
            nested_present = False
            nested_link = None
            continuity_values = continuity

        top_level_present = "previous_scene_id" in normalized
        if not top_level_present and not nested_present:
            raise ValueError(
                "Scene must define previous_scene_id at the scene level "
                "or in continuity."
            )
        if (
            top_level_present
            and nested_present
            and normalized["previous_scene_id"] != nested_link
        ):
            raise ValueError(
                "Top-level previous_scene_id conflicts with "
                "continuity.previous_scene_id."
            )

        canonical_link = (
            normalized["previous_scene_id"]
            if top_level_present
            else nested_link
        )
        normalized["previous_scene_id"] = canonical_link
        if isinstance(continuity_values, dict):
            continuity_values["previous_scene_id"] = canonical_link
            normalized["continuity"] = continuity_values
        return normalized


class Storyboard(BaseModel):
    project: Project
    characters: List[Character]
    locations: List[Location]
    props: List[Prop]
    scenes: List[Scene]