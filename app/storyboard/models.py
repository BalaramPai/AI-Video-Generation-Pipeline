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

from typing import List, Optional

from pydantic import BaseModel, Field


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


class Storyboard(BaseModel):
    project: Project
    characters: List[Character]
    locations: List[Location]
    props: List[Prop]
    scenes: List[Scene]