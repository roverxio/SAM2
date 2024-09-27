from typing import List, Optional

from pydantic import BaseModel, field_validator

from enums import MediaType, SAMModels


class Pointers(BaseModel):
    x: int
    y: int
    label: int


class SAMRequest(BaseModel):
    media_type: MediaType
    media_url: str
    model: SAMModels
    frame_idx: Optional[int]
    pointers: List[Pointers]

