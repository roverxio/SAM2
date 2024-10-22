from typing import List, Optional

from pydantic import BaseModel

from enums import MediaType, SAMModels


class Pointers(BaseModel):
    x: float
    y: float
    label: int


class SAMRequest(BaseModel):
    media_type: MediaType
    media_url: str
    model: SAMModels
    frame_idx: Optional[int] = None
    pointers: List[Pointers]
