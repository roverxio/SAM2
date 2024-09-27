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

    @field_validator('frame_idx')
    def check_frame_idx(self, v, values):
        media_type = values.get('media_type')
        if media_type == MediaType.Video and v is None:
            raise ValueError('frame_idx must be provided when media_type is video.')
        return v
