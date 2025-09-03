from pydantic import BaseModel, Field
from typing import Dict, Any

class GenerateRequest(BaseModel):
    preset: str = Field(..., examples=["txt2img"])
    params: Dict[str, Any] = Field(default_factory=dict)