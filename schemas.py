from pydantic import BaseModel, Field


class AudioRequest(BaseModel):
    audioBase64: str = Field(
        ...,
        description="Base64 encoded audio file"
    )

    class Config:
        extra = "allow"


class DetectionResponse(BaseModel):
    result: str
    confidence: float
