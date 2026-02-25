from pydantic import BaseModel, Field


class AudioRequest(BaseModel):
    audio_base64: str = Field(
        ...,
        description="Base64 encoded MP3 audio"
    )


class DetectionResponse(BaseModel):
    result: str = Field(
        ...,
        description="AI_GENERATED or HUMAN"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to process audio file"
        )