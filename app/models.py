from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, field_validator

Role = Literal["doctor", "patient", "nurse", "other"]

class Word(BaseModel):
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    word: Optional[str] = None
    text: Optional[str] = None

    @field_validator("word", mode="before")
    @classmethod
    def accept_text_as_word(cls, v, values):
        if v is None and values.get("text"):
            return values["text"]
        return v

    class Config:
        extra = "allow"

class Turn(BaseModel):
    speaker: str
    text: Optional[str] = None
    words: Optional[List[Word]] = None

class ClassifiedTurn(Turn):
    role: Role
    display_name: str

class TranscribeResult(BaseModel):
    job_name: str
    service: str
    document_confidence: Optional[float] = None
    turns: List[Turn]
    transcript_txt_path: str
    download_url: str

class PipelineResponse(BaseModel):
    job_name: str
    service: str
    document_confidence: Optional[float] = None
    transcript_txt_path: str
    download_url: str
    turns: List[ClassifiedTurn]
    summary: Dict[str, Any]
