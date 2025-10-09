# settings.py
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

def _get_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}

def _get_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default

class Settings(BaseModel):
    # ---- AWS / S3 ----
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    region: str = os.getenv("AWS_REGION") or os.getenv("REGION", "us-east-1")
    bucket: str = os.getenv("S3_BUCKET") or os.getenv("BUCKET", "")

    # ---- Transcribe options ----
    use_medical: bool = _get_bool("TRANSCRIBE_IS_MEDICAL", _get_bool("USE_MEDICAL", False))
    language: str = os.getenv("TRANSCRIBE_LANGUAGE_CODE") or os.getenv("LANGUAGE", "en-US")
    specialty: str = os.getenv("SPECIALTY", "primarycare")
    channel_identification: bool = _get_bool("CHANNEL_IDENTIFICATION", False)
    max_speakers: int = _get_int("MAX_SPEAKERS", 4)

    # ---- S3 transfer tuning ----
    s3_accelerate: bool = _get_bool("S3_ACCELERATE", True)
    s3_enable_accelerate_if_needed: bool = _get_bool("S3_ENABLE_ACCELERATE_IF_NEEDED", True)
    s3_max_concurrency: int = _get_int("S3_MAX_CONCURRENCY", 16)
    s3_multipart_threshold_mb: int = _get_int("S3_MULTIPART_THRESHOLD_MB", 8)
    s3_multipart_chunksize_mb: int = _get_int("S3_MULTIPART_CHUNKSIZE_MB", 8)

    # ---- Re-encode controls ----
    force_reencode: bool = _get_bool("FORCE_REENCODE", True)
    target_sample_rate: int = _get_int("TARGET_SAMPLE_RATE", 16000)
    target_channels: int = _get_int("TARGET_CHANNELS", 1)
    target_bitrate: str = os.getenv("TARGET_BITRATE", "64k")
    local_audio_dir: str = os.getenv("LOCAL_AUDIO_DIR", "local_audio")

    # ---- OpenAI ----
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ---- Output dirs ----
    transcripts_dir: str = os.getenv("TRANSCRIPTS_DIR", "transcripts")
    output_dir: str = Field(default_factory=lambda: os.getenv("OUTPUT_DIR") or os.getenv("TRANSCRIPTS_DIR", "transcripts"))

    # Keep raw JSON?
    keep_raw_files: bool = _get_bool("KEEP_RAW_FILES", False)

    # ---- NEW: LLM role refiner ----
    role_refiner_enabled: bool = _get_bool("ROLE_REFINER_ENABLED", True)

settings = Settings()

# Ensure dirs exist
os.makedirs(settings.transcripts_dir, exist_ok=True)
os.makedirs(settings.local_audio_dir, exist_ok=True)
if settings.output_dir:
    os.makedirs(settings.output_dir, exist_ok=True)
