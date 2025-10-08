import os, re, uuid, json, time, mimetypes
from typing import Dict, List, Tuple, Optional

import boto3
import requests
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig, S3Transfer
from pydub import AudioSegment

from .settings import settings

AUDIO_EXTS = {".mp3", ".mp4", ".m4a", ".wav", ".flac", ".ogg", ".amr", ".webm", ".wma"}

MEDICAL_SPECIALTIES = {
    "primarycare": "PRIMARYCARE",
    "cardiology": "CARDIOLOGY",
    "neurology": "NEUROLOGY",
    "oncology": "ONCOLOGY",
    "radiology": "RADIOLOGY",
    "urology": "UROLOGY",
}

# ---------- AWS clients (tuned) ----------

def aws_clients():
    # Bigger HTTP pool + accelerated endpoint for S3
    s3_cfg = {"use_accelerate_endpoint": settings.s3_accelerate}
    botocfg = BotoConfig(
        s3=s3_cfg,
        retries={"max_attempts": 5, "mode": "standard"},
        max_pool_connections=max(32, settings.s3_max_concurrency * 2),
        connect_timeout=10,
        read_timeout=300,
    )
    session = boto3.Session(region_name=settings.region)
    s3 = session.client("s3", config=botocfg)
    transcribe = session.client("transcribe", config=BotoConfig(
        retries={"max_attempts": 5, "mode": "standard"},
        max_pool_connections=32,
        read_timeout=300,
        connect_timeout=10,
    ))
    return s3, transcribe

# ---------- Helpers ----------

def sanitize_job_name(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]", "_", name)

def ensure_bucket(s3, bucket: str, region: str):
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
        s3.get_waiter("bucket_exists").wait(Bucket=bucket)

def ensure_acceleration(s3, bucket: str):
    if not settings.s3_enable_accelerate_if_needed:
        return
    try:
        conf = s3.get_bucket_accelerate_configuration(Bucket=bucket)
        status = conf.get("Status")
        if status != "Enabled":
            s3.put_bucket_accelerate_configuration(
                Bucket=bucket,
                AccelerateConfiguration={"Status": "Enabled"}
            )
            print(f"[info] Enabled S3 Transfer Acceleration for bucket: {bucket}")
    except ClientError as e:
        print(f"[warn] Could not enable acceleration: {e}")

def _transfer_config() -> TransferConfig:
    return TransferConfig(
        multipart_threshold=settings.s3_multipart_threshold_mb * 1024 * 1024,
        multipart_chunksize=settings.s3_multipart_chunksize_mb * 1024 * 1024,
        max_concurrency=settings.s3_max_concurrency,
        use_threads=True,
    )

def upload_file_to_s3(s3, local_path: str, bucket: str, key: str):
    transfer = S3Transfer(client=s3, config=_transfer_config())
    content_type, _ = mimetypes.guess_type(local_path)
    extra = {"ContentType": content_type or "application/octet-stream"}
    transfer.upload_file(local_path, bucket, key, extra_args=extra)
    return f"s3://{bucket}/{key}"

# ---------- Audio re-encode (fast & small) ----------

def _is_small_mp3(path: str) -> bool:
    if not path.lower().endswith(".mp3"):
        return False
    try:
        size = os.path.getsize(path)
        return size < 1.5 * 1024 * 1024  # ~1.5MB
    except Exception:
        return False

def to_mp3(src_path: str, dest_dir: str) -> str:
    base = os.path.splitext(os.path.basename(src_path))[0]
    mp3_path = os.path.join(dest_dir, f"{base}.mp3")
    os.makedirs(dest_dir, exist_ok=True)

    if (not settings.force_reencode) and _is_small_mp3(src_path):
        print("[info] Skipping re-encode: already small MP3")
        return src_path

    audio = AudioSegment.from_file(src_path)
    audio = audio.set_channels(settings.target_channels).set_frame_rate(settings.target_sample_rate)
    audio.export(mp3_path, format="mp3", bitrate=settings.target_bitrate)
    print("[info] Re-encoded to optimized MP3")
    return mp3_path

# ---------- Transcribe ----------

def start_job(transcribe, media_s3_uri: str, safe_base: str) -> Tuple[str, str]:
    if settings.use_medical:
        job_name = f"vt_med_{safe_base}_{uuid.uuid4().hex[:8]}"
        args = {
            "MedicalTranscriptionJobName": job_name,
            "LanguageCode": "en-US",
            "Media": {"MediaFileUri": media_s3_uri},
            "OutputBucketName": settings.bucket,
            "OutputKey": f"transcripts/{safe_base}/medical/",
            "Settings": {
                "ShowSpeakerLabels": not settings.channel_identification,
                "ChannelIdentification": settings.channel_identification,
                "MaxSpeakerLabels": max(2, settings.max_speakers),
                "ShowAlternatives": True,
                "MaxAlternatives": 2,
            },
            "Specialty": {
                "primarycare": "PRIMARYCARE",
                "cardiology": "CARDIOLOGY",
                "neurology": "NEUROLOGY",
                "oncology": "ONCOLOGY",
                "radiology": "RADIOLOGY",
                "urology": "UROLOGY",
            }.get(settings.specialty, "PRIMARYCARE"),
            "Type": "CONVERSATION",
        }
        transcribe.start_medical_transcription_job(**args)
        return job_name, "medical"
    else:
        job_name = f"vt_std_{safe_base}_{uuid.uuid4().hex[:8]}"
        args = {
            "TranscriptionJobName": job_name,
            "Media": {"MediaFileUri": media_s3_uri},
            "Settings": {
                "ShowSpeakerLabels": not settings.channel_identification,
                "ChannelIdentification": settings.channel_identification,
                "MaxSpeakerLabels": max(2, settings.max_speakers),
            },
        }
        if settings.language and settings.language.lower() != "auto":
            args["LanguageCode"] = settings.language
        else:
            args["IdentifyLanguage"] = True
        transcribe.start_transcription_job(**args)
        return job_name, "transcribe"

def wait_for_job(transcribe, job_name: str, service: str, poll_sec: int = 3, timeout_min: int = 120) -> dict:
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        if service == "medical":
            resp = transcribe.get_medical_transcription_job(MedicalTranscriptionJobName=job_name)
            job = resp["MedicalTranscriptionJob"]
        else:
            resp = transcribe.get_transcription_job(TranscriptionJobName=job_name)
            job = resp["TranscriptionJob"]
        status = job["TranscriptionJobStatus"]
        if status in ("COMPLETED", "FAILED"):
            return job
        time.sleep(poll_sec)
    raise TimeoutError(f"Job timed out: {job_name}")

# ---------- Download transcript ----------

def download_transcript(s3, url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if url.startswith("s3://") or "amazonaws.com" in url:
        if url.startswith("s3://"):
            bucket, key = url[5:].split("/", 1)
        else:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            parts = parsed.path.lstrip("/").split("/", 1)
            bucket, key = parts[0], parts[1]
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
    else:
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        body = r.content
    with open(dest_path, "wb") as f:
        f.write(body)

# ---------- Formatting ----------

def build_speaker_map(speaker_labels: dict) -> Dict[str, str]:
    ts_to_spk: Dict[str, str] = {}
    for seg in speaker_labels.get("segments", []):
        spk = seg.get("speaker_label")
        for item in seg.get("items", []):
            start = item.get("start_time")
            if start:
                ts_to_spk[start] = spk
    return ts_to_spk

def doc_confidence(results: dict) -> float:
    items = results.get("items", [])
    vals: List[float] = []
    for it in items:
        if it.get("type") == "pronunciation":
            alt = it.get("alternatives", [{}])[0]
            c = alt.get("confidence")
            if c is not None:
                try:
                    vals.append(float(c))
                except (TypeError, ValueError):
                    pass
    return sum(vals)/len(vals) if vals else 1.0

def pretty_turns(results: dict) -> List[dict]:
    items = results.get("items", [])
    speaker_labels = results.get("speaker_labels")
    transcripts = results.get("transcripts", [])

    if not speaker_labels:
        text = transcripts[0]["transcript"] if transcripts else ""
        return [{"speaker": "Speaker 1", "words": [{"text": w} for w in text.split()], "text": text}]

    ts_to_spk = build_speaker_map(speaker_labels)
    spk_index: Dict[str, int] = {}

    def spk_name(lbl: Optional[str]) -> Optional[str]:
        if lbl is None:
            return None
        if lbl not in spk_index:
            spk_index[lbl] = len(spk_index) + 1
        return f"Speaker {spk_index[lbl]}"

    turns: List[dict] = []
    current_speaker: Optional[str] = None
    current_words: List[dict] = []

    def flush():
        nonlocal current_speaker, current_words
        if current_speaker and current_words:
            text = " ".join(w["text"] for w in current_words)
            turns.append({"speaker": current_speaker, "words": current_words, "text": text})
            current_words = []

    for it in items:
        typ = it.get("type")
        if typ == "pronunciation":
            alt = it["alternatives"][0]
            word = alt["content"]
            start_time = it.get("start_time")
            spk = spk_name(ts_to_spk.get(start_time)) or current_speaker or "Speaker 1"
            if spk != current_speaker:
                flush()
                current_speaker = spk
            current_words.append({"text": word})
        elif typ == "punctuation":
            punct = it["alternatives"][0]["content"]
            if current_words:
                current_words[-1]["text"] = current_words[-1]["text"] + punct
            else:
                current_words.append({"text": punct})
    flush()
    return turns

# ---------- Orchestration ----------

def transcribe_uploaded(local_path: str) -> Dict:
    s3, transcribe = aws_clients()
    ensure_bucket(s3, settings.bucket, settings.region)
    if settings.s3_accelerate:
        ensure_acceleration(s3, settings.bucket)

    # 1) Convert (or skip) to small MP3
    mp3_path = to_mp3(local_path, settings.local_audio_dir)

    # 2) Upload (accelerated + multipart + threads)
    base = os.path.basename(mp3_path)
    safe_base = sanitize_job_name(os.path.splitext(base)[0])
    key = f"input/{base}"
    media_uri = upload_file_to_s3(s3, mp3_path, settings.bucket, key)

    # 3) Transcribe
    job_name, service = start_job(transcribe, media_uri, safe_base)
    job = wait_for_job(transcribe, job_name, service, poll_sec=3)
    if job.get("TranscriptionJobStatus") == "FAILED":
        raise RuntimeError(job.get("FailureReason", "Unknown failure"))

    transcript_url = job["Transcript"]["TranscriptFileUri"]

    # 4) Download JSON locally (only to parse; remove unless keeping raw)
    json_path = os.path.join(settings.output_dir, f"{safe_base}_{service}.json")
    download_transcript(s3, transcript_url, json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", {})
    turns = pretty_turns(results)
    conf = doc_confidence(results)

    if not settings.keep_raw_files:
        try:
            os.remove(json_path)
        except Exception:
            pass

    return {
        "job_name": job_name,
        "service": "medical" if settings.use_medical else "standard",
        "document_confidence": conf,
        "transcript_txt_path": "",
        "download_url": "",
        "turns": turns,
    }
