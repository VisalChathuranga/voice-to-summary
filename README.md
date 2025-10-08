# MedScribe Pro — Voice → Roles → Summary (FastAPI + AWS Transcribe)

Convert an uploaded **.mp3 / .webm** into a **role-labeled conversation** (Doctor / Patient / Nurse / Other) and return a **concise clinical summary**.  
The server also writes a single, human-friendly transcript file and exposes a **download URL**.

---

## ✨ Features

- Upload **.mp3 / .webm** → **AWS Transcribe (Medical or Standard)**
- **Speaker role classification** (Doctor / Patient / Nurse / Other)
- **Clinical summary** via your existing prompt
- **Single transcript file per job**, e.g.  
  `recording_20250928_082731_a1b2c3_conversation.txt`
- **Optimized for speed:** S3 Transfer Acceleration, multipart uploads, short polling, concurrent requests
- **Clean JSON** response:  
  ```json
  { "summary_text": "...", "download_url": "/api/download/..." }
  ```

---

## 📁 Project Structure

```
app/
├── __init__.py
├── static/
│   └── index.html
├── fastapi_clinical_summary.py
├── main.py
├── models.py
├── pipeline.py
├── roles.py
├── settings.py
└── transcriber.py
local_audio/
transcripts/
.env
environment.yml
README.md
requirements.txt
```

> **Tip (Windows)**: Always run commands from the **project root** (the folder that contains `app/`). This avoids `ModuleNotFoundError: No module named 'app'`.

---

## 🧰 Prerequisites

- **Conda** (or mamba)
- **Python 3.10+**
- **FFmpeg** on PATH (required by pydub)  
  - Windows: `winget install Gyan.FFmpeg` or `choco install ffmpeg`  
  - macOS: `brew install ffmpeg`  
  - Linux: `sudo apt-get install ffmpeg`
- **AWS credentials** with permissions for:
  - **S3**: create bucket (if needed), put/get objects
  - **Transcribe** (and **Transcribe Medical** if `USE_MEDICAL=true`)
- (Optional) **OpenAI API key** for role classification + summary

---

## 🚀 Setup

### 1) Create & activate environment
```bash
# from project root (folder that contains app/)
conda env create -f environment.yml
conda activate med-pipeline
```

### 2) Create `.env` in the project root

```properties
# ---- AWS / Transcribe (Project 1) ----
AWS_ACCESS_KEY_ID=YOUR_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET

# ---- OpenAI (Role classification + summary) ----
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# ---- Core ----
REGION=us-east-1
BUCKET=my-voice2text-us-east1

# Pick transcribe flavor
USE_MEDICAL=true        # true=Transcribe Medical, false=Standard Transcribe
SPECIALTY=primarycare   # primarycare|cardiology|neurology|oncology|radiology|urology
LANGUAGE=en-US          # Medical supports en-US

# ---- S3 transfer tuning (fast) ----
S3_ACCELERATE=true
S3_ENABLE_ACCELERATE_IF_NEEDED=true
S3_MAX_CONCURRENCY=16
S3_MULTIPART_THRESHOLD_MB=8
S3_MULTIPART_CHUNKSIZE_MB=8

# ---- Audio re-encode ----
FORCE_REENCODE=true     # set false to skip re-encode if already 16k mono (faster)
TARGET_SAMPLE_RATE=16000
TARGET_CHANNELS=1
TARGET_BITRATE=64k

# ---- Output dirs ----
TRANSCRIPTS_DIR=transcripts
```

The app will create `local_audio/` and `transcripts/` automatically.

---

## 🏃‍♂️ Run the server

> **Important:** Start from the project root so `app.main:app` is importable.

### Dev (auto-reload)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Prod-ish (multi-workers for concurrency)
```bash
# 4 workers = 4 processes; each can run multiple requests concurrently
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🧪 Test with cURL

### Example 1 (your path with spaces, Git Bash quoting)
```bash
curl -X POST "http://localhost:8000/api/transcribe-and-summarize"   -H "Accept: application/json"   -F "file=@/D:/MedCube/Projects/1st Month/aws-voice-to-text/audio/recording_2025-09-28T08-27-31-220Z.mp3"
```

### Example 2 (another file)
```bash
curl -X POST "http://localhost:8000/api/transcribe-and-summarize"   -H "Accept: application/json"   -F "file=@/D:/MedCube/Projects/1st Month/Voice to Summery/final.mp3"
```

**Response**
```json
{
  "summary_text": "A patient presents with ...",
  "download_url": "/api/download/recording_20250928_082731_a1b2c3_conversation.txt"
}
```

Download the transcript:
```
GET http://localhost:8000/api/download/<the_file_name_returned>.txt
```

---

## 🔌 API

### `POST /api/transcribe-and-summarize`
- **Body (multipart/form-data):** `file` = `.mp3` or `.webm`
- **Returns:**
  ```json
  { "summary_text": "…", "download_url": "/api/download/<file>.txt" }
  ```

### `GET /api/download/{name}`
- Downloads the transcript file by name (served from `TRANSCRIPTS_DIR`).

---

## ⚡ Concurrency & Performance

This project is designed to **handle multiple client requests at the same time** without changing your endpoints.

- **Uvicorn workers:** Use `--workers 4` (or higher on bigger machines). Each worker process runs requests concurrently (async + threadpool for CPU/FFmpeg).
- **S3 speedups:** We enable **Transfer Acceleration** and multipart uploads. Keep `S3_ACCELERATE=true` and the concurrency/chunk env vars shown above.
- **Skip re-encode if possible:** Set `FORCE_REENCODE=false` if your inputs are already close to 16 kHz mono—saves time.
- **Transcribe Medical vs Standard:** **Medical** is more accurate for clinical conversations but can be slower. If speed is critical and clinical terms are light, set `USE_MEDICAL=false`.
- **Short polling:** The server polls Transcribe with a short interval to detect completion quickly.
- **Windows path note:** For Git Bash/PowerShell, quote paths with spaces exactly as in the examples.

---

## 🖥️ Helpful one-liners (Windows PowerShell)

Run with workers from the correct folder:
```powershell
cd "D:\MedCube\Projectsst Monthws-voice-to-text"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

If you ever see `ModuleNotFoundError: No module named 'app'`, it means you launched outside the project root. Use the `cd` above, or:
```powershell
uvicorn app.main:app --app-dir "D:\MedCube\Projectsst Monthws-voice-to-text" --workers 4
```

---

## 🧯 Troubleshooting

- **It’s slow**
  - Most time is in **AWS Transcribe** compute. Use multiple **workers**, enable **S3 acceleration**, and consider `USE_MEDICAL=false` for speed.
- **S3 acceleration not enabled**
  - Check logs; you should see lines like  
    *“S3 Transfer Acceleration already enabled for bucket …”*
- **FFmpeg not found**
  - Install and ensure it’s on PATH. Try: `ffmpeg -version`.
- **CORS**
  - If calling from a browser on a different origin, enable CORS in `app.main` (already supported).  
- **AWS permissions**
  - Ensure the IAM user/role has **S3** (create bucket, put/get) + **Transcribe** (and **Transcribe Medical** if enabled).
- **Paths with spaces**
  - Use quotes exactly as shown in cURL examples.

---

## 📄 License

MIT (or your preferred license)

---

### Quick Start (all in one block you can paste)

```bash
conda env create -f environment.yml
conda activate med-pipeline

# from project root:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# or for concurrency:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# test (Git Bash):
curl -X POST "http://localhost:8000/api/transcribe-and-summarize"   -H "Accept: application/json"   -F "file=@/D:/MedCube/Projects/1st Month/aws-voice-to-text/audio/recording_2025-09-28T08-27-31-220Z.mp3"
```
