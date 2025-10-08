from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pathlib import Path
import tempfile, shutil, os

from .pipeline import transcribe_classify_summarize
from .settings import settings

app = FastAPI(title="Voice → Roles → Summary")

# CORS (loosen for local dev; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def home():
    idx = Path(__file__).with_name("index.html")
    if idx.exists():
        return idx.read_text(encoding="utf-8")
    return "<html><body><h3>POST /api/transcribe-and-summarize</h3></body></html>"

@app.post("/api/transcribe-and-summarize")
async def api_transcribe_and_summarize(file: UploadFile = File(...)):
    """
    Same endpoint/contract, but all heavy work runs in a background thread.
    This keeps the event loop free so many clients can be served concurrently.
    """
    try:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".mp3", ".webm", ".wav", ".m4a"}:
            raise HTTPException(status_code=400, detail="Only .mp3/.webm/.wav/.m4a accepted")

        # Save upload to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # ⬇️ Run your whole pipeline off the event loop
        result = await run_in_threadpool(
            transcribe_classify_summarize, tmp_path, file.filename
        )

        # Best-effort temp cleanup
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return {
            "summary_text": result.summary.get("summary_text", ""),
            "download_url": result.download_url,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{name}")
def download_txt(name: str):
    p = Path(settings.transcripts_dir) / name
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(p), media_type="text/plain", filename=name)

static_dir = Path(__file__).with_name("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
