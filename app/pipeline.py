from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid, re
from datetime import datetime

from .settings import settings
from .models import TranscribeResult, Turn, PipelineResponse, ClassifiedTurn
from .roles import classify_roles, relabel_turns
from .transcriber import transcribe_uploaded
from .fastapi_clinical_summary import generate_summary_chat

def _slugify(name: str) -> str:
    base = Path(name).stem
    base = base.strip().lower()
    base = re.sub(r"[^a-z0-9]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    if not base:
        base = "conversation"
    return base[:20]

def _friendly_job_name(original_filename: Optional[str]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{_slugify(original_filename or 'conversation')}_{ts}_{uuid.uuid4().hex[:6]}"

def _turns_to_text(turns: List[ClassifiedTurn]) -> str:
    return "\n".join(f"[{t.display_name}] {t.text}".strip() for t in turns)

def _save_txt(content: str, friendly_job: str) -> str:
    out_name = f"{friendly_job}_conversation.txt"
    out_path = Path(settings.transcripts_dir) / out_name
    out_path.write_text(content, encoding="utf-8")
    return str(out_path)

def _as_result(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise TypeError("Unexpected return from transcribe_uploaded()")
    obj.setdefault("transcript_txt_path", "")
    obj.setdefault("download_url", "")
    return obj

def transcribe_classify_summarize(upload_path: str, original_filename: Optional[str] = None) -> PipelineResponse:
    # 1) Transcribe
    t: Dict[str, Any] = _as_result(transcribe_uploaded(upload_path))
    tr = TranscribeResult(**t)

    # 2) Base turns
    def _words_to_text(words):
        if isinstance(words, list):
            parts = []
            for w in words:
                token = ""
                if isinstance(w, dict):
                    token = (w.get("word") or w.get("text") or "").strip()
                else:
                    token = (getattr(w, "word", None) or getattr(w, "text", None) or "").strip()
                if token:
                    parts.append(token)
            return " ".join(parts)
        return str(words or "")

    base_turns: List[Turn] = []
    for r in tr.turns:
        text = r.text or _words_to_text(r.words)
        base_turns.append(Turn(speaker=r.speaker, text=text, words=r.words))

    # 3) Role classification
    mapping = classify_roles(base_turns)
    classified: List[ClassifiedTurn] = relabel_turns(base_turns, mapping)

    # 4) Save final .txt
    friendly_job = _friendly_job_name(original_filename)
    rendered = _turns_to_text(classified)
    if tr.document_confidence is not None:
        rendered = f"Document confidence: {tr.document_confidence:.4f} ({tr.document_confidence*100:.1f}%)\n\n" + rendered
    conversation_txt_path = _save_txt(rendered, friendly_job)

    # 5) Summary (OpenAI)
    context = f"MEDICAL CASE DATA:\n\n=== HPI / TRANSCRIPT ===\n{rendered}\n"
    summary_text = generate_summary_chat(context)
    summary_json = {"summary_text": summary_text}

    return PipelineResponse(
        job_name=tr.job_name,
        service=tr.service,
        document_confidence=tr.document_confidence,
        transcript_txt_path=conversation_txt_path,
        download_url=f"/api/download/{Path(conversation_txt_path).name}",
        turns=classified,
        summary=summary_json,
    )
