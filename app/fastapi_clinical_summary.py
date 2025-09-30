import os
import re
import time
from typing import Dict, Optional

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

# -----------------------------------------------------------------------------
# ENV & OPENAI SETUP
# -----------------------------------------------------------------------------

def get_api_key() -> Optional[str]:
    """Load env and return the OpenAI API key."""
    _ = load_dotenv(find_dotenv(), override=False)
    key = os.getenv("OPENAI_API_KEY")
    return key

API_KEY = get_api_key()
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment.")

client = OpenAI(api_key=API_KEY)

# Optional: Assistants mode requires an Assistant ID to reuse the assistant
# across requests. We will create NEW *threads* per request and delete them
# after finishing. Set ASSISTANT_ID in your environment if you want to use
# mode="assistants".
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

# -----------------------------------------------------------------------------
# PARSING & SUMMARIZATION HELPERS
# -----------------------------------------------------------------------------

def parse_structured_input(user_input: str) -> Dict[str, Dict[str, str]]:
    """Parse the structured input format into sections and Q&A pairs."""
    sections: Dict[str, Dict[str, str]] = {}
    current_section = None
    current_content: Dict[str, str] = {}

    lines = user_input.split('\n')
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        section_match = re.match(r'^(\d+\.\s+.*|.*History.*:|.*HPI.*:.*)$', line, re.IGNORECASE)
        if section_match:
            if current_section and current_content:
                sections[current_section] = current_content.copy()
                current_content = {}
            current_section = line.strip()
        else:
            qa_match = re.match(r'^([^:]+):\s*(.+)$', line)
            if qa_match and current_section:
                question = qa_match.group(1).strip()
                answer = qa_match.group(2).strip()
                current_content[question] = answer
            elif current_section and current_content:
                last_question = list(current_content.keys())[-1]
                current_content[last_question] += " " + line

    if current_section and current_content:
        sections[current_section] = current_content

    return sections


def build_context(sections: Dict[str, Dict[str, str]]) -> str:
    if not sections:
        return ""
    context = "MEDICAL CASE DATA:\n\n"
    for section_name, qa_pairs in sections.items():
        context += f"=== {section_name.upper()} ===\n"
        for question, answer in qa_pairs.items():
            context += f"• {question}: {answer}\n"
        context += "\n"
    return context


BASE_TASK_INSTRUCTIONS = (
    "You are a medical consultant creating concise, professional clinical summaries in plain text for doctors. "
    "Treat every request as independent and stateless; do not use or reference prior inputs, outputs, or memories."

    "OUTPUT FORMAT:"
    "• Return exactly one cohesive paragraph in plain text."
    "• No headings, lists, bullets, or labeled sections (do not use 'Problem:', 'History:', 'Symptoms:', 'Social:', or 'Findings:')."
    "• Target length ~60–130 words (may exceed slightly only to include critical safety information)."

    "CONTENT TO COVER (prioritized):"
    "• Chief complaint and onset/mechanism with clear chronology."
    "• Key symptoms with severity, location/radiation, and functional impact."
    "• Pertinent past medical history (and medications/allergies only if clinically relevant)."
    "• Pertinent social factors (e.g., smoking, occupation) that affect risk or management."
    "• Critical exam or imaging findings and the most relevant working impression if implied by the data."
    "• Current/initial treatment and practical next steps/plan."

    "STYLE & SAFETY:"
    "• Use precise medical terminology without unnecessary jargon; third-person, objective tone."
    "• Do not invent or infer facts not present in the input; omit unspecified details."
    "• Exclude normal/negative findings unless they materially change decisions."
    "• Preserve units, dates, and timeframes as given."
)


EXAMPLE_SNIPPET = ""


def clean_plain_text(text: str) -> str:
    # Remove common markdown artifacts if the model returns them
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    return text.strip()


# --- Chat Completions (stateless) -------------------------------------------

def generate_summary_chat(context: str) -> str:
    prompt = (
        "You are a skilled medical professional creating a concise clinical summary for doctors.\n\n"
        f"MEDICAL DATA:\n{context}\n\n"
        + BASE_TASK_INSTRUCTIONS
        + EXAMPLE_SNIPPET
        + "Ensure the summary is concise, doctor-friendly, and highlights critical details."
    )

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),  # Use a known supported model as default
        messages=[
            {"role": "system", "content": (
                "You are a medical consultant creating concise summaries. "
                "Treat every request as independent and stateless. "
                "Do not rely on prior runs or any memory from earlier inputs." )},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.1  # Keep deterministic for medical summaries
    )
    out = resp.choices[0].message.content or ""
    return clean_plain_text(out)


# --- Assistants API (new thread per request + cleanup) ----------------------

def generate_summary_assistants(context: str, *, timeout_sec: float = 60.0) -> str:
    if not ASSISTANT_ID:
        raise RuntimeError("ASSISTANT_ID env var is required for assistants mode.")

    user_prompt = (
        f"MEDICAL DATA:\n{context}\n\n"
        + BASE_TASK_INSTRUCTIONS
        + EXAMPLE_SNIPPET
        + "Ensure the summary is concise, doctor-friendly, and highlights critical details."
    )

    # Create a NEW thread per request
    thread = client.beta.threads.create()
    try:
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_prompt,
        )
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
        )

        # Poll the run until completed or timed out
        start = time.time()
        status = run.status
        while status in {"queued", "in_progress", "cancelling"}:
            if time.time() - start > timeout_sec:
                raise TimeoutError("Assistants run timed out")
            time.sleep(0.5)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            status = run.status

        if status != "completed":
            raise RuntimeError(f"Assistants run failed with status: {status}")

        # Fetch the latest assistant message from the thread
        msgs = client.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=10)
        for m in msgs.data:
            if m.role == "assistant":
                # Messages can have multiple content parts; pick text parts
                parts = []
                for c in m.content:
                    if getattr(c, "type", None) == "text":
                        parts.append(c.text.value)
                if parts:
                    return clean_plain_text("\n".join(parts))
        raise RuntimeError("No assistant message found in thread")

    finally:
        # Always clean up the thread to avoid carrying state
        try:
            client.beta.threads.delete(thread.id)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------------------------------

class SummarizeRequest(BaseModel):
    user_input: str = Field(..., description="Structured medical Q&A text.")
    mode: str = Field("chat", description="'chat' (default) or 'assistants'.")

class SummarizeResponse(BaseModel):
    summary: str
    characters: int
    lines: int
    mode: str

app = FastAPI(title="Clinical Summary API", version="1.0.0")


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    if not req.user_input or not req.user_input.strip():
        raise HTTPException(status_code=400, detail="user_input is required")

    sections = parse_structured_input(req.user_input)
    if not sections:
        raise HTTPException(status_code=400, detail="Could not parse the input data")

    context = build_context(sections)

    try:
        if req.mode == "assistants":
            summary = generate_summary_assistants(context)
        else:
            summary = generate_summary_chat(context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SummarizeResponse(
        summary=summary,
        characters=len(summary),
        lines=len(summary.splitlines()),
        mode=req.mode,
    )


# Convenience root endpoint
@app.get("/")
def root():
    return {
        "name": "Clinical Summary API",
        "version": "1.0.0",
        "endpoints": {
            "POST /summarize": {
                "body": {
                    "user_input": "...structured Q&A...",
                    "mode": "chat | assistants"
                }
            }
        }
    }


# To run: uvicorn fastapi_clinical_summary:app --host 0.0.0.0 --port 8000
