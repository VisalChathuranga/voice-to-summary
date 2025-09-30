from typing import Dict, List
from openai import OpenAI
from .settings import settings
from .models import Turn, ClassifiedTurn, Role

client = OpenAI(api_key=settings.openai_api_key)

ROLE_NAMES = {"doctor": "Doctor", "patient": "Patient", "nurse": "Nurse", "other": "Other"}

def _build_role_prompt(turns: List[Turn]) -> str:
    by_speaker = {}
    for t in turns:
        by_speaker.setdefault(t.speaker, []).append((t.text or "").strip())
    excerpts = {spk: " ".join([s for s in v if s])[:2000] for spk, v in by_speaker.items()}
    bullets = "\n".join([f'- "{spk}": "{excerpts[spk].replace(chr(10), " ")}"' for spk in excerpts])
    return f"""
You are labeling speakers in a medical conversation.
Speakers are named like "Speaker 1", "Speaker 2", etc.
Assign exactly one role per speaker from: ["doctor","patient","nurse","other"].

Return ONLY JSON:
{{
  "mapping": {{
    "Speaker 1": "doctor|patient|nurse|other",
    "Speaker 2": "doctor|patient|nurse|other",
    ...
  }}
}}

Guidelines:
- "doctor": clinician assessing, ordering tests, counseling.
- "patient": symptoms/experience questions/answers.
- "nurse": triage/vitals/logistics.
- "other": family/admin/interpreter.

Speakers & excerpts:
{bullets}
""".strip()

def _heuristic_mapping(turns: List[Turn]) -> Dict[str, Role]:
    unique = []
    for t in turns:
        if t.speaker not in unique:
            unique.append(t.speaker)
    mapping: Dict[str, Role] = {}
    for i, spk in enumerate(unique):
        mapping[spk] = "doctor" if i == 0 else ("patient" if i == 1 else "other")
    for spk in unique:
        joined = " ".join([tt.text or "" for tt in turns if tt.speaker == spk]).lower()
        if any(k in joined for k in ["i'm feeling", "i feel", "my ", "dizzy", "fever", "since ", "for the past"]):
            mapping[spk] = "patient"
        if any(k in joined for k in ["i'll check", "we'll run", "let me examine", "bp", "tests", "rule out"]):
            mapping[spk] = "doctor"
    return mapping

def classify_roles(turns: List[Turn]) -> Dict[str, Role]:
    if not settings.openai_api_key:
        return _heuristic_mapping(turns)
    try:
        prompt = _build_role_prompt(turns)
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a careful, deterministic classifier."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        import json
        mapping_raw = json.loads(resp.choices[0].message.content).get("mapping", {})
        mapping: Dict[str, Role] = {}
        for spk, role in mapping_raw.items():
            r = str(role).lower().strip()
            mapping[spk] = r if r in {"doctor", "patient", "nurse", "other"} else "other"
        if mapping and all(v == "other" for v in mapping.values()):
            return _heuristic_mapping(turns)
        return mapping or _heuristic_mapping(turns)
    except Exception:
        return _heuristic_mapping(turns)

def relabel_turns(turns: List[Turn], mapping: Dict[str, Role]) -> List[ClassifiedTurn]:
    out: List[ClassifiedTurn] = []
    for t in turns:
        role = mapping.get(t.speaker, "other")
        out.append(ClassifiedTurn(
            speaker=t.speaker,
            text=t.text,
            words=t.words,
            role=role,
            display_name=ROLE_NAMES[role],
        ))
    return out
