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
- "other": family/admin/interpreter/third party.

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

def refine_dialogue_with_llm(classified_turns: List[ClassifiedTurn]) -> List[ClassifiedTurn]:
    """
    Uses LLM with holistic prompt to review the entire conversation: fix labels (doctor/patient/other/third party),
    correct errors, reorder/split/merge for logical flow, clean text. Outputs new turns list for final .txt.
    """
    if not settings.openai_api_key or not classified_turns:
        return classified_turns  # No-op if no key or empty

    # Build initial formatted dialogue
    dialogue_input = "\n".join([f"[{t.display_name}] {t.text}" for t in classified_turns if t.text and t.text.strip()])

    system_prompt = """You are a medical professional reviewing a voice-to-text transcription of a doctor–patient conversation. The dialogue may contain errors such as mixed-up speaker labels, merged sentences, or misplaced responses due to transcription inaccuracies. Your task is to carefully read through the conversation and ensure that the dialogue is presented in the correct logical order, accurately distinguishing between the doctor and the patient. Make any necessary corrections to improve clarity and flow, but do not alter the original meaning or intent of the conversation.

Roles to use: [Doctor] for clinician, [Patient] for the person describing symptoms, [Other] for any third party (nurse, family, etc.).

1. Review the entire transcript for logical flow: questions from Doctor followed by Patient responses.
2. Fix speaker labels if misattributed (e.g., swap if a symptom description is labeled as Doctor).
3. Split merged turns or merge fragmented ones for natural dialogue.
4. Clean text: fix errors (e.g., 'sleep' → 'slipped', 'aging' → 'aching'), add punctuation, keep fillers ('uh', 'um').
5. Preserve all core content; do not add/remove facts.

Format output strictly as:
- Each turn on a new line: [Doctor|Patient|Other] Cleaned dialogue here.
- No extra text, explanations, or headers—just the conversation."""

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Raw transcript:\n\n{dialogue_input}"},
            ],
            temperature=0.1,
            max_tokens=2000  # Adjust for length
        )

        cleaned_output = response.choices[0].message.content.strip()

        # Parse into new turns
        refined_turns = []
        lines = cleaned_output.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("[") and "]" in line and line.endswith("?") or line.endswith(".") or " " in line.split("]", 1)[1]:
                # Valid turn
                role_part, text_part = line.split("]", 1)
                role_str = role_part[1:].strip()  # e.g., "Doctor"
                text = text_part.strip()
                if role_str in ROLE_NAMES.values() and text:
                    role_key = next((k for k, v in ROLE_NAMES.items() if v == role_str), "other")
                    refined_turns.append(ClassifiedTurn(
                        speaker="refined",  # Dummy
                        text=text,
                        words=None,  # Not preserved in holistic pass
                        role=role_key,
                        display_name=role_str,
                    ))

        if not refined_turns:
            print("[warn] No valid turns parsed from refinement; falling back")
            return classified_turns

        print(f"[info] Refined {len(classified_turns)} → {len(refined_turns)} turns")
        return refined_turns

    except Exception as e:
        print(f"[warn] LLM holistic refinement failed: {e}; using original")
        return classified_turns