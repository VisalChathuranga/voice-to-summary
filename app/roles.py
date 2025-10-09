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

def refine_dialogue_with_llm(classified_turns: List[ClassifiedTurn]) -> List[ClassifiedTurn]:
    """
    Cleans each turn's text individually with LLM to fix errors/punctuation,
    preserving exact order, number, and roles. No risk of turn count mismatch.
    """
    if not settings.openai_api_key or not classified_turns:
        return classified_turns  # No-op if no key or empty

    system_prompt = """You are a medical transcription expert. Given a single speaker turn from a medical conversation, clean and correct ONLY this text:

1. Fix obvious transcription errors (e.g., 'sleep' → 'slipped', 'p' → 'foot' or 'pace', 'buckled' for knee injury, 'aging' → 'aching', 'injured occur' → 'injury occurred').
2. Make the sentence(s) complete and natural; keep filler words like 'uh', 'um'.
3. Add punctuation (periods, commas, question marks) for readability.
4. Do not add, remove, or change facts/content—keep it concise and true to the original.
5. Output ONLY the cleaned text (no [Role], no explanations)."""

    refined_turns = classified_turns.copy()  # Start with originals
    batch_size = 5  # Process in small batches to avoid token limits/costs

    for i in range(0, len(refined_turns), batch_size):
        batch = refined_turns[i:i + batch_size]
        batch_texts = [t.text.strip() for t in batch if t.text and t.text.strip()]
        if not batch_texts:
            continue

        # Prompt for batch (include role context if helpful)
        user_prompt = f"Clean these {len(batch)} turns (roles: {[t.display_name for t in batch]}):\n\n" + "\n\n".join(batch_texts)

        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1000  # Plenty for small batch
            )

            cleaned_batch = response.choices[0].message.content.strip().split("\n\n")  # Assume LLM separates by double newline
            cleaned_batch = [t.strip() for t in cleaned_batch if t.strip()]

            # Update turns (match by index; trim if lengths differ)
            for j, cleaned_text in enumerate(cleaned_batch[:len(batch)]):
                orig_idx = i + j
                if orig_idx < len(refined_turns) and cleaned_text:
                    refined_turns[orig_idx].text = cleaned_text

        except Exception as e:
            print(f"[warn] LLM batch refinement failed at index {i}: {e}; skipping batch")
            continue  # Keep originals for this batch

    return refined_turns