"""
Explanation layer: sits on top of the structured model.
Default mode uses templated plain-language generation (fast, deterministic, auditable).
If HUGGINGFACE_API_KEY is set, an instruct model is called via HF's Inference Providers
router (OpenAI-compatible chat-completions API) to rewrite the same explanation more
naturally — strictly grounded in the model's own ranked output, never allowed to
introduce a new cause or confidence score.
"""
import os
import re
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("explainer")

HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
# Default model must be one actually served by an HF Inference Provider today.
# Not every model on the Hub is backed by a provider under the router API --
# gated/license-restricted models (e.g. Meta's Llama family) often aren't, which
# returns a 400 "not supported by any provider" error even with a valid token.
# openai/gpt-oss-120b is confirmed served by multiple providers as of HF's docs.
HF_MODEL = os.environ.get("HF_MODEL", "openai/gpt-oss-120b")
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

USE_LLM = bool(HF_API_KEY)


def build_explanation(fault_codes, vehicle_type: str, mileage_km: int,
                       vehicle_age_months: int, ranked_causes: list, notes: str = "",
                       low_confidence: bool = False, unknown_codes: list = None,
                       code_found_in: dict = None):
    if isinstance(fault_codes, str):
        fault_codes = [fault_codes]
    code_str = " + ".join(fault_codes)
    unknown_codes = unknown_codes or []
    code_found_in = code_found_in or {}

    top = ranked_causes[0]
    runner_up = ranked_causes[1] if len(ranked_causes) > 1 else None

    if unknown_codes:
        unknown_str = ", ".join(unknown_codes)
        plural = "s" if len(unknown_codes) > 1 else ""

        # If any of these codes is actually a real, known code -- just not for the
        # vehicle type selected -- say that explicitly. That's a much more useful
        # and more likely-correct message than a generic "code not recognized",
        # since it points at the probable actual mistake (wrong vehicle type
        # selected) rather than implying the code itself might be invalid.
        mismatch_notes = [
            f"{code} is a known code for {'/'.join(found_in)}, not {vehicle_type.split(' ')[-1].strip('()')}"
            for code, found_in in code_found_in.items()
        ]
        mismatch_sentence = (
            " " + "; ".join(mismatch_notes) + " — check the vehicle type selected."
            if mismatch_notes else ""
        )

        base = (
            f"Fault code{plural} {unknown_str} not in this system's reference set for "
            f"this vehicle type — there is no training data for {'this code' if len(unknown_codes)==1 else 'these codes'}."
            f"{mismatch_sentence} "
            f"The ranking below ('{top['cause']}', {top['confidence']*100:.0f}%) reflects the "
            "closest known pattern, NOT this specific code, and should not be treated as a "
            "code-specific finding. Look up the code definition directly and inspect accordingly."
        )
        if USE_LLM:
            return _hf_rewrite(base, notes)
        return base

    if low_confidence:
        base = (
            f"Low confidence match for fault code(s) {code_str} on this {vehicle_type} "
            f"({mileage_km:,} km, {vehicle_age_months} months old) — top candidate "
            f"'{top['cause']}' is only {top['confidence']*100:.0f}% confident. "
            "Pattern match is weak; defer to technician judgment and inspection rather "
            "than this ranking."
        )
        if USE_LLM:
            return _hf_rewrite(base, notes)
        return base

    base = (
        f"For fault code(s) {code_str} on this {vehicle_type} "
        f"({mileage_km:,} km, {vehicle_age_months} months old), the most likely cause is "
        f"'{top['cause']}' with {top['confidence']*100:.0f}% confidence."
    )
    if runner_up:
        base += (
            f" Next most likely is '{runner_up['cause']}' "
            f"({runner_up['confidence']*100:.0f}% confidence)."
        )
    if mileage_km > 30000 or vehicle_age_months > 36:
        base += " Higher mileage/age increases likelihood of wear-related causes."

    if USE_LLM:
        return _hf_rewrite(base, notes)
    return base


def _hf_rewrite(base_explanation: str, notes: str) -> str:
    notes = (notes or "").strip()

    if notes:
        notes_instruction = (
            f'Technician-reported symptom: "{notes}". '
            "Do not just acknowledge that this symptom exists -- say something concrete "
            "about it: does it fit the predicted cause or point somewhere else, and is "
            "there one added thing worth checking because of it? If the symptom doesn't "
            "fit this vehicle type or the predicted cause at all, say so plainly instead "
            "of working around it."
        )
    else:
        notes_instruction = "No technician notes were given -- don't invent any."

    prompt = (
        "Rewrite this vehicle repair diagnosis for a technician in 2-3 clear sentences. "
        "Copy every percentage number exactly as given, character for character -- "
        "do not round, adjust, or recompute them, and do not change the cause names. "
        f"{notes_instruction}\n\nDiagnosis: {base_explanation}\n\nRewritten:"
    )
    try:
        resp = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": HF_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.3,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0]["message"]
        # Reasoning models sometimes return content: null with the actual text
        # elsewhere (e.g. reasoning_content), or an empty string on a truncated
        # response -- treat any of those as "no usable rewrite" rather than
        # crashing on .strip() against None.
        text = (message.get("content") or message.get("reasoning_content") or "").strip()
        if not text:
            return base_explanation

        # Numeric fidelity check: the model is instructed not to change confidence
        # scores, but instructions aren't guarantees -- LLMs do sometimes drift a
        # number (e.g. round 33% to 32%) while paraphrasing. Since the confidence
        # score is the one number in this whole system that has to be exactly what
        # the classifier produced, don't trust the model's compliance -- verify it.
        # Every "NN%" in the template must appear verbatim in the rewrite, or the
        # whole rewrite is discarded in favor of the template.
        required_numbers = set(re.findall(r"\d+%", base_explanation))
        rewritten_numbers = set(re.findall(r"\d+%", text))
        if not required_numbers.issubset(rewritten_numbers):
            logger.warning(
                "HF rewrite dropped/changed a confidence number (expected %s, got %s) -- using template",
                required_numbers, rewritten_numbers,
            )
            return base_explanation

        return text
    except Exception as exc:
        # Model unavailable on any provider, rate-limited, or unreachable — fall back
        # to template, never fail the request. Logged (not swallowed) so this is
        # diagnosable from `docker compose logs backend` instead of looking identical
        # to "LLM disabled".
        logger.warning("HF rewrite failed, falling back to template: %s", exc)
        return base_explanation
