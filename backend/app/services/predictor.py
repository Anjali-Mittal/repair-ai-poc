import os
import joblib
import numpy as np

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")

# Below this confidence, ranked causes are still returned but flagged low_confidence
# so the API/UI can defer to technician judgment instead of presenting a guess as
# if it were a finding.
LOW_CONFIDENCE_THRESHOLD = 0.35

_VEHICLE_TYPES = ["ICE", "Scooter", "EV"]
_artifacts = {}


def _load(vehicle_type: str):
    if vehicle_type in _artifacts:
        return _artifacts[vehicle_type]
    vt_dir = os.path.join(ARTIFACT_DIR, vehicle_type)
    bundle = {
        "model": joblib.load(os.path.join(vt_dir, "xgb_model.joblib")),
        "fault_code_encoder": joblib.load(os.path.join(vt_dir, "fault_code_encoder.joblib")),
        "root_cause_encoder": joblib.load(os.path.join(vt_dir, "root_cause_encoder.joblib")),
        "fault_code_fallback": joblib.load(os.path.join(vt_dir, "fault_code_fallback.joblib")),
    }
    _artifacts[vehicle_type] = bundle
    return bundle


def known_fault_codes(vehicle_type: str):
    bundle = _load(vehicle_type)
    return list(bundle["fault_code_encoder"].classes_)


def _predict_single_code(vehicle_type: str, fault_code: str, vehicle_age_months: int,
                          mileage_km: int, prior_services: int, top_k: int):
    bundle = _load(vehicle_type)
    fc_le = bundle["fault_code_encoder"]
    root_cause_le = bundle["root_cause_encoder"]

    is_known = fault_code in fc_le.classes_
    if is_known:
        fc_enc = fc_le.transform([fault_code])[0]
    else:
        # No training data for this exact code -- falling back to the most common
        # code's encoding lets the model still run, but the prediction it produces
        # is NOT specific to this code. Caller must surface is_known=False rather
        # than let this look like a real code-specific prediction.
        fc_enc = fc_le.transform([bundle["fault_code_fallback"]])[0]

    X = np.array([[fc_enc, vehicle_age_months, mileage_km, prior_services]])
    probs = bundle["model"].predict_proba(X)[0]

    top_idx = np.argsort(probs)[::-1][:top_k]
    ranked = [
        {"cause": root_cause_le.inverse_transform([i])[0], "confidence": round(float(probs[i]), 3)}
        for i in top_idx
    ]
    return ranked, is_known


def predict_top_causes(vehicle_type: str, fault_codes, vehicle_age_months: int,
                        mileage_km: int, prior_services: int, top_k: int = 3):
    """
    fault_codes: a single code (str) or a list of codes. When multiple codes are
    given, each is scored independently against the same vehicle context, then
    combined per-cause using the independent-evidence OR rule
    (combined = 1 - prod(1 - p_i)) -- a cause flagged by more than one code gets a
    higher combined score than either code alone, which is the honest way to treat
    corroborating evidence rather than just averaging or picking one code's ranking.

    Any code not seen in training is tracked in unknown_codes and forces
    low_confidence -- its "prediction" only reflects the fallback code's pattern,
    not this code, and must never be presented as if it were code-specific.
    """
    if vehicle_type not in _VEHICLE_TYPES:
        vehicle_type = "ICE"  # most common type, safest default for an unrecognized value

    if isinstance(fault_codes, str):
        fault_codes = [fault_codes]
    fault_codes = [c.strip().upper() for c in fault_codes if c and c.strip()]
    if not fault_codes:
        fault_codes = ["UNKNOWN"]

    per_code = {}
    combined = {}
    unknown_codes = []
    for code in fault_codes:
        ranked, is_known = _predict_single_code(vehicle_type, code, vehicle_age_months,
                                                  mileage_km, prior_services, top_k=max(top_k, 5))
        per_code[code] = ranked[:top_k]
        if not is_known:
            unknown_codes.append(code)
            continue  # don't let an unrecognized code's fallback-based guess pollute the combined ranking
        for r in ranked:
            prev = combined.get(r["cause"], 0.0)
            combined[r["cause"]] = 1 - (1 - prev) * (1 - r["confidence"])

    if not combined:
        # every submitted code was unrecognized -- still return the fallback-based
        # ranking for the first code so the UI has something to show, but it will
        # be forced low_confidence below regardless of the raw number.
        ranked_combined = per_code[fault_codes[0]][:top_k]
    else:
        ranked_combined = sorted(
            ({"cause": c, "confidence": round(p, 3)} for c, p in combined.items()),
            key=lambda x: x["confidence"], reverse=True,
        )[:top_k]

    low_confidence = bool(unknown_codes) or ranked_combined[0]["confidence"] < LOW_CONFIDENCE_THRESHOLD
    return ranked_combined, low_confidence, per_code, unknown_codes
