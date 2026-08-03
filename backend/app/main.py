from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional, List, Union
import os

from app.services.predictor import predict_top_causes, known_fault_codes
from app.services.explainer import build_explanation
from app.repair_knowledge import lookup
from app.db import get_conn, run_migrations

app = FastAPI(title="AI Repair Intelligence API")


@app.on_event("startup")
def _startup():
    run_migrations()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
app.mount("/dashboard", StaticFiles(directory=static_dir, html=True), name="dashboard")


@app.get("/")
def root():
    return RedirectResponse(url="/dashboard")

# Feedback loop is currently manual: overrides are logged but nothing retrains
# automatically yet. This just reports progress toward a stated batch size so the
# dashboard is honest about that ("collecting for next retrain", not "retraining now").
RETRAIN_BATCH_SIZE = 25


class DiagnosisRequest(BaseModel):
    vehicle_type: str          # ICE, Scooter, EV
    fault_code: Union[str, List[str]]   # single code, or comma-separated / list of codes
    vehicle_age_months: int
    mileage_km: int
    prior_services: int
    technician_notes: Optional[str] = ""
    # Vehicle identity -- stored for record-keeping and shown in the explanation/
    # history views. NOT fed into the model: it's trained on vehicle_type only, so
    # make/model/year/VIN don't change the prediction. Adding them as real model
    # features would need retraining on data that actually varies by make/model,
    # which the current synthetic dataset doesn't.
    vin: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None


class OverrideRequest(BaseModel):
    prediction_id: int
    technician_override: str


def _normalize_codes(fault_code: Union[str, List[str]]) -> List[str]:
    if isinstance(fault_code, list):
        return [c.strip().upper() for c in fault_code if c and c.strip()]
    return [c.strip().upper() for c in fault_code.split(",") if c.strip()]


def _find_codes_in_other_types(unknown_codes: List[str], current_type: str) -> dict:
    """
    For each code the current vehicle type doesn't recognize, check the other two
    vehicle types -- if the code is a real known code there, that's a much more
    useful thing to tell the technician than just "unrecognized" (it's likely the
    wrong vehicle type was selected, not a made-up code).
    """
    if not unknown_codes:
        return {}
    other_types = [vt for vt in ("ICE", "Scooter", "EV") if vt != current_type]
    result = {}
    for code in unknown_codes:
        found_in = [vt for vt in other_types if code in known_fault_codes(vt)]
        if found_in:
            result[code] = found_in
    return result


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/known-codes/{vehicle_type}")
def known_codes(vehicle_type: str):
    try:
        return {"vehicle_type": vehicle_type, "codes": sorted(known_fault_codes(vehicle_type))}
    except FileNotFoundError:
        return {"vehicle_type": vehicle_type, "codes": []}


@app.post("/diagnose")
def diagnose(req: DiagnosisRequest):
    codes = _normalize_codes(req.fault_code)

    ranked, low_confidence, per_code, unknown_codes = predict_top_causes(
        req.vehicle_type, codes, req.vehicle_age_months,
        req.mileage_km, req.prior_services, top_k=3,
    )
    vehicle_label = req.vehicle_type
    if req.make or req.model or req.year:
        vehicle_label = " ".join(str(p) for p in [req.year, req.make, req.model] if p) + f" ({req.vehicle_type})"

    explanation = build_explanation(
        codes, vehicle_label, req.mileage_km,
        req.vehicle_age_months, ranked, req.technician_notes, low_confidence,
        unknown_codes=unknown_codes,
        code_found_in=_find_codes_in_other_types(unknown_codes, req.vehicle_type),
    )

    top_knowledge = lookup(ranked[0]["cause"])
    ranked_with_knowledge = [
        {**c, **lookup(c["cause"])} for c in ranked
    ]

    conn = get_conn()
    with conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO predictions
               (fault_code, vehicle_type, vehicle_age_months, mileage_km, prior_services,
                vin, make, model, year,
                predicted_cause, confidence, explanation)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (",".join(codes), req.vehicle_type, req.vehicle_age_months, req.mileage_km,
             req.prior_services, req.vin, req.make, req.model, req.year,
             ranked[0]["cause"], ranked[0]["confidence"], explanation),
        )
        pred_id = cur.fetchone()["id"]

        cur.execute(
            """SELECT predicted_cause, confidence, technician_override, make, model, year, created_at
               FROM predictions
               WHERE fault_code = ANY(%s) AND id != %s
               ORDER BY created_at DESC LIMIT 5""",
            (codes, pred_id),
        )
        similar_cases = cur.fetchall()
    conn.close()

    return {
        "prediction_id": pred_id,
        "fault_codes": codes,
        "unknown_codes": unknown_codes,
        "ranked_causes": ranked_with_knowledge,
        "per_code_breakdown": per_code,
        "explanation": explanation,
        "low_confidence": low_confidence,
        "top_cause_risk": top_knowledge["risk"],
        "top_cause_recommendation": top_knowledge["recommendation"],
        "top_cause_checklist": top_knowledge["checklist"],
        "similar_cases": similar_cases,
    }


@app.post("/feedback")
def feedback(req: OverrideRequest):
    conn = get_conn()
    with conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE predictions SET technician_override=%s WHERE id=%s",
            (req.technician_override, req.prediction_id),
        )
    conn.close()
    return {"status": "recorded"}


@app.get("/analytics/summary")
def analytics_summary():
    conn = get_conn()
    with conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS total FROM predictions")
        total = cur.fetchone()["total"]

        cur.execute(
            """SELECT predicted_cause, COUNT(*) AS n
               FROM predictions GROUP BY predicted_cause ORDER BY n DESC LIMIT 5"""
        )
        top_causes = cur.fetchall()

        cur.execute(
            """SELECT COUNT(*) AS overridden FROM predictions
               WHERE technician_override IS NOT NULL"""
        )
        overridden = cur.fetchone()["overridden"]

        cur.execute("SELECT AVG(confidence) AS avg_conf FROM predictions")
        avg_conf = cur.fetchone()["avg_conf"]
    conn.close()

    overrides_toward_retrain = overridden % RETRAIN_BATCH_SIZE

    return {
        "total_predictions": total,
        "top_causes": top_causes,
        "override_count": overridden,
        "avg_confidence": round(float(avg_conf), 3) if avg_conf is not None else None,
        "agreement_rate": round(1 - (overridden / total), 3) if total else None,
        "retrain_batch_size": RETRAIN_BATCH_SIZE,
        "overrides_toward_retrain": overrides_toward_retrain,
    }
