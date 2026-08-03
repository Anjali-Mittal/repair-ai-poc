# AI Repair Intelligence — Project Handoff

Hero MotoCorp PwC POC. Fault-code-to-cause diagnostic tool for two-wheeler
technicians. Structured ML model ranks probable root cause + confidence; an
optional LLM layer rewrites the explanation in plain language. Read this whole
file before touching code — several non-obvious decisions below prevent
re-introducing bugs that were already found and fixed once.

## What this actually is (be honest about this)

This is a **POC demonstrating an architecture pattern**, not a validated
diagnostic model. The training data is **synthetic** — generated, not real
Hero service records. Accuracy numbers (ICE 52.5%, Scooter 54.3%, EV 69.6%)
describe how well the model fits its own synthetic labels, not real-world
diagnostic accuracy. Do not let this get cited as "the model is X% accurate"
without that caveat attached — that's a promise the current build cannot back up.

What *is* real: the ~47 fault codes are genuine SAE J2012 OBD-II codes (for
ICE/Scooter) mapped to their actual documented causes — that mapping is
domain-correct, hand-built, and worth keeping even after real data arrives. EV
codes are explicitly NOT from any published standard (none exists for EVs the
way OBD-II exists) — they're representative categories, labeled as such in
code comments.

## Architecture

```
Technician input → XGBoost classifier (per vehicle type) → ranked causes + confidence
                                                                    ↓
                                            Static knowledge base (recommendation/
                                            checklist/risk per cause — NOT model output)
                                                                    ↓
                                    Template explanation (deterministic, always generated)
                                                                    ↓
                                    [optional] LLM rewrite (paraphrase only, verified
                                    to never change numbers — see explainer.py)
```

**Core principle, do not violate it going forward**: the structured model
decides (ranked causes + confidence numbers). The LLM only rephrases. Every
number the LLM outputs is checked against the template — if even one `NN%`
doesn't survive the rewrite verbatim, the whole rewrite is discarded and the
template is used instead (`explainer.py::_hf_rewrite`, numeric fidelity check).
This was a real bug that shipped once (LLM rounded 33%→32%) — the check exists
because of that, not speculatively.

## Why models are split per vehicle type

`train_model.py` trains **three separate XGBoost models** (ICE, Scooter, EV),
not one shared model with vehicle_type as a feature. Two reasons:
1. Failure physics genuinely differs (fuel/ignition wear vs. battery/BMS) —
   one shared model tends to average across regimes rather than learn either well.
2. **Side benefit, load-bearing now**: each model's fault-code encoder only
   knows that type's codes. An ICE code submitted against the EV model is
   literally unrecognized to it — this is what makes cross-vehicle-type code
   misuse detectable (see below), not an extra check bolted on.

## Known-code / unknown-code handling (important, don't regress this)

Early in this build, an unrecognized fault code was **silently substituted**
with the most common training code and predicted as if it were real — a
technician got a confident wrong answer with no indication anything was off.
This was found and fixed. Current behavior (`predictor.py` +
`main.py::_find_codes_in_other_types` + `explainer.py`):

- Every code is checked against `fault_code_encoder.classes_` for the selected
  vehicle type.
- Unknown code → `low_confidence: True`, `unknown_codes: [...]` in the API
  response, and the ranking is explicitly labeled "NOT a code-specific finding."
- If the code IS known for a *different* vehicle type, the explanation says so
  explicitly: `"P0301 is a known code for ICE/Scooter, not EV — check the
  vehicle type selected."` This catches the common real mistake (wrong vehicle
  type selected) rather than just saying "unrecognized."
- **If you add more fault codes later, this behavior is automatic** — it's
  driven by the encoder's known classes, not a hardcoded list.

## Multi-DTC support

`fault_code` accepts comma-separated codes. Each is scored independently, then
combined per-cause via the independent-evidence OR rule:
`combined = 1 - ∏(1 - p_i)`. A cause flagged by multiple codes ranks higher
than either code alone — this is deliberate, not a bug, and is the honest way
to treat corroborating evidence (not averaging, not just picking one code).

## Vehicle identity fields (VIN/make/model/year)

Stored in DB, shown in explanation text and "similar past cases" — **NOT fed
into the model**. Stated explicitly in `main.py`'s `DiagnosisRequest` comment.
Adding them as real model features needs training data where failure patterns
actually vary by make/model, which the synthetic dataset doesn't have. Don't
wire these into `predict_top_causes` without real data backing it — that would
recreate the exact "fake precision" problem already avoided elsewhere.

## LLM layer specifics

- HF Inference Providers router: `https://router.huggingface.co/v1/chat/completions`
  (OpenAI-compatible chat completions, NOT the old per-model
  `api-inference.huggingface.co` endpoint — that's retired).
- Default model: `openai/gpt-oss-120b` (confirmed served by multiple providers
  per HF docs). **Do not default to a Meta Llama model** — gated, requires
  manual license acceptance per-user, and often isn't served by any provider
  under the router even after approval. Wasted significant debugging time on
  this once already.
- `gpt-oss-120b` is a reasoning model — `message.content` can be `null` with
  the real answer elsewhere, or truncated to empty if `max_tokens` is too low
  because reasoning tokens eat the budget first. Current `max_tokens: 400`,
  and `_hf_rewrite` checks `content` then falls back to `reasoning_content`
  before giving up. If content is empty, falls back to template silently
  (logged, not crashed).
- Notes-handling: the prompt explicitly forbids vague acknowledgment ("this
  should be considered") and requires the model to say something concrete —
  does the symptom fit the cause or not, is there one added check worth doing,
  or does the symptom not even make sense for this vehicle type. This was a
  direct fix for a real bad response the LLM gave.
- All HF failures are logged via `logger.warning`, never silently swallowed —
  check with `docker compose logs backend | Select-String "HF"` (PowerShell)
  when debugging.

## Frontend

`backend/static/index.html` — single file, light theme, Plus Jakarta Sans +
IBM Plex Mono, Chart.js for the top-cause donut. No sidebar (explicit user
preference). No decorative icons on KPI cards (removed per feedback — label
text alone conveys what each card is). No "Most Predicted Causes" bar chart
(removed, wasn't useful). Key UI elements:
- Multi-DTC input, optional VIN/make/model/year fields
- Ranked causes with donut (top cause) + progress bars (rest) —
  **bars use `<span class="fill">` and require `display: block`**, a `<span>`
  is inline by default and silently ignores `width` — this broke once, fixed,
  don't revert.
- Risk badge (high/medium/low), recommendation + inspection checklist per cause
- One-click "Confirm '<cause>'" button alongside free-text override
- "Similar past cases" panel — real DB query, not fabricated

## Docker / infra gotchas already hit and fixed

- `docker-compose.yml`: Postgres has a `pg_isready` healthcheck; backend
  `depends_on: condition: service_healthy`. Without this, the backend races
  Postgres's post-`initdb` restart on first boot and crashes
  (`connection refused`). Don't remove the healthcheck.
- `db.py::run_migrations()` retries on `OperationalError` as a second line of
  defense, and uses `ADD COLUMN IF NOT EXISTS` so upgrading an existing volume
  doesn't require wiping it.
- `.env` changes require `docker compose down` + `up` (not just `up --build`)
  to be picked up reliably — compose doesn't always re-inject env vars into an
  already-created container otherwise.

## File map

```
backend/app/main.py                    FastAPI app, all endpoints
backend/app/db.py                      DB connection + migrations
backend/app/repair_knowledge.py        Static KB: risk/recommendation/checklist per cause
backend/app/train_model.py             Trains 3 per-vehicle-type XGBoost models
backend/app/services/predictor.py      Inference, multi-DTC combination, unknown-code detection
backend/app/services/explainer.py      Template + LLM rewrite, numeric fidelity check
backend/app/model_artifacts/<type>/    Trained model + encoders per vehicle type (regenerate via train_model.py)
backend/static/index.html              Frontend, single file
data/generate_dataset.py               Synthetic dataset generator — FAULT_CODES dict is the source of truth for code→cause mapping
data/repair_history.csv                Generated training data (20,000 rows)
docker-compose.yml                     Postgres + backend, healthcheck-gated
docker/init.sql                        Fresh-install schema
.env / .env.example                    HUGGINGFACE_API_KEY, HF_MODEL, DATABASE_URL
```

## Backlog not yet done (from prioritization earlier in the project)

Picked and built: multi-DTC, repair recommendations, inspection checklists,
risk tagging, similar-past-cases, vehicle identity fields, expanded fault-code
coverage (47 codes).

Not done, explicitly deferred:
- **Freeze-frame data / live sensor values** — needs real vehicle telemetry,
  can't be fabricated honestly.
- **NLP on technician notes as a model feature** — currently notes only reach
  the LLM layer for explanation, not the classifier. A real feature (even
  simple keyword-based) would need labeled data connecting notes to causes.
- **Automated retraining from overrides** — feedback loop is manual today.
  `RETRAIN_BATCH_SIZE = 25` in `main.py` only tracks progress toward a batch,
  doesn't trigger anything. Was previously shown in the UI ("N more →  next
  retrain batch"), then removed per user request to declutter — logic still
  in the API response if needed again.
- **Real Hero service data integration** — the actual path to making this a
  working product instead of a demo. Same CSV schema
  (`vehicle_type, fault_code, vehicle_age_months, mileage_km, prior_services,
  technician_notes, replaced_part, root_cause`) so it's a data swap +
  retrain, not a rebuild — provided the real data's fault codes and causes
  follow the same domain-accurate mapping principle as `FAULT_CODES` here.

## Person context

Building this solo as a PwC intern (GDC SBU, Advisory), with an associate who
reviews deliverables occasionally. Prefers terse responses, no narration
before acting, tool calls run silently with results shown. Wants a heads-up +
this kind of handoff file generated near the end of long sessions.
