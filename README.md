# AI Repair Intelligence Platform — POC

## Structure
- `data/generate_dataset.py` — synthetic repair history (real DTC codes, mileage-weighted causes)
- `backend/app/train_model.py` — trains XGBoost cause-ranking model
- `backend/app/services/predictor.py` — loads model, ranks top-k causes
- `backend/app/services/explainer.py` — plain-language layer (template by default; set `HUGGINGFACE_API_KEY` in `.env` to rewrite via an open-source model)
- `backend/app/main.py` — FastAPI: `/diagnose`, `/feedback`, `/analytics/summary`
- `backend/static/index.html` — technician console + analytics dashboard
- `docker-compose.yml` + `docker/init.sql` — Postgres + API containers

## Run locally

1. Copy `.env.example` to `.env` and fill in a free Hugging Face token (optional — leave blank to run template-only, no LLM call):
   ```
   cp .env.example .env
   ```

2. Generate data + train model (already done once, re-run if you change the generator):
   ```
   python3 data/generate_dataset.py
   cd backend/app && python3 train_model.py
   ```

3. Start Postgres + API:
   ```
   docker compose up --build
   ```

3. Open dashboard:
   ```
   http://localhost:8000/dashboard
   ```

4. API docs (Swagger):
   ```
   http://localhost:8000/docs
   ```

## Optional: enable LLM explanation rewrite (free, open-source)
Get a free token at https://huggingface.co/settings/tokens, then in `.env`:
```
HUGGINGFACE_API_KEY=hf_...
HF_MODEL=google/flan-t5-large
```
Any instruct model on the HF Inference API works — swap `HF_MODEL` for something larger (e.g. `mistralai/Mistral-7B-Instruct-v0.3`) for richer phrasing.

Without a key, explanations are template-generated — deterministic and audit-safe by default, matching the "structured model decides, LLM only explains" design from the concept note.

## Notes
- Model accuracy ~46% overall on synthetic data (expected — causes are intentionally noisy/ambiguous except wear-related ones, which hit 75-89% recall). Swap in real Hero repair history via `data/repair_history.csv` schema to improve.
- Technician overrides via `/feedback` feed the `agreement_rate` metric on the dashboard — this is your feedback loop from the architecture diagram.
