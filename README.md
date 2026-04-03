# Cold-Chain Supply OpenEnv

A complete real-world OpenEnv environment for training AI agents on cold-chain vaccine distribution decisions.

## What makes this real-world

This environment models practical operational tradeoffs:
- procurement lead times and supplier risk
- route cost versus on-time delivery
- cold-chain integrity drift and maintenance spending
- spoilage and unmet demand penalties
- emissions accumulation

## Task Suite (Easy to Hard)

| Task ID | Difficulty | Goal | Max Steps |
|---|---|---|---|
| task_easy | easy | Keep stock stable above safety levels with cost control | 12 |
| task_medium | medium | Improve routing timeliness while staying efficient | 16 |
| task_hard | hard | Jointly optimize resilience, service, waste, and spend | 22 |

Each task has a dedicated grader in tasks.py and scores are always in the range 0.0 to 1.0.

## API (OpenEnv style)

- GET / -> health check
- POST /reset?task_id=task_easy&seed=7 -> reset episode
- POST /step -> submit action as JSON
- GET /state -> full internal state
- GET /tasks -> list task metadata

## Observation Space

Returned by reset and step:
- day: int
- budget: float
- active_supplier: str
- cold_chain_integrity: float
- inventory: dict
- pending_orders: list
- deliveries_completed: int
- deliveries_on_time: int
- unmet_demand_units: int
- spoilage_units: int
- emissions_kg: float

## Action Space

Action payload format:
- procure: {"type":"procure","item":"vaccine_a","quantity":120,"supplier":"balanced","expedited":false}
- dispatch: {"type":"dispatch","route":"standard","units":80}
- switch_supplier: {"type":"switch_supplier","supplier":"resilient"}
- maintenance: {"type":"maintenance","spend":400}
- wait: {"type":"wait"}

## Repository Layout

- app.py: FastAPI app and endpoints
- environment.py: environment dynamics and reward shaping
- tasks.py: task specs and graders
- models.py: typed request/response models
- openenv.yaml: OpenEnv metadata/spec
- inference.py: baseline LLM runner (required filename)
- validator.py: pre-submission checks
- Dockerfile: HF Spaces Docker deployment

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Mandatory Environment Variables for Inference

Define these before running inference.py:
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN
- ENV_URL

Windows PowerShell example:

```powershell
$env:API_BASE_URL = "https://api.openai.com/v1"
$env:MODEL_NAME = "gpt-4o-mini"
$env:HF_TOKEN = "your_api_key"
$env:ENV_URL = "http://localhost:7860"
$env:EVAL_SEED = "7"
python inference.py
```

## Structured Log Format Compliance

inference.py emits strict JSON logs with events:
- [START]
- [STEP]
- [END]

No extra event type is emitted to avoid evaluator parser issues.

## Pre-Submission Validation

Run the validator:

```bash
python validator.py
```

Validator checks:
- openenv.yaml structure
- API /reset /step /state behavior
- 3+ tasks and grader score range
- Docker build
- inference structured logs

## Deploy to Hugging Face Spaces (Docker)

1. Push this repository to GitHub.
2. Create a new Hugging Face Space using Docker SDK.
3. Link your repo or push the same files to the Space repo.
4. In Space settings, add Variables/Secrets:
   - API_BASE_URL
   - MODEL_NAME
   - HF_TOKEN
   - ENV_URL (if needed by your evaluation flow)
5. Confirm Space build succeeds.
6. Validate health endpoint and reset endpoint.

## GitHub Push Steps

```bash
git init
git add .
git commit -m "Initial cold-chain OpenEnv implementation"
git branch -M main
git remote add origin <your_repo_url>
git push -u origin main
```

## Notes on Originality

To keep this submission defensible and original:
- keep your own problem framing and assumptions
- document design decisions and reward choices
- include your own experiments and improvements
