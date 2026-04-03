from fastapi import FastAPI, HTTPException, Query

from environment import ColdChainSupplyEnv
from models import ResetResponse, StateResponse, StepRequest, StepResponse
from tasks import TASKS


app = FastAPI(title="Cold-Chain Supply OpenEnv", version="1.0.0")
env = ColdChainSupplyEnv()


@app.get("/")
def root() -> dict:
    return {"status": "ok", "env": "cold-chain-supply-openenv"}


@app.post("/reset", response_model=ResetResponse)
def reset(
    task_id: str = Query(default="task_easy"),
    seed: int = Query(default=7, ge=0, le=10_000),
) -> ResetResponse:
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail="invalid_task_id")
    result = env.reset(task_id=task_id, seed=seed)
    return ResetResponse(**result)


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    result = env.step(req.action)
    return StepResponse(**result)


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return StateResponse(**env.get_state())


@app.get("/tasks")
def list_tasks() -> dict:
    return {"tasks": list(TASKS.keys()), "details": TASKS}
