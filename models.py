from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal["procure", "dispatch", "switch_supplier", "maintenance", "wait"]


class StepRequest(BaseModel):
    action: Dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any]


class StateResponse(BaseModel):
    state: Dict[str, Any]
    task_id: str
    step_count: int
    total_reward: float


class TaskMeta(BaseModel):
    id: str
    name: str
    description: str
    max_steps: int


class TasksResponse(BaseModel):
    tasks: List[str]
    details: Dict[str, TaskMeta]
