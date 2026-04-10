import json
import os
from typing import Any, Dict, List

import requests
from openai import OpenAI
from requests import RequestException


API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
SEED = int(os.environ.get("EVAL_SEED", "7"))
MAX_CLIENT_STEPS = int(os.environ.get("MAX_CLIENT_STEPS", "30"))

TASKS = ["task_easy", "task_medium", "task_hard"]


def _require_env() -> None:
    missing: List[str] = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sanitize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    action_type = action.get("type", "wait")
    if action_type not in {"procure", "dispatch", "switch_supplier", "maintenance", "wait"}:
        return {"type": "wait"}

    if action_type == "procure":
        return {
            "type": "procure",
            "item": str(action.get("item", "vaccine_a")),
            "quantity": max(1, _safe_int(action.get("quantity", 120), 120)),
            "supplier": str(action.get("supplier", "balanced")),
            "expedited": bool(action.get("expedited", False)),
        }

    if action_type == "dispatch":
        return {
            "type": "dispatch",
            "route": str(action.get("route", "standard")),
            "units": max(1, _safe_int(action.get("units", 80), 80)),
        }

    if action_type == "switch_supplier":
        return {
            "type": "switch_supplier",
            "supplier": str(action.get("supplier", "balanced")),
        }

    if action_type == "maintenance":
        spend = action.get("spend", 400.0)
        try:
            spend_value = float(spend)
        except (TypeError, ValueError):
            spend_value = 400.0
        return {
            "type": "maintenance",
            "spend": max(100.0, spend_value),
        }

    return {"type": "wait"}


def llm_action(client: OpenAI, observation: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    prompt = (
        "You control a cold-chain vaccine supply environment. "
        "Return exactly one JSON object and nothing else.\n"
        "Allowed actions:\n"
        "1) procure: {\"type\":\"procure\",\"item\":\"vaccine_a|vaccine_b|vaccine_c\",\"quantity\":int,\"supplier\":\"low_cost|balanced|resilient\",\"expedited\":bool}\n"
        "2) dispatch: {\"type\":\"dispatch\",\"route\":\"economy|standard|cold_express\",\"units\":int}\n"
        "3) switch_supplier: {\"type\":\"switch_supplier\",\"supplier\":\"low_cost|balanced|resilient\"}\n"
        "4) maintenance: {\"type\":\"maintenance\",\"spend\":float}\n"
        "5) wait: {\"type\":\"wait\"}\n"
        f"Task: {task_id}\n"
        "Observation JSON:\n"
        f"{json.dumps(observation, separators=(',', ':'))}\n"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=100,
            messages=[
                {"role": "system", "content": "You are a precise operations policy model."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()
    except Exception:
        return {"type": "wait"}

    if not raw:
        return {"type": "wait"}

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return _sanitize_action(parsed)
    except Exception:
        pass
    return {"type": "wait"}


def run_task(client: OpenAI, task_id: str) -> float:
    try:
        reset_resp = requests.post(
            f"{ENV_URL}/reset",
            params={"task_id": task_id, "seed": SEED},
            timeout=30,
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json()["observation"]
    except (RequestException, KeyError, TypeError, ValueError, json.JSONDecodeError):
        print(
            json.dumps(
                {
                    "event": "[END]",
                    "task": task_id,
                    "total_steps": 0,
                    "total_reward": 0.0,
                    "final_score": 0.0,
                }
            )
        )
        return 0.0

    print(json.dumps({"event": "[START]", "task": task_id, "observation": obs}))

    step_num = 0
    total_reward = 0.0
    final_score = 0.0

    while True:
        action = llm_action(client, obs, task_id)

        try:
            step_resp = requests.post(
                f"{ENV_URL}/step",
                json={"action": action},
                timeout=30,
            )
            step_resp.raise_for_status()
            data = step_resp.json()

            obs = data["observation"]
            reward = float(data["reward"])
            done = bool(data["done"])
            info = data["info"]
        except (RequestException, KeyError, TypeError, ValueError, json.JSONDecodeError):
            print(
                json.dumps(
                    {
                        "event": "[END]",
                        "task": task_id,
                        "total_steps": step_num,
                        "total_reward": round(total_reward, 3),
                        "final_score": 0.0,
                    }
                )
            )
            return 0.0

        step_num += 1
        total_reward += reward

        print(
            json.dumps(
                {
                    "event": "[STEP]",
                    "task": task_id,
                    "step": step_num,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "observation": obs,
                }
            )
        )

        if done or step_num >= MAX_CLIENT_STEPS:
            final_score = float(info.get("final_score", 0.0))
            print(
                json.dumps(
                    {
                        "event": "[END]",
                        "task": task_id,
                        "total_steps": step_num,
                        "total_reward": round(total_reward, 3),
                        "final_score": final_score,
                    }
                )
            )
            return final_score


def main() -> None:
    try:
        _require_env()
    except SystemExit:
        # Keep a zero exit for evaluator robustness while still emitting valid task end logs.
        for task in TASKS:
            print(
                json.dumps(
                    {
                        "event": "[END]",
                        "task": task,
                        "total_steps": 0,
                        "total_reward": 0.0,
                        "final_score": 0.0,
                    }
                )
            )
        return

    try:
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    except Exception:
        for task in TASKS:
            print(
                json.dumps(
                    {
                        "event": "[END]",
                        "task": task,
                        "total_steps": 0,
                        "total_reward": 0.0,
                        "final_score": 0.0,
                    }
                )
            )
        return

    for task in TASKS:
        try:
            run_task(client, task)
        except Exception:
            print(
                json.dumps(
                    {
                        "event": "[END]",
                        "task": task,
                        "total_steps": 0,
                        "total_reward": 0.0,
                        "final_score": 0.0,
                    }
                )
            )


if __name__ == "__main__":
    main()
