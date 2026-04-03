import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import requests
import yaml

from tasks import GRADERS, TASKS


ROOT = Path(__file__).resolve().parent
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def _fail(message: str) -> None:
    print(f"[FAIL] {message}")


def check_openenv_yaml() -> bool:
    path = ROOT / "openenv.yaml"
    if not path.exists():
        _fail("openenv.yaml missing")
        return False

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    required_top = ["name", "version", "tasks", "api", "observation_space", "action_space"]
    for key in required_top:
        if key not in data:
            _fail(f"openenv.yaml missing key: {key}")
            return False

    if len(data["tasks"]) < 3:
        _fail("openenv.yaml has fewer than 3 tasks")
        return False

    _ok("openenv.yaml structure validated")
    return True


def check_api() -> bool:
    try:
        root = requests.get(f"{ENV_URL}/", timeout=10)
        if root.status_code != 200:
            _fail(f"GET / returned {root.status_code}")
            return False

        reset = requests.post(
            f"{ENV_URL}/reset",
            params={"task_id": "task_easy", "seed": 7},
            timeout=10,
        )
        reset.raise_for_status()
        reset_json = reset.json()
        if "observation" not in reset_json:
            _fail("/reset response missing observation")
            return False

        step = requests.post(
            f"{ENV_URL}/step",
            json={"action": {"type": "wait"}},
            timeout=10,
        )
        step.raise_for_status()
        step_json = step.json()
        for key in ["observation", "reward", "done", "info"]:
            if key not in step_json:
                _fail(f"/step response missing {key}")
                return False

        state = requests.get(f"{ENV_URL}/state", timeout=10)
        state.raise_for_status()
        state_json = state.json()
        for key in ["state", "task_id", "step_count", "total_reward"]:
            if key not in state_json:
                _fail(f"/state response missing {key}")
                return False

        _ok("API endpoints respond correctly")
        return True
    except Exception as exc:
        _fail(f"API check failed: {exc}")
        return False


def check_graders() -> bool:
    base_state: Dict[str, object] = {
        "total_demand_units": 1000,
        "unmet_demand_units": 120,
        "inventory": {"vaccine_a": 240, "vaccine_b": 170, "vaccine_c": 130},
        "safety_stock": {"vaccine_a": 180, "vaccine_b": 150, "vaccine_c": 120},
        "initial_budget": 25000.0,
        "cumulative_spend": 8200.0,
        "deliveries_completed": 24,
        "deliveries_on_time": 18,
        "cold_chain_integrity": 0.86,
        "units_received": 1500,
        "spoilage_units": 90,
        "supplier_risk": 0.2,
        "disruption_impact": 0.1,
    }

    for task_id, grader in GRADERS.items():
        score = float(grader(base_state))
        if not (0.0 <= score <= 1.0):
            _fail(f"grader out of range for {task_id}: {score}")
            return False

    if len(TASKS) < 3:
        _fail("fewer than 3 tasks defined")
        return False

    _ok("Tasks and graders validated (3+ tasks, score range 0.0-1.0)")
    return True


def check_docker_build() -> bool:
    try:
        result = subprocess.run(
            ["docker", "build", "-t", "cold-chain-openenv:local", "."],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode != 0:
            _fail("Docker build failed")
            print(result.stdout)
            print(result.stderr)
            return False
        _ok("Docker build succeeded")
        return True
    except FileNotFoundError:
        _fail("Docker not installed or not in PATH")
        return False
    except Exception as exc:
        _fail(f"Docker build check failed: {exc}")
        return False


def check_inference() -> bool:
    required = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        _fail(
            "Skipping inference check because environment variables are missing: "
            + ", ".join(missing)
        )
        return False

    proc = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=1200,
        env=os.environ.copy(),
    )

    if proc.returncode != 0:
        _fail("inference.py failed")
        print(proc.stdout)
        print(proc.stderr)
        return False

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    parsed: List[Dict[str, object]] = []
    for line in lines:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    has_start = any(item.get("event") == "[START]" for item in parsed)
    has_step = any(item.get("event") == "[STEP]" for item in parsed)
    has_end = any(item.get("event") == "[END]" for item in parsed)

    if not (has_start and has_step and has_end):
        _fail("inference logs do not include required [START]/[STEP]/[END] events")
        return False

    _ok("inference.py ran and produced required structured logs")
    return True


def main() -> None:
    checks = {
        "openenv_yaml": check_openenv_yaml(),
        "graders": check_graders(),
        "api": check_api(),
        "docker": check_docker_build(),
        "inference": check_inference(),
    }

    print("\nValidation Summary")
    for name, status in checks.items():
        print(f"- {name}: {'PASS' if status else 'FAIL'}")

    if all(checks.values()):
        print("\nAll checks passed.")
        raise SystemExit(0)

    print("\nOne or more checks failed. Fix the failing items before submission.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
