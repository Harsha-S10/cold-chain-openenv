import math
import random
from typing import Any, Dict, List

from tasks import GRADERS, TASKS


SUPPLIERS = {
    "low_cost": {
        "lead_time": 4,
        "unit_cost": 7.5,
        "reliability": 0.78,
        "risk": 0.42,
        "emissions_per_unit": 0.25,
    },
    "balanced": {
        "lead_time": 3,
        "unit_cost": 9.0,
        "reliability": 0.88,
        "risk": 0.25,
        "emissions_per_unit": 0.20,
    },
    "resilient": {
        "lead_time": 2,
        "unit_cost": 11.5,
        "reliability": 0.95,
        "risk": 0.14,
        "emissions_per_unit": 0.17,
    },
}


ROUTES = {
    "economy": {"cost": 220.0, "on_time_bias": -0.10, "emissions": 35.0},
    "standard": {"cost": 330.0, "on_time_bias": 0.00, "emissions": 48.0},
    "cold_express": {"cost": 540.0, "on_time_bias": 0.10, "emissions": 62.0},
}


class ColdChainSupplyEnv:
    def __init__(self) -> None:
        self.task_id = "task_easy"
        self.step_count = 0
        self.total_reward = 0.0
        self.rng = random.Random(7)
        self.state: Dict[str, Any] = {}
        self._last_composite = 0.0

    def reset(self, task_id: str = "task_easy", seed: int = 7) -> Dict[str, Any]:
        if task_id not in TASKS:
            raise ValueError(f"unknown task_id: {task_id}")

        self.task_id = task_id
        self.step_count = 0
        self.total_reward = 0.0
        self.rng = random.Random(seed)
        self._last_composite = 0.0

        self.state = {
            "day": 1,
            "seed": seed,
            "initial_budget": 25000.0,
            "budget": 25000.0,
            "active_supplier": "balanced",
            "supplier_risk": SUPPLIERS["balanced"]["risk"],
            "cold_chain_integrity": 0.84,
            "disruption_impact": 0.0,
            "inventory": {
                "vaccine_a": 320,
                "vaccine_b": 240,
                "vaccine_c": 180,
            },
            "safety_stock": {
                "vaccine_a": 180,
                "vaccine_b": 150,
                "vaccine_c": 120,
            },
            "in_transit": [],
            "pending_orders": [],
            "total_demand_units": 0,
            "fulfilled_units": 0,
            "unmet_demand_units": 0,
            "deliveries_completed": 0,
            "deliveries_on_time": 0,
            "spoilage_units": 0,
            "units_received": 0,
            "cumulative_spend": 0.0,
            "emissions_kg": 0.0,
        }

        self._last_composite = self._composite_score(self.task_id)
        return {
            "observation": self._observe(),
            "info": {
                "task": TASKS[task_id],
                "seed": seed,
            },
        }

    def _observe(self) -> Dict[str, Any]:
        return {
            "day": self.state["day"],
            "budget": round(self.state["budget"], 2),
            "active_supplier": self.state["active_supplier"],
            "cold_chain_integrity": round(self.state["cold_chain_integrity"], 3),
            "inventory": dict(self.state["inventory"]),
            "pending_orders": list(self.state["pending_orders"]),
            "deliveries_completed": self.state["deliveries_completed"],
            "deliveries_on_time": self.state["deliveries_on_time"],
            "unmet_demand_units": self.state["unmet_demand_units"],
            "spoilage_units": self.state["spoilage_units"],
            "emissions_kg": round(self.state["emissions_kg"], 2),
        }

    def _maybe_disruption(self) -> None:
        base_prob = 0.06 + (0.12 * self.state["supplier_risk"])
        if self.rng.random() < base_prob:
            impact = round(self.rng.uniform(0.05, 0.22), 3)
            self.state["disruption_impact"] = impact
            self.state["cold_chain_integrity"] = max(
                0.45,
                self.state["cold_chain_integrity"] - impact,
            )
        else:
            self.state["disruption_impact"] = max(0.0, self.state["disruption_impact"] - 0.03)

    def _receive_shipments(self) -> None:
        arrivals: List[Dict[str, Any]] = []
        remaining: List[Dict[str, Any]] = []
        for shipment in self.state["in_transit"]:
            if shipment["arrival_day"] <= self.state["day"]:
                arrivals.append(shipment)
            else:
                remaining.append(shipment)
        self.state["in_transit"] = remaining

        for shipment in arrivals:
            qty = shipment["quantity"]
            item = shipment["item"]
            integrity_factor = self.state["cold_chain_integrity"] * shipment["supplier_reliability"]
            spoilage = int(qty * max(0.0, 1.0 - integrity_factor) * 0.35)
            accepted = max(0, qty - spoilage)

            self.state["inventory"][item] = self.state["inventory"].get(item, 0) + accepted
            self.state["spoilage_units"] += spoilage
            self.state["units_received"] += qty

    def _simulate_demand(self) -> None:
        seasonality = 1.0 + 0.25 * math.sin(self.state["day"] / 3.0)
        demand_profile = {
            "vaccine_a": int(self.rng.randint(35, 55) * seasonality),
            "vaccine_b": int(self.rng.randint(26, 42) * seasonality),
            "vaccine_c": int(self.rng.randint(18, 32) * seasonality),
        }

        backlog_next = []
        for pending in self.state["pending_orders"]:
            backlog_next.append(pending)

        for item, demand in demand_profile.items():
            available = self.state["inventory"].get(item, 0)
            fulfilled = min(available, demand)
            unmet = demand - fulfilled

            self.state["inventory"][item] = available - fulfilled
            self.state["fulfilled_units"] += fulfilled
            self.state["unmet_demand_units"] += unmet
            self.state["total_demand_units"] += demand

            if unmet > 0:
                backlog_next.append({"item": item, "units": unmet})

        self.state["pending_orders"] = backlog_next[-9:]

    def _apply_action(self, action: Dict[str, Any], info: Dict[str, Any]) -> None:
        action_type = action.get("type", "wait")

        if action_type == "procure":
            item = action.get("item", "vaccine_a")
            quantity = int(action.get("quantity", 120))
            supplier = action.get("supplier", self.state["active_supplier"])
            expedited = bool(action.get("expedited", False))

            if supplier not in SUPPLIERS:
                info["error"] = "invalid_supplier"
                return
            if quantity <= 0:
                info["error"] = "invalid_quantity"
                return

            profile = SUPPLIERS[supplier]
            lead_time = max(1, profile["lead_time"] - (1 if expedited else 0))
            multiplier = 1.28 if expedited else 1.0
            unit_cost = profile["unit_cost"] * multiplier
            total_cost = quantity * unit_cost

            if self.state["budget"] < total_cost:
                info["error"] = "insufficient_budget"
                return

            self.state["budget"] -= total_cost
            self.state["cumulative_spend"] += total_cost
            self.state["emissions_kg"] += quantity * profile["emissions_per_unit"]
            self.state["in_transit"].append(
                {
                    "item": item,
                    "quantity": quantity,
                    "arrival_day": self.state["day"] + lead_time,
                    "supplier_reliability": profile["reliability"],
                }
            )

        elif action_type == "dispatch":
            route = action.get("route", "standard")
            units = int(action.get("units", 80))
            if route not in ROUTES:
                info["error"] = "invalid_route"
                return
            if units <= 0:
                info["error"] = "invalid_units"
                return

            route_profile = ROUTES[route]
            if self.state["budget"] < route_profile["cost"]:
                info["error"] = "insufficient_budget"
                return

            self.state["budget"] -= route_profile["cost"]
            self.state["cumulative_spend"] += route_profile["cost"]
            self.state["emissions_kg"] += route_profile["emissions"]

            self.state["deliveries_completed"] += 1
            on_time_prob = min(
                0.99,
                max(
                    0.25,
                    0.62
                    + route_profile["on_time_bias"]
                    + 0.28 * self.state["cold_chain_integrity"]
                    - 0.18 * self.state["disruption_impact"],
                ),
            )
            if self.rng.random() < on_time_prob:
                self.state["deliveries_on_time"] += 1

            remaining = units
            for item in ["vaccine_a", "vaccine_b", "vaccine_c"]:
                if remaining <= 0:
                    break
                available = self.state["inventory"].get(item, 0)
                take = min(available, remaining)
                self.state["inventory"][item] = available - take
                remaining -= take

        elif action_type == "switch_supplier":
            supplier = action.get("supplier", "balanced")
            if supplier not in SUPPLIERS:
                info["error"] = "invalid_supplier"
                return
            self.state["active_supplier"] = supplier
            self.state["supplier_risk"] = SUPPLIERS[supplier]["risk"]

        elif action_type == "maintenance":
            spend = float(action.get("spend", 400.0))
            if spend <= 0:
                info["error"] = "invalid_spend"
                return
            if self.state["budget"] < spend:
                info["error"] = "insufficient_budget"
                return

            self.state["budget"] -= spend
            self.state["cumulative_spend"] += spend
            boost = min(0.12, spend / 8000.0)
            self.state["cold_chain_integrity"] = min(
                0.99,
                self.state["cold_chain_integrity"] + boost,
            )
            self.state["supplier_risk"] = max(0.08, self.state["supplier_risk"] - boost * 0.4)

        elif action_type == "wait":
            return

        else:
            info["error"] = "invalid_action_type"

    def _composite_score(self, task_id: str) -> float:
        grader = GRADERS[task_id]
        return float(grader(self.state))

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self.step_count += 1
        info: Dict[str, Any] = {}

        self._apply_action(action, info)
        self._maybe_disruption()
        self._receive_shipments()
        self._simulate_demand()

        self.state["cold_chain_integrity"] = max(0.40, self.state["cold_chain_integrity"] - 0.01)
        self.state["day"] += 1

        composite = self._composite_score(self.task_id)
        reward = round(composite - self._last_composite, 3)
        self._last_composite = composite
        self.total_reward += reward

        max_steps = int(TASKS[self.task_id]["max_steps"])
        done = self.step_count >= max_steps or self.state["budget"] <= 0

        if done:
            info["final_score"] = round(self._composite_score(self.task_id), 3)
            info["grader"] = self.task_id

        return {
            "observation": self._observe(),
            "reward": reward,
            "done": done,
            "info": info,
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "state": dict(self.state),
            "task_id": self.task_id,
            "step_count": self.step_count,
            "total_reward": round(self.total_reward, 3),
        }
