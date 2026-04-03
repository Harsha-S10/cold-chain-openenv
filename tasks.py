from typing import Any, Dict


TASKS: Dict[str, Dict[str, Any]] = {
    "task_easy": {
        "name": "Clinic Stock Stability",
        "description": "Keep vaccine stock above safety levels while controlling basic procurement cost.",
        "max_steps": 12,
        "reward_weights": {
            "service_level": 0.45,
            "inventory_health": 0.35,
            "cost_efficiency": 0.20,
        },
    },
    "task_medium": {
        "name": "Regional Routing Efficiency",
        "description": "Balance dispatch route decisions, timeliness, and spend under variable demand.",
        "max_steps": 16,
        "reward_weights": {
            "service_level": 0.35,
            "on_time_rate": 0.35,
            "cost_efficiency": 0.20,
            "cold_chain_integrity": 0.10,
        },
    },
    "task_hard": {
        "name": "Resilient Cold-Chain Optimization",
        "description": "Optimize service, spoilage, disruptions, risk, and emissions simultaneously.",
        "max_steps": 22,
        "reward_weights": {
            "service_level": 0.25,
            "on_time_rate": 0.20,
            "cost_efficiency": 0.15,
            "cold_chain_integrity": 0.15,
            "waste_control": 0.15,
            "resilience": 0.10,
        },
    },
}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _service_level(state: Dict[str, Any]) -> float:
    demand = float(state.get("total_demand_units", 0.0))
    unmet = float(state.get("unmet_demand_units", 0.0))
    return _clip01(1.0 - (unmet / max(demand, 1.0)))


def _inventory_health(state: Dict[str, Any]) -> float:
    inventory = state.get("inventory", {})
    safety = state.get("safety_stock", {})
    if not inventory:
        return 0.0
    healthy = 0
    for item, units in inventory.items():
        if units >= safety.get(item, 0):
            healthy += 1
    return _clip01(healthy / max(len(inventory), 1))


def _cost_efficiency(state: Dict[str, Any]) -> float:
    budget = float(state.get("initial_budget", 1.0))
    spend = float(state.get("cumulative_spend", 0.0))
    return _clip01(1.0 - (spend / max(budget, 1.0)))


def _on_time_rate(state: Dict[str, Any]) -> float:
    total = float(state.get("deliveries_completed", 0.0))
    on_time = float(state.get("deliveries_on_time", 0.0))
    return _clip01(on_time / max(total, 1.0))


def _waste_control(state: Dict[str, Any]) -> float:
    received = float(state.get("units_received", 0.0))
    spoiled = float(state.get("spoilage_units", 0.0))
    return _clip01(1.0 - (spoiled / max(received, 1.0)))


def _resilience(state: Dict[str, Any]) -> float:
    integrity = float(state.get("cold_chain_integrity", 0.0))
    supplier_risk = float(state.get("supplier_risk", 1.0))
    disruption = float(state.get("disruption_impact", 1.0))
    return _clip01((0.5 * integrity) + (0.3 * (1.0 - supplier_risk)) + (0.2 * (1.0 - disruption)))


def _blend(weights: Dict[str, float], metrics: Dict[str, float]) -> float:
    total = 0.0
    for key, weight in weights.items():
        total += weight * metrics.get(key, 0.0)
    return _clip01(total)


def grade_easy(state: Dict[str, Any]) -> float:
    metrics = {
        "service_level": _service_level(state),
        "inventory_health": _inventory_health(state),
        "cost_efficiency": _cost_efficiency(state),
    }
    return round(_blend(TASKS["task_easy"]["reward_weights"], metrics), 3)


def grade_medium(state: Dict[str, Any]) -> float:
    metrics = {
        "service_level": _service_level(state),
        "on_time_rate": _on_time_rate(state),
        "cost_efficiency": _cost_efficiency(state),
        "cold_chain_integrity": _clip01(float(state.get("cold_chain_integrity", 0.0))),
    }
    return round(_blend(TASKS["task_medium"]["reward_weights"], metrics), 3)


def grade_hard(state: Dict[str, Any]) -> float:
    metrics = {
        "service_level": _service_level(state),
        "on_time_rate": _on_time_rate(state),
        "cost_efficiency": _cost_efficiency(state),
        "cold_chain_integrity": _clip01(float(state.get("cold_chain_integrity", 0.0))),
        "waste_control": _waste_control(state),
        "resilience": _resilience(state),
    }
    return round(_blend(TASKS["task_hard"]["reward_weights"], metrics), 3)


GRADERS = {
    "task_easy": grade_easy,
    "task_medium": grade_medium,
    "task_hard": grade_hard,
}
