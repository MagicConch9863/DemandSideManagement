from dataclasses import dataclass
from typing import Dict, Any, List

import gurobipy as gp
from gurobipy import GRB


@dataclass
class ProsumerResult:
    p_grid_kw: List[float]
    p_bat_kw: List[float]
    energy_kwh: List[float]
    objective_value: float


def solve_prosumer_problem(
    load_kw,
    pv_kw,
    lambda_price_eur_per_kwh,
    battery_capacity_kwh,
    battery_pmax_kw,
    energy_init_kwh,
    energy_min_kwh,
    energy_max_kwh,
    dt_hours,
    battery_cycle_cost_eur_per_kwh2=0.01,
    solver_output_flag=0,
    enforce_terminal_soc=True,
):
    T = len(load_kw)
    if not (len(pv_kw) == len(lambda_price_eur_per_kwh) == T):
        raise ValueError("Input lengths must match.")

    model = gp.Model("prosumer_problem")
    model.Params.OutputFlag = solver_output_flag

    p_grid = model.addVars(T, lb=-GRB.INFINITY, name="p_grid_kw")
    p_bat = model.addVars(T, lb=-battery_pmax_kw, ub=battery_pmax_kw, name="p_bat_kw")
    energy = model.addVars(T + 1, lb=energy_min_kwh, ub=energy_max_kwh, name="energy_kwh")

    model.addConstr(energy[0] == energy_init_kwh, name="energy_init")

    for t in range(T):
        model.addConstr(
            p_grid[t] == load_kw[t] + p_bat[t] - pv_kw[t],
            name=f"power_balance_{t}",
        )
        model.addConstr(
            energy[t + 1] == energy[t] + p_bat[t] * dt_hours,
            name=f"energy_dyn_{t}",
        )

    if enforce_terminal_soc and battery_pmax_kw > 0.0:
        model.addConstr(energy[T] == energy_init_kwh, name="terminal_energy")

    objective = gp.quicksum(
        lambda_price_eur_per_kwh[t] * p_grid[t] * dt_hours
        + battery_cycle_cost_eur_per_kwh2 * (p_bat[t] * p_bat[t]) * dt_hours
        for t in range(T)
    )

    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Prosumer optimization failed. Status={model.Status}")

    return ProsumerResult(
        p_grid_kw=[p_grid[t].X for t in range(T)],
        p_bat_kw=[p_bat[t].X for t in range(T)],
        energy_kwh=[energy[t].X for t in range(T + 1)],
        objective_value=float(model.ObjVal),
    )


def solve_all_prosumers(node_data: Dict[str, Any], leader_price, cfg) -> Dict[int, ProsumerResult]:
    results = {}

    for bus_id in node_data["bus_ids"]:
        has_battery = node_data["has_battery"][bus_id]

        result = solve_prosumer_problem(
            load_kw=node_data["load_kw"][bus_id],
            pv_kw=node_data["pv_kw"][bus_id],
            lambda_price_eur_per_kwh=leader_price,
            battery_capacity_kwh=node_data["battery_capacity_kwh"][bus_id] if has_battery else 0.0,
            battery_pmax_kw=node_data["battery_pmax_kw"][bus_id] if has_battery else 0.0,
            energy_init_kwh=node_data["energy_init_kwh"][bus_id] if has_battery else 0.0,
            energy_min_kwh=cfg.soc_min_kwh if has_battery else 0.0,
            energy_max_kwh=cfg.soc_max_kwh if has_battery else 0.0,
            dt_hours=cfg.time_step_hours,
            battery_cycle_cost_eur_per_kwh2=cfg.battery_cycle_cost_eur_per_kwh2,
            solver_output_flag=cfg.gurobi_output_flag,
            enforce_terminal_soc=cfg.enforce_terminal_soc,
        )
        results[bus_id] = result

    return results