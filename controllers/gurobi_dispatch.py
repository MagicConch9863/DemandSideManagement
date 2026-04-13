from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import gurobipy as gp
from gurobipy import GRB


@dataclass
class DispatchConfig:
    """
    Single prosumer dispatch configuration.

    Sign convention:
        p_grid[t] > 0 : import from grid
        p_grid[t] < 0 : export to grid

        p_bat[t] > 0  : charging
        p_bat[t] < 0  : discharging
    """
    dt_hours: float = 1.0

    battery_capacity_kwh: float = 10.0
    battery_power_kw: float = 5.0

    energy_init_kwh: float = 5.0
    energy_min_kwh: float = 1.0
    energy_max_kwh: float = 10.0

    battery_cycle_cost_eur_per_kwh2: float = 0.01
    enforce_terminal_energy: bool = True

    verbose: bool = False


def _to_numpy_1d(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    return arr


def _validate_inputs(
    load_profile_kw: np.ndarray,
    pv_profile_kw: np.ndarray,
    price_profile: np.ndarray,
    config: DispatchConfig,
) -> None:
    if not (len(load_profile_kw) == len(pv_profile_kw) == len(price_profile)):
        raise ValueError("load_profile_kw, pv_profile_kw, and price_profile must have the same length.")

    if len(load_profile_kw) == 0:
        raise ValueError("Input profiles must not be empty.")

    if config.dt_hours <= 0:
        raise ValueError("dt_hours must be positive.")

    if config.battery_capacity_kwh < 0:
        raise ValueError("battery_capacity_kwh must be nonnegative.")

    if config.battery_power_kw < 0:
        raise ValueError("battery_power_kw must be nonnegative.")

    if config.energy_min_kwh > config.energy_max_kwh:
        raise ValueError("energy_min_kwh must be <= energy_max_kwh.")

    if not (config.energy_min_kwh <= config.energy_init_kwh <= config.energy_max_kwh):
        raise ValueError("energy_init_kwh must lie within [energy_min_kwh, energy_max_kwh].")


def solve_dispatch_gurobi(
    load_profile_kw,
    pv_profile_kw,
    price_profile,
    config: DispatchConfig,
    solver_time_limit: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Solve a single prosumer dispatch problem over the full horizon.

    Power balance:
        p_grid[t] = load[t] + p_bat[t] - pv[t]

    Energy dynamics:
        E[t+1] = E[t] + p_bat[t] * dt
    """
    load = _to_numpy_1d(load_profile_kw, "load_profile_kw")
    pv = _to_numpy_1d(pv_profile_kw, "pv_profile_kw")
    price = _to_numpy_1d(price_profile, "price_profile")

    _validate_inputs(load, pv, price, config)

    T = len(load)

    model = gp.Model("single_prosumer_dispatch")
    model.Params.OutputFlag = 1 if config.verbose else 0
    if solver_time_limit is not None:
        model.Params.TimeLimit = float(solver_time_limit)

    p_grid = model.addVars(T, lb=-GRB.INFINITY, name="p_grid_kw")
    p_bat = model.addVars(
        T,
        lb=-config.battery_power_kw,
        ub=config.battery_power_kw,
        name="p_bat_kw",
    )
    energy = model.addVars(
        T + 1,
        lb=config.energy_min_kwh,
        ub=config.energy_max_kwh,
        name="energy_kwh",
    )

    model.addConstr(energy[0] == config.energy_init_kwh, name="energy_init")

    for t in range(T):
        model.addConstr(
            p_grid[t] == load[t] + p_bat[t] - pv[t],
            name=f"power_balance_{t}",
        )
        model.addConstr(
            energy[t + 1] == energy[t] + p_bat[t] * config.dt_hours,
            name=f"energy_dyn_{t}",
        )

    if config.enforce_terminal_energy and config.battery_power_kw > 0.0:
        model.addConstr(
            energy[T] == config.energy_init_kwh,
            name="terminal_energy",
        )

    objective = gp.quicksum(
        price[t] * p_grid[t] * config.dt_hours
        + config.battery_cycle_cost_eur_per_kwh2 * (p_bat[t] * p_bat[t]) * config.dt_hours
        for t in range(T)
    )
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()

    status = model.Status
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
    }
    status_str = status_map.get(status, f"STATUS_{status}")

    result: Dict[str, Any] = {
        "status": status_str,
        "objective": None,
        "p_grid_kw": None,
        "p_bat_kw": None,
        "energy_kwh": None,
        "load_kw": load.copy(),
        "pv_kw": pv.copy(),
        "price": price.copy(),
    }

    if status in {GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT} and model.SolCount > 0:
        result["objective"] = float(model.ObjVal)
        result["p_grid_kw"] = np.array([p_grid[t].X for t in range(T)], dtype=float)
        result["p_bat_kw"] = np.array([p_bat[t].X for t in range(T)], dtype=float)
        result["energy_kwh"] = np.array([energy[t].X for t in range(T + 1)], dtype=float)
    else:
        if status == GRB.INFEASIBLE:
            try:
                model.computeIIS()
                model.write("dispatch_model.ilp")
            except Exception:
                pass

    return result


def solve_dispatch_from_node_data(
    node_data: Dict[str, Any],
    bus_id: int,
    leader_price,
    cfg,
    solver_time_limit: Optional[float] = None,
) -> Dict[str, Any]:
    has_battery = bool(node_data["has_battery"][bus_id])

    dispatch_cfg = DispatchConfig(
        dt_hours=cfg.time_step_hours,
        battery_capacity_kwh=node_data["battery_capacity_kwh"][bus_id] if has_battery else 0.0,
        battery_power_kw=node_data["battery_pmax_kw"][bus_id] if has_battery else 0.0,
        energy_init_kwh=node_data["energy_init_kwh"][bus_id] if has_battery else 0.0,
        energy_min_kwh=cfg.soc_min_kwh if has_battery else 0.0,
        energy_max_kwh=cfg.soc_max_kwh if has_battery else 0.0,
        battery_cycle_cost_eur_per_kwh2=cfg.battery_cycle_cost_eur_per_kwh2,
        enforce_terminal_energy=cfg.enforce_terminal_soc,
        verbose=bool(cfg.gurobi_output_flag),
    )

    return solve_dispatch_gurobi(
        load_profile_kw=node_data["load_kw"][bus_id],
        pv_profile_kw=node_data["pv_kw"][bus_id],
        price_profile=leader_price,
        config=dispatch_cfg,
        solver_time_limit=solver_time_limit,
    )