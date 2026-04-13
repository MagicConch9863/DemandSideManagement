from __future__ import annotations

import os
from typing import Dict, Any

import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.stackelberg_config import StackelbergConfig
from networks.plot_network_layout import plot_network_layout


def _cfg(cfg, name: str, default):
    return getattr(cfg, name, default)


def get_raw_total_demand(node_data: Dict[str, Any], horizon: int) -> np.ndarray:
    if "raw_total_load_kw" in node_data:
        return np.asarray(node_data["raw_total_load_kw"], dtype=float)

    total = np.zeros(horizon, dtype=float)
    for bus_id in node_data["bus_ids"]:
        total += np.asarray(node_data["load_kw"][bus_id], dtype=float)
    return total


def solve_window(
    raw_load: np.ndarray,
    battery_buses: list[int],
    current_soc: Dict[int, float],
    pmax_map: Dict[int, float],
    cfg: StackelbergConfig,
    start: int,
    H: int,
    prev_total: float | None,
    lref: float,
):
    """
    Simplified RHG:
        min sum [
            w_track * (P_total - Lref)^2
          + w_ramp  * (P_total - P_total_prev)^2
          + w_u     * sum_i u_i^2
        ]
    """
    dt = cfg.time_step_hours
    alpha = _cfg(cfg, "rhg_alpha", 0.999)
    beta = _cfg(cfg, "rhg_beta", 0.95)

    w_track = _cfg(cfg, "rhg_weight_track", 30.0)
    w_ramp = _cfg(cfg, "rhg_weight_ramp", 20.0)
    w_u = _cfg(cfg, "rhg_weight_u", 0.05)

    m = gp.Model("rhg_window")
    m.Params.OutputFlag = cfg.gurobi_output_flag

    u = {}
    q = {}

    for b in battery_buses:
        pmax = pmax_map[b]
        q0 = current_soc[b]

        for k in range(H):
            u[b, k] = m.addVar(lb=-pmax, ub=pmax, name=f"u_{b}_{k}")
        for k in range(H + 1):
            q[b, k] = m.addVar(lb=cfg.soc_min_kwh, ub=cfg.soc_max_kwh, name=f"q_{b}_{k}")

        m.addConstr(q[b, 0] == q0, name=f"q_init_{b}")

        for k in range(H):
            m.addConstr(
                q[b, k + 1] == alpha * q[b, k] + dt * beta * u[b, k],
                name=f"q_dyn_{b}_{k}",
            )

    total_expr = []
    for k in range(H):
        t = min(start + k, len(raw_load) - 1)
        expr = raw_load[t] + gp.quicksum(u[b, k] for b in battery_buses)
        total_expr.append(expr)

    obj = 0.0
    for k in range(H):
        obj += w_track * (total_expr[k] - lref) * (total_expr[k] - lref)

        if k == 0:
            if prev_total is not None:
                obj += w_ramp * (total_expr[k] - prev_total) * (total_expr[k] - prev_total)
        else:
            obj += w_ramp * (total_expr[k] - total_expr[k - 1]) * (total_expr[k] - total_expr[k - 1])

        obj += w_u * gp.quicksum(u[b, k] * u[b, k] for b in battery_buses)

    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"RHG optimization failed at window start {start}, status={m.Status}")

    u0 = {b: float(u[b, 0].X) for b in battery_buses}
    q1 = {b: float(q[b, 1].X) for b in battery_buses}
    total0 = float(total_expr[0].getValue())

    return u0, q1, total0


def run_rhg_controller(node_data: Dict[str, Any], cfg: StackelbergConfig):
    horizon = cfg.horizon
    H = int(_cfg(cfg, "rhg_window_steps", 24))

    raw_load = get_raw_total_demand(node_data, horizon)
    lref = float(np.mean(raw_load))

    battery_buses = [b for b in node_data["bus_ids"] if node_data["has_battery"][b]]
    current_soc = {b: float(node_data["energy_init_kwh"][b]) for b in battery_buses}
    pmax_map = {b: float(node_data["battery_pmax_kw"][b]) for b in battery_buses}

    optimized_load = []
    battery_sum = []
    prev_total = None

    for t in range(horizon):
        u0, q1, total0 = solve_window(
            raw_load=raw_load,
            battery_buses=battery_buses,
            current_soc=current_soc,
            pmax_map=pmax_map,
            cfg=cfg,
            start=t,
            H=min(H, horizon - t),
            prev_total=prev_total,
            lref=lref,
        )

        batt_sum_t = 0.0
        for b in battery_buses:
            current_soc[b] = q1[b]
            batt_sum_t += u0[b]

        optimized_load.append(total0)
        battery_sum.append(batt_sum_t)
        prev_total = total0

    return raw_load, np.asarray(optimized_load, dtype=float), np.asarray(battery_sum, dtype=float), lref


def compute_metrics(raw_load: np.ndarray, opt_load: np.ndarray, lref: float, cfg: StackelbergConfig):
    peak_before = float(np.max(raw_load))
    peak_after = float(np.max(opt_load))

    valley_before = float(np.min(raw_load))
    valley_after = float(np.min(opt_load))

    std_before = float(np.std(raw_load))
    std_after = float(np.std(opt_load))

    peak_shaving = 0.0 if peak_before < 1e-9 else 100.0 * (peak_before - peak_after) / peak_before
    valley_filling = 0.0 if abs(valley_before) < 1e-9 else 100.0 * (valley_after - valley_before) / abs(valley_before)
    fluct_red = 0.0 if std_before < 1e-9 else 100.0 * (std_before - std_after) / std_before

    e_before = float(np.sum(raw_load) * cfg.time_step_hours)
    e_after = float(np.sum(opt_load) * cfg.time_step_hours)
    e_diff_pct = 0.0 if abs(e_before) < 1e-9 else 100.0 * (e_after - e_before) / e_before

    mse_before = float(np.mean((raw_load - lref) ** 2))
    mse_after = float(np.mean((opt_load - lref) ** 2))

    return {
        "lref_kw": lref,
        "peak_before_kw": peak_before,
        "peak_after_kw": peak_after,
        "peak_shaving_pct": peak_shaving,
        "valley_before_kw": valley_before,
        "valley_after_kw": valley_after,
        "valley_filling_pct": valley_filling,
        "std_before_kw": std_before,
        "std_after_kw": std_after,
        "fluctuation_reduction_pct": fluct_red,
        "energy_diff_pct": e_diff_pct,
        "mse_before_to_lref": mse_before,
        "mse_after_to_lref": mse_after,
    }


def save_outputs(raw_load: np.ndarray, opt_load: np.ndarray, batt_sum: np.ndarray, metrics: dict, cfg: StackelbergConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    pd.DataFrame(
        {
            "step": np.arange(cfg.horizon),
            "raw_simbench_demand_kw": raw_load,
            "rhg_optimized_load_kw": opt_load,
            "battery_sum_kw": batt_sum,
        }
    ).to_csv(os.path.join(cfg.output_dir, "rhg_summary.csv"), index=False)

    pd.DataFrame([metrics]).to_csv(
        os.path.join(cfg.output_dir, "rhg_metrics_summary.csv"),
        index=False,
    )


def plot_result(raw_load: np.ndarray, opt_load: np.ndarray, metrics: dict, cfg: StackelbergConfig):
    t = np.arange(cfg.horizon)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(t, raw_load, color="gray", linewidth=1.5, alpha=0.85, label="Raw SimBench demand")
    ax.plot(t, opt_load, color="#d62728", linewidth=2.8, label="RHG optimized load")

    ax.set_title("RHG Result: Raw Demand and Optimized Load")
    ax.set_xlabel("15-min step")
    ax.set_ylabel("Power (kW)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    txt = (
        f"Lref: {metrics['lref_kw']:.2f} kW\n"
        f"Peak shaving: {metrics['peak_shaving_pct']:.1f}%\n"
        f"Valley filling: {metrics['valley_filling_pct']:.1f}%\n"
        f"Fluctuation reduction: {metrics['fluctuation_reduction_pct']:.1f}%\n"
        f"Daily energy diff: {metrics['energy_diff_pct']:.2f}%\n"
        f"MSE to Lref before: {metrics['mse_before_to_lref']:.2f}\n"
        f"MSE to Lref after: {metrics['mse_after_to_lref']:.2f}"
    )
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        fontsize=10,
    )

    plt.tight_layout()
    if cfg.save_plots:
        os.makedirs(cfg.output_dir, exist_ok=True)
        fig.savefig(os.path.join(cfg.output_dir, "rhg_main_result.png"), dpi=250, bbox_inches="tight")
    plt.show()


def run_rhg_simulation(cfg: StackelbergConfig):
    cfg.validate()

    _, _, _, node_data, _, _ = plot_network_layout(
        cfg=cfg,
        save_path=os.path.join(cfg.output_dir, "network_layout.png"),
        show_plot=False,
    )

    raw_load, opt_load, batt_sum, lref = run_rhg_controller(node_data, cfg)
    m = compute_metrics(raw_load, opt_load, lref, cfg)

    save_outputs(raw_load, opt_load, batt_sum, m, cfg)

    print("\n" + "=" * 60)
    print("RHG simulation summary")
    print("=" * 60)
    print(f"Window steps            : {_cfg(cfg, 'rhg_window_steps', 24)}")
    print(f"Lref                    : {m['lref_kw']:.3f} kW")
    print(f"Peak shaving            : {m['peak_shaving_pct']:.2f}%")
    print(f"Valley filling          : {m['valley_filling_pct']:.2f}%")
    print(f"Fluctuation reduction   : {m['fluctuation_reduction_pct']:.2f}%")
    print(f"Daily energy difference : {m['energy_diff_pct']:.3f}%")
    print(f"MSE to Lref before      : {m['mse_before_to_lref']:.3f}")
    print(f"MSE to Lref after       : {m['mse_after_to_lref']:.3f}")
    print("=" * 60)

    plot_result(raw_load, opt_load, m, cfg)

    return {
        "raw_load": raw_load,
        "optimized_load": opt_load,
        "metrics": m,
    }