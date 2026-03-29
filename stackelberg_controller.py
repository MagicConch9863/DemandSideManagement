import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyomo.environ import ConcreteModel, RangeSet, Var, Constraint, Objective, SolverFactory, value, minimize

# ==========================================
# 0. Paths
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if os.path.basename(project_root).lower() == "analysis":
    project_root = os.path.dirname(project_root)

DATA_DIR = os.path.join(project_root, "data")
OUTPUT_DIR = os.path.join(project_root, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "load_profile_forecast_18bus.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "stackelberg_results_18bus_fast.csv")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "stackelberg_load_price_48h.png")

# ==========================================
# 1. Core settings
# ==========================================
SOLVER_NAME = "gurobi"
DT = 0.25                         # 15 min
TOTAL_SIM_STEPS = 192             # 48 h
NUM_AGENTS = 14
HORIZON_N = 12                    # keep your original horizon

# ==========================================
# 2. ESS settings
# Based on your original structure, but scaled down to avoid blow-ups.
# ==========================================
ALPHA = 0.999
BETA = 0.95
S_MAX = 0.0020                    # MW, reduced from 0.015 to avoid unrealistic spikes
BATTERY_CAP = 0.0060              # MWh, reduced and matched to power scale
SOC_INIT = 0.0036                 # 60%
SOC_MIN = 0.0012                  # 20%
SOC_MAX = 0.0054                  # 90%

# ==========================================
# 3. Leader pricing settings
# Keep your original profit-search structure.
# ==========================================
C1_COST = 3.5
BASE_PRICE = 0.10
RHO_CANDIDATES = np.linspace(1, 100, 15)
MGMT_PENALTY_COEF = 0.000005
MGMT_PENALTY_EXP = 2.2

# Small additions only, to stabilize behavior
TRACK_WEIGHT = 120.0              # push optimized load toward smoothed reference
ENERGY_BAL_WEIGHT = 2200.0        # penalize horizon net charging/discharging imbalance
SOC_BAL_WEIGHT = 4500.0           # penalize terminal SoC drift
PRICE_RAMP_WEIGHT = 0.010         # smooth rho changes over time
PRICE_SMOOTH_ALPHA = 0.55         # smooth executed price, but preserve visible variation
REF_WINDOW = 12                   # 3h smoothed baseline reference
MIN_NET_LOAD = 0.005              # MW floor
MAX_PRICE_RATE = 4.0              # clamp final electricity rate inside follower objective


# ==========================================
# 4. Helpers
# ==========================================
def load_forecast_df(csv_file: str, total_steps: int = TOTAL_SIM_STEPS, num_agents: int = NUM_AGENTS) -> pd.DataFrame:
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Data file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    bus_cols = [c for c in df.columns if str(c).strip().lower().startswith("bus")]
    if not bus_cols:
        bus_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(bus_cols) < num_agents:
        raise ValueError(f"Need at least {num_agents} bus/numeric columns, found {len(bus_cols)}")

    df = df[bus_cols[:num_agents]].copy()

    if len(df) == total_steps:
        out = df.copy().reset_index(drop=True)
    elif len(df) == 96 and total_steps == 192:
        out = pd.concat([df, df], ignore_index=True)
    elif len(df) == 192 and total_steps == 96:
        out = df.iloc[:96].copy().reset_index(drop=True)
    else:
        x_old = np.linspace(0, 1, len(df))
        x_new = np.linspace(0, 1, total_steps)
        arr = np.zeros((total_steps, df.shape[1]))
        for j, col in enumerate(df.columns):
            arr[:, j] = np.interp(x_new, x_old, df[col].values)
        out = pd.DataFrame(arr, columns=df.columns)

    return out


def get_horizon_slice(df: pd.DataFrame, t_now: int, horizon_n: int) -> pd.DataFrame:
    slice_df = df.iloc[t_now:t_now + horizon_n].copy()
    if len(slice_df) < horizon_n:
        last_row = slice_df.iloc[[-1]].copy()
        pad_count = horizon_n - len(slice_df)
        pad_df = pd.concat([last_row] * pad_count, ignore_index=True)
        slice_df = pd.concat([slice_df.reset_index(drop=True), pad_df], ignore_index=True)
    return slice_df


def compute_reference_series(total_load_kw: np.ndarray, window: int = REF_WINDOW) -> np.ndarray:
    s = pd.Series(total_load_kw)
    return s.rolling(window=window, center=True, min_periods=1).mean().values


# ==========================================
# 5. Prosumer response
# Keep the original form, add only terminal SoC return pressure.
# ==========================================
def solve_prosumer_response(current_soc, forecast_slice, rho1_price, ref_slice_mw, num_agents, N):
    model = ConcreteModel()
    model.I = RangeSet(0, num_agents - 1)
    model.T = RangeSet(0, N - 1)
    model.s = Var(model.I, model.T, bounds=(-S_MAX, S_MAX))
    model.q = Var(model.I, model.T, bounds=(SOC_MIN, SOC_MAX))

    def soc_con(m, i, t):
        if t == 0:
            return m.q[i, t] == current_soc[i] + DT * BETA * m.s[i, t]
        return m.q[i, t] == ALPHA * m.q[i, t - 1] + DT * BETA * m.s[i, t]
    model.soc_limit = Constraint(model.I, model.T, rule=soc_con)

    def obj_rule(m):
        total_bill = 0.0
        for t in m.T:
            base_load = float(forecast_slice.iloc[t].sum())
            net_load = base_load + sum(m.s[i, t] for i in m.I)

            # keep your original idea: price rises with system load,
            # but center it around the smoothed reference so it does not only push downward.
            deviation = net_load - float(ref_slice_mw[t])
            current_rate = BASE_PRICE + rho1_price * deviation
            current_rate = min(MAX_PRICE_RATE, max(0.01, current_rate))

            total_bill += current_rate * net_load
            total_bill += 0.01 * sum(m.s[i, t] ** 2 for i in m.I)

        # terminal SoC soft constraint for energy balance
        for i in m.I:
            total_bill += SOC_BAL_WEIGHT * ((m.q[i, N - 1] - current_soc[i]) ** 2)

        return total_bill

    model.obj = Objective(rule=obj_rule, sense=minimize)
    SolverFactory(SOLVER_NAME).solve(model, tee=False)

    decisions0 = [float(value(model.s[i, 0]) or 0.0) for i in model.I]
    soc_next = [float(value(model.q[i, 0]) or current_soc[i]) for i in model.I]

    horizon_dispatch = np.zeros(N)
    terminal_soc_gap = 0.0
    for t in range(N):
        horizon_dispatch[t] = sum(float(value(model.s[i, t]) or 0.0) for i in model.I)
    for i in range(num_agents):
        q_end = float(value(model.q[i, N - 1]) or current_soc[i])
        terminal_soc_gap += (q_end - current_soc[i]) ** 2

    return decisions0, soc_next, horizon_dispatch, terminal_soc_gap


# ==========================================
# 6. Leader search
# Keep your original dense search, but store the best actual load correctly
# and add only minimal physical penalties.
# ==========================================
def choose_best_action(current_soc, horizon_slice, ref_slice_kw, prev_rho1):
    ref_slice_mw = ref_slice_kw / 1000.0
    best = None

    for trial_rho1 in RHO_CANDIDATES:
        decisions0, soc_next, horizon_dispatch, terminal_soc_gap = solve_prosumer_response(
            current_soc=current_soc,
            forecast_slice=horizon_slice,
            rho1_price=float(trial_rho1),
            ref_slice_mw=ref_slice_mw,
            num_agents=NUM_AGENTS,
            N=HORIZON_N,
        )

        base_horizon = horizon_slice.sum(axis=1).values.astype(float)
        opt_horizon = np.maximum(MIN_NET_LOAD, base_horizon + horizon_dispatch)
        actual_l = float(opt_horizon[0])

        revenue = (trial_rho1 * max(actual_l - ref_slice_mw[0], 0.0) + BASE_PRICE) * actual_l
        purchase_cost = C1_COST * actual_l ** 2
        management_penalty = MGMT_PENALTY_COEF * (trial_rho1 ** MGMT_PENALTY_EXP)

        track_penalty = TRACK_WEIGHT * np.sum((opt_horizon - ref_slice_mw) ** 2)
        energy_gap = float(np.sum(horizon_dispatch) * DT)
        energy_balance_penalty = ENERGY_BAL_WEIGHT * (energy_gap ** 2)
        soc_balance_penalty = SOC_BAL_WEIGHT * terminal_soc_gap
        price_ramp_penalty = PRICE_RAMP_WEIGHT * ((trial_rho1 - prev_rho1) ** 2)

        score = -(revenue - purchase_cost - management_penalty) + track_penalty + energy_balance_penalty + soc_balance_penalty + price_ramp_penalty

        if best is None or score < best["score"]:
            best = {
                "score": float(score),
                "rho1": float(trial_rho1),
                "decisions0": decisions0,
                "soc_next": soc_next,
                "actual_l": actual_l,
                "energy_gap": energy_gap,
                "terminal_soc_gap": terminal_soc_gap,
                "profit": float(revenue - purchase_cost - management_penalty),
            }

    return best


# ==========================================
# 7. Main simulation
# ==========================================
def run_stackelberg_controller() -> pd.DataFrame:
    t0_all = time.time()
    forecast_df = load_forecast_df(DATA_FILE, total_steps=TOTAL_SIM_STEPS, num_agents=NUM_AGENTS)
    total_load_kw = forecast_df.sum(axis=1).values.astype(float) * 1000.0
    ref_series_kw = compute_reference_series(total_load_kw, window=REF_WINDOW)

    soc_tracker = [SOC_INIT] * NUM_AGENTS
    prev_rho1 = float(RHO_CANDIDATES[0])
    prev_exec_price = BASE_PRICE

    history = {
        "time_step": [],
        "base_load_kw": [],
        "ref_load_kw": [],
        "opt_load_kw": [],
        "price_signal": [],
        "price_gain": [],
        "battery_dispatch_kw": [],
        "utility_profit": [],
        "horizon_energy_gap_kwh": [],
        "terminal_soc_gap": [],
        "avg_soc_kwh": [],
        "step_runtime_s": [],
    }

    print(">>> Running Stackelberg controller from the original dynamic-price-search template...")

    for t_now in range(TOTAL_SIM_STEPS):
        step_t0 = time.time()
        horizon_slice = get_horizon_slice(forecast_df, t_now, HORIZON_N)
        ref_slice_kw = get_horizon_slice(pd.DataFrame({"ref": ref_series_kw}), t_now, HORIZON_N)["ref"].values

        best = choose_best_action(
            current_soc=soc_tracker,
            horizon_slice=horizon_slice,
            ref_slice_kw=ref_slice_kw,
            prev_rho1=prev_rho1,
        )

        for i in range(NUM_AGENTS):
            soc_tracker[i] = float(np.clip(best["soc_next"][i], SOC_MIN, SOC_MAX))

        # keep visible price variation but avoid jagged chattering
        raw_exec_price = BASE_PRICE + 0.001 * best["rho1"]
        exec_price = PRICE_SMOOTH_ALPHA * raw_exec_price + (1.0 - PRICE_SMOOTH_ALPHA) * prev_exec_price

        base_kw = float(total_load_kw[t_now])
        opt_kw = float(best["actual_l"] * 1000.0)
        ref_kw = float(ref_series_kw[t_now])
        batt_kw = opt_kw - base_kw
        avg_soc_kwh = float(np.mean(soc_tracker) * 1000.0)
        step_runtime = time.time() - step_t0

        history["time_step"].append(t_now)
        history["base_load_kw"].append(base_kw)
        history["ref_load_kw"].append(ref_kw)
        history["opt_load_kw"].append(opt_kw)
        history["price_signal"].append(exec_price)
        history["price_gain"].append(best["rho1"])
        history["battery_dispatch_kw"].append(batt_kw)
        history["utility_profit"].append(best["profit"])
        history["horizon_energy_gap_kwh"].append(best["energy_gap"] * 1000.0)
        history["terminal_soc_gap"].append(best["terminal_soc_gap"])
        history["avg_soc_kwh"].append(avg_soc_kwh)
        history["step_runtime_s"].append(step_runtime)

        prev_rho1 = best["rho1"]
        prev_exec_price = exec_price

        if t_now % 12 == 0:
            print(
                f"t={t_now:03d} | base={base_kw:7.3f} kW | ref={ref_kw:7.3f} kW | "
                f"opt={opt_kw:7.3f} kW | batt={batt_kw:7.3f} kW | price={exec_price:6.3f} | "
                f"rho={best['rho1']:5.1f} | Egap={best['energy_gap'] * 1000:6.3f} kWh"
            )

    res_df = pd.DataFrame(history)
    res_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n>>> Results saved to: {OUTPUT_CSV}")
    print(f">>> Plot saved to   : {OUTPUT_PNG}")
    print(f">>> Total runtime   : {time.time() - t0_all:.2f} s")

    plot_results(res_df)
    print_metrics(res_df)
    return res_df


# ==========================================
# 8. Plot and metrics
# ==========================================
def plot_results(res_df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(14, 6))
    t = res_df["time_step"].values

    l1, = ax1.plot(t, res_df["base_load_kw"].values, linestyle="--", color="gray", label="Baseline Load")
    l2, = ax1.plot(t, res_df["opt_load_kw"].values, linewidth=2.5, color="blue", label="Stackelberg Optimized Load")
    l3, = ax1.plot(t, res_df["ref_load_kw"].values, linewidth=2.0, color="green", label="Reference Load")
    ax1.set_xlabel("Time Step (15 min)")
    ax1.set_ylabel("Power (kW)")
    ax1.set_title("Stackelberg Load and Dynamic Price (48h)")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    l4, = ax2.plot(t, res_df["price_signal"].values, linewidth=1.8, color="red", label="Price Signal")
    ax2.set_ylabel("Price")

    ax1.legend(handles=[l1, l2, l3, l4], loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.show()


def print_metrics(res_df: pd.DataFrame) -> None:
    base = res_df["base_load_kw"].values
    opt = res_df["opt_load_kw"].values
    hgap = res_df["horizon_energy_gap_kwh"].values

    peak_base = float(np.max(base))
    peak_opt = float(np.max(opt))
    valley_base = float(np.min(base))
    valley_opt = float(np.min(opt))
    std_base = float(np.std(base))
    std_opt = float(np.std(opt))
    ramp_base = float(np.mean(np.abs(np.diff(base))))
    ramp_opt = float(np.mean(np.abs(np.diff(opt))))
    energy_base = float(np.sum(base) * DT)
    energy_opt = float(np.sum(opt) * DT)

    print("\n===== Stackelberg Metrics =====")
    print(f"Peak reduction (%)     : {(peak_base - peak_opt) / max(peak_base, 1e-9) * 100:.4f}")
    print(f"Valley filling (%)     : {(valley_opt - valley_base) / max(abs(valley_base), 1e-9) * 100:.4f}")
    print(f"Std reduction (%)      : {(std_base - std_opt) / max(std_base, 1e-9) * 100:.4f}")
    print(f"Ramp reduction (%)     : {(ramp_base - ramp_opt) / max(ramp_base, 1e-9) * 100:.4f}")
    print(f"Total energy gap (%)   : {(energy_opt - energy_base) / max(energy_base, 1e-9) * 100:.4f}")
    print(f"Avg horizon gap (kWh)  : {np.mean(np.abs(hgap)):.4f}")
    print(f"Max horizon gap (kWh)  : {np.max(np.abs(hgap)):.4f}")


if __name__ == "__main__":
    run_stackelberg_controller()
