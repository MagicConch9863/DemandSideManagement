import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyomo.environ import (
    ConcreteModel, RangeSet, Var, Constraint, Objective, SolverFactory,
    minimize, value, Param, Reals
)

# ============================================================
# 0. Paths
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
if os.path.basename(project_root).lower() == "analysis":
    project_root = os.path.dirname(project_root)

data_dir_candidates = [
    os.path.join(current_dir, "data"),
    os.path.join(project_root, "data"),
]

output_dir_candidates = [
    os.path.join(current_dir, "outputs"),
    os.path.join(project_root, "outputs"),
]

DATA_DIR = next((p for p in data_dir_candidates if os.path.isdir(p)), data_dir_candidates[-1])
OUTPUT_DIR = next((p for p in output_dir_candidates if os.path.isdir(p)), output_dir_candidates[-1])

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "load_profile_forecast_18bus.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "stackelberg_results_18bus_fast.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "stackelberg_load_price_24h.png")
# ============================================================
# 1. Solver
# ============================================================
SOLVER_NAME = "gurobi"

# ============================================================
# 2. Time settings
# ============================================================
DT_HOURS = 0.25
TOTAL_HOURS = 24
TOTAL_STEPS = int(TOTAL_HOURS / DT_HOURS)   # 96
HORIZON_STEPS = 16                          # 4 hours, stronger anticipation

# Faster but more stable leader iteration
LEADER_MAX_ITER = 10
LEADER_TOL = 3e-3
LEADER_STEP_SIZE = 0.06
SPSA_DELTA = 3e-3
SPSA_DECAY = 0.96
RANDOM_SEED = 42

# ============================================================
# 3. Agent / battery settings
# ============================================================
NUM_AGENTS = 14
BATTERY_CAP_KWH = 4.0
SOC_INIT_RATIO = 0.60
SOC_MIN_RATIO = 0.20
SOC_MAX_RATIO = 0.90
Q_MIN = SOC_MIN_RATIO * BATTERY_CAP_KWH
Q_MAX = SOC_MAX_RATIO * BATTERY_CAP_KWH
Q_INIT = SOC_INIT_RATIO * BATTERY_CAP_KWH

P_CH_MAX_KW = 0.90
P_DIS_MAX_KW = 0.90
ETA_CH = 0.95
ETA_DIS = 0.95

# Lower degradation cost => batteries react to price more actively
GAMMA_VEC = np.linspace(0.0035, 0.0070, NUM_AGENTS)
PI_END_VEC = np.linspace(0.12, 0.16, NUM_AGENTS)

# ============================================================
# 4. Leader settings
# ============================================================
Y_MIN = 0.02
Y_MAX = 0.60
Y_AVG = 0.22

# Stronger system objective, milder price penalties
ALPHA_LOAD = 35.0
BETA_PRICE = 0.08
RHO_PRICE_RAMP = 2.2

# Terminal SoC soft target: keep batteries near initial SoC, but not too rigid
KAPPA_TERM = 0.12
Q_REF = Q_INIT

# Extra safety smoothing on realized first-step price
PRICE_EXEC_SMOOTH = 0.65

# ============================================================
# 5. Utility / reporting
# ============================================================
WHOLESALE_PRICE = 0.11
PRICE_DISPLAY_SCALE = 100.0


# ============================================================
# 6. Data loading
# ============================================================
def load_agent_profiles_kw(csv_file: str, total_steps: int = TOTAL_STEPS, num_agents: int = NUM_AGENTS) -> np.ndarray:
    df = pd.read_csv(csv_file)

    bus_cols = [c for c in df.columns if str(c).strip().lower().startswith("bus")]
    if not bus_cols:
        bus_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(bus_cols) < num_agents:
        raise ValueError(f"Need at least {num_agents} numeric/bus columns, found {len(bus_cols)}")

    bus_cols = bus_cols[:num_agents]
    arr_mw = df[bus_cols].astype(float).values

    if len(arr_mw) == total_steps:
        arr_out = arr_mw.copy()
    elif len(arr_mw) == 96 and total_steps == 96:
        arr_out = arr_mw.copy()
    elif len(arr_mw) == 192 and total_steps == 96:
        arr_out = arr_mw[:96].copy()
    else:
        x_old = np.linspace(0, 1, len(arr_mw))
        x_new = np.linspace(0, 1, total_steps)
        arr_out = np.zeros((total_steps, arr_mw.shape[1]))
        for j in range(arr_mw.shape[1]):
            arr_out[:, j] = np.interp(x_new, x_old, arr_mw[:, j])

    return arr_out * 1000.0  # MW -> kW


# ============================================================
# 7. Helpers
# ============================================================
def get_horizon_slice(arr: np.ndarray, t_now: int, horizon_steps: int) -> np.ndarray:
    slc = arr[t_now:t_now + horizon_steps].copy()
    if len(slc) < horizon_steps:
        last_row = slc[-1:, :].copy()
        pad = np.repeat(last_row, horizon_steps - len(slc), axis=0)
        slc = np.vstack([slc, pad])
    return slc


def project_price_vector(y: np.ndarray, y_min: float, y_max: float, y_avg: float, iters: int = 12) -> np.ndarray:
    z = np.array(y, dtype=float).copy()
    for _ in range(iters):
        z = z - (np.mean(z) - y_avg)
        z = np.clip(z, y_min, y_max)
    z = z - (np.mean(z) - y_avg)
    return np.clip(z, y_min, y_max)


def compute_reference_load(base_total_horizon_kw: np.ndarray) -> np.ndarray:
    """
    Reference load used only inside the leader objective.
    It is a smoothed target trajectory built from the forecasted baseline horizon,
    not a measured load and not an executed control signal.
    """
    s = pd.Series(base_total_horizon_kw)
    ref = s.rolling(window=6, center=True, min_periods=1).mean().values
    return ref


def leader_objective(y: np.ndarray, L_net: np.ndarray, L_ref: np.ndarray, y_avg: float) -> float:
    obj = ALPHA_LOAD * float(np.sum((L_net - L_ref) ** 2))
    obj += BETA_PRICE * float(np.sum((y - y_avg) ** 2))
    if len(y) >= 2:
        obj += RHO_PRICE_RAMP * float(np.sum(np.diff(y) ** 2))
    return obj


def utility_profit(price: float, actual_load_kw: float) -> float:
    return float((price - WHOLESALE_PRICE) * actual_load_kw * DT_HOURS)


# ============================================================
# 8. Follower QP (strict best response)
# ============================================================
def solve_follower_qp(
    price_vec: np.ndarray,
    load_vec_kw: np.ndarray,
    q0_kwh: float,
    gamma_i: float,
    pi_end_i: float,
    q_ref_kwh: float = Q_REF,
    kappa_term: float = KAPPA_TERM,
):
    T = len(price_vec)

    model = ConcreteModel()
    model.T = RangeSet(0, T - 1)

    model.price = Param(model.T, initialize={t: float(price_vec[t]) for t in range(T)})
    model.base = Param(model.T, initialize={t: float(load_vec_kw[t]) for t in range(T)})

    model.p_ch = Var(model.T, bounds=(0.0, P_CH_MAX_KW))
    model.p_dis = Var(model.T, bounds=(0.0, P_DIS_MAX_KW))
    model.q = Var(model.T, bounds=(Q_MIN, Q_MAX))
    model.s_net = Var(model.T, domain=Reals)

    def s_net_def_rule(m, t):
        return m.s_net[t] == m.p_ch[t] - m.p_dis[t]
    model.s_net_def = Constraint(model.T, rule=s_net_def_rule)

    def soc_rule(m, t):
        if t == 0:
            return m.q[t] == q0_kwh + DT_HOURS * (ETA_CH * m.p_ch[t] - m.p_dis[t] / ETA_DIS)
        return m.q[t] == m.q[t - 1] + DT_HOURS * (ETA_CH * m.p_ch[t] - m.p_dis[t] / ETA_DIS)
    model.soc_con = Constraint(model.T, rule=soc_rule)

    def obj_rule(m):
        expr = 0.0
        for t in m.T:
            expr += m.price[t] * (m.base[t] + m.s_net[t]) * DT_HOURS
            expr += gamma_i * (m.s_net[t] ** 2)
        expr -= pi_end_i * m.q[T - 1]
        expr += kappa_term * ((m.q[T - 1] - q_ref_kwh) ** 2)
        return expr

    model.obj = Objective(rule=obj_rule, sense=minimize)

    solver = SolverFactory(SOLVER_NAME)
    solver.solve(model, tee=False)

    s_net = np.array([value(model.s_net[t]) if value(model.s_net[t]) is not None else 0.0 for t in model.T], dtype=float)
    q = np.array([value(model.q[t]) if value(model.q[t]) is not None else q0_kwh for t in model.T], dtype=float)
    return s_net, q


# ============================================================
# 9. Aggregate follower responses
# ============================================================
def solve_all_followers(price_vec: np.ndarray, load_horizon_kw: np.ndarray, soc_now_kwh: np.ndarray):
    T = load_horizon_kw.shape[0]
    all_s = np.zeros((T, NUM_AGENTS), dtype=float)
    all_q = np.zeros((T, NUM_AGENTS), dtype=float)

    for i in range(NUM_AGENTS):
        s_i, q_i = solve_follower_qp(
            price_vec=price_vec,
            load_vec_kw=load_horizon_kw[:, i],
            q0_kwh=float(soc_now_kwh[i]),
            gamma_i=float(GAMMA_VEC[i]),
            pi_end_i=float(PI_END_VEC[i]),
        )
        all_s[:, i] = s_i
        all_q[:, i] = q_i

    return all_s, all_q


# ============================================================
# 10. Fast leader iterative solve with SPSA gradient
# ============================================================
def solve_leader_iterative(load_horizon_kw: np.ndarray, soc_now_kwh: np.ndarray, y_init: np.ndarray | None = None, rng=None):
    T = load_horizon_kw.shape[0]
    base_total = np.sum(load_horizon_kw, axis=1)
    L_ref = compute_reference_load(base_total)

    if y_init is None:
        y = np.ones(T, dtype=float) * Y_AVG
    else:
        y = project_price_vector(np.array(y_init, dtype=float), Y_MIN, Y_MAX, Y_AVG)

    best_y = y.copy()
    best_obj = np.inf

    for k in range(LEADER_MAX_ITER):
        s_all, _ = solve_all_followers(y, load_horizon_kw, soc_now_kwh)
        L_net = base_total + np.sum(s_all, axis=1)
        obj = leader_objective(y, L_net, L_ref, Y_AVG)

        if obj < best_obj:
            best_obj = obj
            best_y = y.copy()

        delta_sign = rng.choice([-1.0, 1.0], size=T)
        ck = SPSA_DELTA * (SPSA_DECAY ** k)

        y_plus = project_price_vector(y + ck * delta_sign, Y_MIN, Y_MAX, Y_AVG)
        s_plus, _ = solve_all_followers(y_plus, load_horizon_kw, soc_now_kwh)
        L_plus = base_total + np.sum(s_plus, axis=1)
        obj_plus = leader_objective(y_plus, L_plus, L_ref, Y_AVG)

        y_minus = project_price_vector(y - ck * delta_sign, Y_MIN, Y_MAX, Y_AVG)
        s_minus, _ = solve_all_followers(y_minus, load_horizon_kw, soc_now_kwh)
        L_minus = base_total + np.sum(s_minus, axis=1)
        obj_minus = leader_objective(y_minus, L_minus, L_ref, Y_AVG)

        grad = ((obj_plus - obj_minus) / (2.0 * ck)) * (1.0 / delta_sign)
        step = LEADER_STEP_SIZE * (0.94 ** k)
        y_new = project_price_vector(y - step * grad, Y_MIN, Y_MAX, Y_AVG)

        if np.linalg.norm(y_new - y) < LEADER_TOL:
            y = y_new
            break
        y = y_new

    s_all, q_all = solve_all_followers(best_y, load_horizon_kw, soc_now_kwh)
    L_net = base_total + np.sum(s_all, axis=1)

    return {
        "price_vec": best_y,
        "s_net_kw": s_all,
        "soc_kwh": q_all,
        "base_total_kw": base_total,
        "ref_total_kw": L_ref,
        "net_total_kw": L_net,
        "objective": best_obj,
    }


# ============================================================
# 11. Plot + metrics
# ============================================================
def print_metrics(res_df: pd.DataFrame) -> None:
    base = res_df["base_load_kw"].values
    opt = res_df["opt_load_kw"].values

    peak_base = float(np.max(base))
    peak_opt = float(np.max(opt))
    valley_base = float(np.min(base))
    valley_opt = float(np.min(opt))
    mean_base = float(np.mean(base))
    mean_opt = float(np.mean(opt))
    std_base = float(np.std(base))
    std_opt = float(np.std(opt))
    ramp_base = float(np.mean(np.abs(np.diff(base))))
    ramp_opt = float(np.mean(np.abs(np.diff(opt))))
    energy_base = float(np.sum(base) * DT_HOURS)
    energy_opt = float(np.sum(opt) * DT_HOURS)

    print("\n===== Fast Strict Stackelberg Metrics =====")
    print(f"Peak reduction (%) : {(peak_base - peak_opt) / max(peak_base, 1e-9) * 100:.4f}")
    print(f"Valley filling (%) : {(valley_opt - valley_base) / max(abs(valley_base), 1e-9) * 100:.4f}")
    print(f"Mean load delta    : {mean_opt - mean_base:.4f} kW")
    print(f"Std reduction (%)  : {(std_base - std_opt) / max(std_base, 1e-9) * 100:.4f}")
    print(f"Ramp reduction (%) : {(ramp_base - ramp_opt) / max(ramp_base, 1e-9) * 100:.4f}")
    print(f"Energy gap (%)     : {(energy_opt - energy_base) / max(energy_base, 1e-9) * 100:.4f}")


def plot_results(res_df: pd.DataFrame) -> None:
    os.makedirs(os.path.join(current_dir, "outputs"), exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(14, 6))
    t = res_df["time_step"].values

    line1, = ax1.plot(t, res_df["base_load_kw"].values, linestyle="--", color="gray", label="Baseline Load")
    line2, = ax1.plot(t, res_df["opt_load_kw"].values, linewidth=2.2, color="blue", label="Stackelberg Optimized Load")
    line3, = ax1.plot(t, res_df["ref_load_kw"].values, linewidth=2.0, color="green", label="Reference Load")
    ax1.set_xlabel("Time Step (15 min)")
    ax1.set_ylabel("Power (kW)")
    ax1.set_title("Strict Stackelberg Load and Price (24h)")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    line4, = ax2.plot(t, res_df["price_signal"].values, linewidth=1.8, color="red", alpha=0.85, label="Price Signal")
    ax2.set_ylabel("Price")

    ax1.legend(handles=[line1, line2, line3, line4], loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=200, bbox_inches="tight")
    plt.show()


# ============================================================
# 12. Main rolling Stackelberg-MPC controller
# ============================================================
def run_stackelberg_controller() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    t_start = time.time()

    agent_loads_kw = load_agent_profiles_kw(DATA_FILE, total_steps=TOTAL_STEPS, num_agents=NUM_AGENTS)
    soc_now = np.ones(NUM_AGENTS, dtype=float) * Q_INIT
    prev_price_vec = np.ones(HORIZON_STEPS, dtype=float) * Y_AVG
    prev_exec_price = Y_AVG

    history = {
        "time_step": [],
        "base_load_kw": [],
        "ref_load_kw": [],
        "opt_load_kw": [],
        "price_signal": [],
        "price_feedback_display": [],
        "battery_dispatch_kw": [],
        "utility_profit": [],
        "avg_soc_kwh": [],
        "step_runtime_s": [],
    }

    for t_now in range(TOTAL_STEPS):
        step_t0 = time.time()
        load_horizon = get_horizon_slice(agent_loads_kw, t_now, HORIZON_STEPS)
        eq = solve_leader_iterative(load_horizon, soc_now, prev_price_vec, rng=rng)

        price_vec = eq["price_vec"]
        s_all = eq["s_net_kw"]
        q_all = eq["soc_kwh"]
        base_h = eq["base_total_kw"]
        ref_h = eq["ref_total_kw"]
        net_h = eq["net_total_kw"]

        # Execute only first-step action
        s0_all = s_all[0, :]
        q0_next = q_all[0, :]
        batt_sum_0 = float(np.sum(s0_all))
        base_0 = float(base_h[0])
        ref_0 = float(ref_h[0])
        net_0 = float(net_h[0])

        # Smooth the executed price to avoid step-to-step price chattering
        raw_price_0 = float(price_vec[0])
        price_0 = PRICE_EXEC_SMOOTH * prev_exec_price + (1.0 - PRICE_EXEC_SMOOTH) * raw_price_0
        price_0 = float(np.clip(price_0, Y_MIN, Y_MAX))

        soc_now = q0_next.copy()
        prev_exec_price = price_0

        # Warm start next horizon with shifted optimized price vector
        shifted = np.roll(price_vec, -1)
        shifted[-1] = price_vec[-1]
        shifted[0] = price_0
        prev_price_vec = shifted

        step_runtime = time.time() - step_t0

        history["time_step"].append(t_now)
        history["base_load_kw"].append(base_0)
        history["ref_load_kw"].append(ref_0)
        history["opt_load_kw"].append(net_0)
        history["price_signal"].append(price_0)
        history["price_feedback_display"].append(price_0 * PRICE_DISPLAY_SCALE)
        history["battery_dispatch_kw"].append(batt_sum_0)
        history["utility_profit"].append(utility_profit(price_0, net_0))
        history["avg_soc_kwh"].append(float(np.mean(soc_now)))
        history["step_runtime_s"].append(step_runtime)

        if t_now % 8 == 0:
            print(
                f"t={t_now:03d} | base={base_0:7.3f} kW | ref={ref_0:7.3f} kW | "
                f"opt={net_0:7.3f} kW | batt={batt_sum_0:7.3f} kW | "
                f"price={price_0:6.3f} | step={step_runtime:5.2f}s"
            )

    res_df = pd.DataFrame(history)
    res_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    total_runtime = time.time() - t_start
    print(f"\n>>> Results saved to: {OUTPUT_FILE}")
    print(f">>> Plot saved to   : {OUTPUT_PLOT}")
    print(f">>> Total runtime   : {total_runtime:.2f} s")
    plot_results(res_df)
    print_metrics(res_df)
    return res_df


if __name__ == "__main__":
    run_stackelberg_controller()
