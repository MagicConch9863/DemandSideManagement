import pandapower as pp
import pandapower.networks as nw
from pyomo.environ import *
import pandas as pd
import numpy as np
import os
import sys

# ==========================================
# 1. Konfigurationsparameter
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(current_dir, "data", "load_profile_forecast_18bus.csv")

HORIZON_N = 32         # 预测窗口：32个15min = 8小时
NUM_AGENTS = 14        # R1-R14 sind aktive Prosumer
SOLVER_NAME = "gurobi"

# 扩展为 48h = 192 个时刻
TOTAL_SIM_STEPS = 192

# Physikalische Batteriekonstanten
BATTERY_CAP = 0.015    # MWh
SOC_INIT = 0.0075      # MWh, initial 50%
ALPHA = 0.999
BETA = 0.95
S_MAX = 0.015          # MW

# ==========================================
# 2. 数据预处理：把原始负荷扩展到192点
# ==========================================
def prepare_forecast_48h(df: pd.DataFrame, total_steps: int = 192) -> pd.DataFrame:
    """
    若原始数据为96点，则复制成48小时（192点）；
    若已是192点，则直接返回；
    其他长度则插值到192点。
    """
    df = df.copy()

    if len(df) == total_steps:
        return df.reset_index(drop=True)

    if len(df) == 96 and total_steps == 192:
        df_48h = pd.concat([df, df], ignore_index=True)
        return df_48h

    # 其他长度：逐列插值到 total_steps
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("forecast_df 中未找到数值列，无法插值。")

    x_old = np.linspace(0, 1, len(df))
    x_new = np.linspace(0, 1, total_steps)

    out = pd.DataFrame()
    for col in df.columns:
        if col in numeric_cols:
            out[col] = np.interp(x_new, x_old, df[col].values)
        else:
            # 非数值列直接丢弃，或者保留空值都可以；这里不保留
            pass

    return out.reset_index(drop=True)


# ==========================================
# 3. 取滚动窗口：末端不足HORIZON_N时补齐
# ==========================================
def get_horizon_slice(df: pd.DataFrame, t_now: int, horizon_n: int) -> pd.DataFrame:
    """
    在末端如果剩余数据不足 horizon_n，则用最后一行重复补齐。
    """
    slice_df = df.iloc[t_now: t_now + horizon_n].copy()

    if len(slice_df) < horizon_n:
        last_row = slice_df.iloc[[-1]].copy()
        pad_count = horizon_n - len(slice_df)
        pad_df = pd.concat([last_row] * pad_count, ignore_index=True)
        slice_df = pd.concat([slice_df.reset_index(drop=True), pad_df], ignore_index=True)

    return slice_df


# ==========================================
# 4. Optimierungsmodell
# ==========================================
def solve_rhg_optimization(current_soc, forecast_slice, num_agents, N):
    model = ConcreteModel()
    model.I = RangeSet(0, num_agents - 1)
    model.T = RangeSet(0, N - 1)

    # 决策变量：s = 功率, q = 电池电量
    model.s = Var(model.I, model.T, bounds=(-S_MAX, S_MAX))
    model.q = Var(model.I, model.T, bounds=(0, BATTERY_CAP))

    # Batterie-Dynamik
    def soc_dynamics(m, i, t):
        if t == 0:
            return m.q[i, t] == current_soc[i]
        return m.q[i, t] == ALPHA * m.q[i, t - 1] + 0.25 * BETA * m.s[i, t - 1]

    model.soc_con = Constraint(model.I, model.T, rule=soc_dynamics)

    # Zielfunktion：整体负荷平滑 + 电池动作平滑
    def objective_rule(m):
        total_obj = 0.0
        forecast_array = forecast_slice.values
        avg_load_ref = np.sum(forecast_array) / N

        for t in m.T:
            base_load_t = np.sum(forecast_array[t, :])
            net_load = base_load_t + sum(m.s[i, t] for i in m.I)

            # 主目标：削峰填谷 / 向参考均值靠拢
            total_obj += (net_load - avg_load_ref) ** 2 * 1000

            # 辅助目标：避免过激充放电
            total_obj += 0.1 * sum(m.s[i, t] ** 2 for i in m.I)

        return total_obj

    model.obj = Objective(rule=objective_rule, sense=minimize)

    solver = SolverFactory(SOLVER_NAME)
    solver.solve(model, tee=False)

    # 只返回当前时刻的控制量（RHG / MPC思想）
    u_opt = []
    for i in range(num_agents):
        val = value(model.s[i, 0])
        if val is None:
            val = 0.0
        u_opt.append(float(val))

    return u_opt


# ==========================================
# 5. Hauptsimulationsschleife
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"Fehler: Datendatei {DATA_FILE} nicht gefunden. Bitte führen Sie zuerst prepare_agents.py aus.")
        sys.exit()

    # 初始化低压网络
    net = nw.create_cigre_network_lv()

    # 安装14个储能
    agent_bus_indices = []
    for i in range(1, NUM_AGENTS + 1):
        idx = net.bus[net.bus.name == f"Bus R{i}"].index[0]
        agent_bus_indices.append(idx)
        pp.create_storage(net, bus=idx, p_mw=0.0, max_e_mwh=BATTERY_CAP)

    # 读取预测数据
    forecast_df_raw = pd.read_csv(DATA_FILE)

    # 扩展为48小时 / 192点
    forecast_df = prepare_forecast_48h(forecast_df_raw, TOTAL_SIM_STEPS)

    # SoC 初值
    soc_tracker = [SOC_INIT] * NUM_AGENTS

    # 结果记录
    history = {
        "time_step": [],
        "base_load_mw": [],
        "opt_load_mw": [],
        "transformer_loading_percent": [],
        "battery_sum_mw": []
    }

    print(f">>> Führe RHG-Simulation für 48 Stunden aus ({TOTAL_SIM_STEPS} Zeitschritte)...")

    for t_now in range(TOTAL_SIM_STEPS):
        # 滚动窗口
        horizon_slice = get_horizon_slice(forecast_df, t_now, HORIZON_N)

        # 求当前时刻最优控制
        u_opt = solve_rhg_optimization(soc_tracker, horizon_slice, NUM_AGENTS, HORIZON_N)

        # 物理系统更新
        batt_sum = 0.0
        for i in range(NUM_AGENTS):
            net.storage.at[i, "p_mw"] = float(u_opt[i])

            # 更新SOC并限制在物理范围内
            soc_new = ALPHA * soc_tracker[i] + 0.25 * BETA * u_opt[i]
            soc_tracker[i] = float(np.clip(soc_new, 0.0, BATTERY_CAP))

            batt_sum += float(u_opt[i])

        # 潮流计算
        pp.runpp(net)

        # 基础总负荷
        actual_base = float(forecast_df.iloc[t_now].sum())
        optimized_load = actual_base + batt_sum

        # 记录结果
        history["time_step"].append(t_now)
        history["base_load_mw"].append(actual_base)
        history["opt_load_mw"].append(optimized_load)
        history["transformer_loading_percent"].append(float(net.res_trafo.loading_percent.values[0]))
        history["battery_sum_mw"].append(batt_sum)

        if t_now % 12 == 0:
            print(
                f"t={t_now:03d} | "
                f"Basislast: {actual_base:.4f} MW | "
                f"Optimiert: {optimized_load:.4f} MW | "
                f"BattSum: {batt_sum:.4f} MW | "
                f"Trafo: {history['transformer_loading_percent'][-1]:.2f}%"
            )

    # 保存结果
    res_df = pd.DataFrame(history)
    output_path = os.path.join(current_dir, "data", "simulation_results_rhg_18bus.csv")
    res_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSimulation erfolgreich abgeschlossen! Ergebnisse gespeichert unter: {output_path}")