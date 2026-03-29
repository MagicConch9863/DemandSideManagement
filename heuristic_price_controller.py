import os
import numpy as np
import pandas as pd

# ============================================================
# 0. 路径
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data", "load_profile_forecast_18bus.csv")
out_csv_path = os.path.join(current_dir, "data", "heuristic_results_18bus.csv")

# ============================================================
# 1. 基本参数
# ============================================================
DT_HOURS = 0.25
TOTAL_HOURS = 48
TOTAL_STEPS = int(TOTAL_HOURS / DT_HOURS)   # 192

# ============================================================
# 2. 储能参数
# ============================================================
NUM_AGENTS = 14
BATTERY_CAP_KWH = 4.0
SOC_INIT_RATIO = 0.60
SOC_MIN = 0.20
SOC_MAX = 0.90

P_CH_MAX_KW = 0.90
P_DIS_MAX_KW = 0.90
ETA_CH = 0.95
ETA_DIS = 0.95

# ============================================================
# 3. 在线 RHG 参数
# ============================================================
# 预测窗口：12个点 = 3小时
HORIZON_STEPS = 12

# 在线目标轨迹平滑
TARGET_SMOOTH_ALPHA = 0.82
TARGET_RAMP_LIMIT_KW = 0.35
MAX_TARGET_SHIFT_KW = 4.5

# RHG 输出平滑
RHG_RAMP_LIMIT_KW = 0.18
RHG_INERTIA = 0.98
TRACKING_GAIN = 0.22
DEADBAND_KW = 1.20

# ============================================================
# 4. 价格参数（非负、可升可降、均值回归）
# ============================================================
BASE_PRICE = 0.22
PRICE_DISPLAY_SCALE = 100.0

# 价格反馈长期回归中心
BASE_ADDER = 0.03

# 比例-微分-回归项
PRICE_KP = 0.08
PRICE_KD = 0.03
PRICE_KR = 0.15
PRICE_SMOOTH = 0.8

# ============================================================
# 5. 利润参数
# ============================================================
WHOLESALE_PRICE = 0.11
DEVIATION_PENALTY = 0.010
RAMP_PENALTY = 0.004

# ============================================================
# 6. 读入负荷：MW -> kW
# ============================================================
def load_base_profile(csv_file: str) -> np.ndarray:
    df = pd.read_csv(csv_file)

    bus_cols = [c for c in df.columns if str(c).strip().lower().startswith("bus")]
    if not bus_cols:
        bus_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not bus_cols:
        raise ValueError("未找到负荷列，请检查 load_profile_forecast_18bus.csv")

    total_mw = df[bus_cols].sum(axis=1).astype(float).values

    if len(total_mw) == TOTAL_STEPS:
        total_mw_48h = total_mw.copy()
    elif len(total_mw) == 96:
        total_mw_48h = np.tile(total_mw, 2)
    else:
        x_old = np.linspace(0, 1, len(total_mw))
        x_new = np.linspace(0, 1, TOTAL_STEPS)
        total_mw_48h = np.interp(x_new, x_old, total_mw)

    return total_mw_48h * 1000.0


# ============================================================
# 7. 在线参考轨迹：只用当前到未来H个预测点
# ============================================================
def compute_online_target(
    base_kw: np.ndarray,
    t: int,
    prev_target: float | None
) -> float:
    end = min(len(base_kw), t + HORIZON_STEPS)
    horizon_slice = base_kw[t:end]

    raw_target = float(np.mean(horizon_slice))

    # 限制目标不要离当前 base 太远
    raw_target = np.clip(
        raw_target,
        base_kw[t] - MAX_TARGET_SHIFT_KW,
        base_kw[t] + MAX_TARGET_SHIFT_KW
    )

    if prev_target is None:
        target = raw_target
    else:
        # 轻量 EMA
        target = TARGET_SMOOTH_ALPHA * prev_target + (1 - TARGET_SMOOTH_ALPHA) * raw_target

        # 坡度限制
        delta = target - prev_target
        delta = np.clip(delta, -TARGET_RAMP_LIMIT_KW, TARGET_RAMP_LIMIT_KW)
        target = prev_target + delta

    return float(target)


# ============================================================
# 8. 非负价格反馈：比例 + 微分 + 均值回归
#    目标：价格可升可降，不再长期单调漂移
# ============================================================
def compute_price_adder(
    base_adder: float,
    opt_load: float,
    target_load: float,
    prev_dev: float,
    prev_adder: float
) -> tuple[float, float]:
    # 当前偏差（归一化）
    dev = np.tanh((opt_load - target_load) / max(target_load, 1e-6))

    # 偏差变化率
    ddev = dev - prev_dev

    # 原始反馈
    raw_adder = (
        prev_adder
        + PRICE_KP * dev
        + PRICE_KD * ddev
        - PRICE_KR * (prev_adder - base_adder)
    )

    # 平滑
    adder = PRICE_SMOOTH * prev_adder + (1 - PRICE_SMOOTH) * raw_adder

    # 仅保证不为负，不做硬上限截断
    adder = max(0.0, adder)

    return float(adder), float(dev)


# ============================================================
# 9. 储能可用充放电能力
# ============================================================
def get_available_charge_discharge(soc_agents: np.ndarray) -> tuple[float, float]:
    total_charge = 0.0
    total_discharge = 0.0

    for soc in soc_agents:
        max_energy = SOC_MAX * BATTERY_CAP_KWH
        min_energy = SOC_MIN * BATTERY_CAP_KWH

        room_kwh = max(0.0, max_energy - soc)
        avail_kwh = max(0.0, soc - min_energy)

        charge_kw = min(P_CH_MAX_KW, room_kwh / (ETA_CH * DT_HOURS))
        discharge_kw = min(P_DIS_MAX_KW, avail_kwh * ETA_DIS / DT_HOURS)

        total_charge += charge_kw
        total_discharge += discharge_kw

    return total_charge, total_discharge


# ============================================================
# 10. 分配总电池功率到各 agent
# ============================================================
def dispatch_battery(total_batt_kw: float, soc_agents: np.ndarray) -> tuple[float, np.ndarray]:
    new_soc = soc_agents.copy()

    if total_batt_kw > 0:
        # 充电
        remain = total_batt_kw
        actual = 0.0

        for i in range(len(new_soc)):
            max_energy = SOC_MAX * BATTERY_CAP_KWH
            room_kwh = max(0.0, max_energy - new_soc[i])
            room_kw = min(P_CH_MAX_KW, room_kwh / (ETA_CH * DT_HOURS))

            use_kw = min(room_kw, remain)
            if use_kw > 0:
                new_soc[i] += use_kw * ETA_CH * DT_HOURS
                actual += use_kw
                remain -= use_kw

            if remain <= 1e-9:
                break

        return float(actual), new_soc

    if total_batt_kw < 0:
        # 放电
        remain = -total_batt_kw
        actual = 0.0

        for i in range(len(new_soc)):
            min_energy = SOC_MIN * BATTERY_CAP_KWH
            avail_kwh = max(0.0, new_soc[i] - min_energy)
            avail_kw = min(P_DIS_MAX_KW, avail_kwh * ETA_DIS / DT_HOURS)

            use_kw = min(avail_kw, remain)
            if use_kw > 0:
                new_soc[i] -= (use_kw * DT_HOURS) / ETA_DIS
                actual += use_kw
                remain -= use_kw

            if remain <= 1e-9:
                break

        return float(-actual), new_soc

    return 0.0, new_soc


# ============================================================
# 11. Follower：跟踪在线参考轨迹
# ============================================================
def follower_dispatch_step(
    base_load_kw: float,
    target_load_kw: float,
    prev_opt_load_kw: float,
    price_adder: float,
    soc_agents: np.ndarray
) -> tuple[float, float, np.ndarray, float]:
    # 慢变化理想下一步
    desired_next = RHG_INERTIA * prev_opt_load_kw + (1 - RHG_INERTIA) * target_load_kw

    delta_to_target = desired_next - prev_opt_load_kw
    delta_to_target = np.clip(delta_to_target, -RHG_RAMP_LIMIT_KW, RHG_RAMP_LIMIT_KW)
    smooth_ref = prev_opt_load_kw + delta_to_target

    # 当前 base 相对平滑轨迹的偏差
    dev = base_load_kw - smooth_ref

    if abs(dev) < DEADBAND_KW:
        desired_batt_kw = 0.0
    else:
        if dev > 0:
            # 削峰更积极
            desired_batt_kw = -TRACKING_GAIN * dev * (1.0 + 2.0 * price_adder / max(BASE_ADDER, 1e-9))
        else:
            # 填谷更温和
            desired_batt_kw = -0.65 * TRACKING_GAIN * dev * (1.0 + 0.8 * price_adder / max(BASE_ADDER, 1e-9))

    total_charge_max, total_discharge_max = get_available_charge_discharge(soc_agents)
    desired_batt_kw = np.clip(desired_batt_kw, -total_discharge_max, total_charge_max)

    batt_kw, new_soc = dispatch_battery(desired_batt_kw, soc_agents)

    raw_opt = base_load_kw + batt_kw

    # 输出再做一次惯性平滑
    blended_opt = RHG_INERTIA * prev_opt_load_kw + (1 - RHG_INERTIA) * raw_opt
    delta = blended_opt - prev_opt_load_kw
    delta = np.clip(delta, -RHG_RAMP_LIMIT_KW, RHG_RAMP_LIMIT_KW)
    opt_load_kw = prev_opt_load_kw + delta
    opt_load_kw = max(0.0, opt_load_kw)

    batt_kw_effective = opt_load_kw - base_load_kw

    return float(opt_load_kw), float(batt_kw_effective), new_soc, float(smooth_ref)


# ============================================================
# 12. 利润
# ============================================================
def utility_profit(price: float, actual_load_kw: float, target_load_kw: float, prev_load_kw: float) -> float:
    revenue = price * actual_load_kw * DT_HOURS
    purchase = WHOLESALE_PRICE * actual_load_kw * DT_HOURS
    deviation_cost = DEVIATION_PENALTY * ((actual_load_kw - target_load_kw) ** 2)
    ramp_cost = RAMP_PENALTY * ((actual_load_kw - prev_load_kw) ** 2)
    return float(revenue - purchase - deviation_cost - ramp_cost)


# ============================================================
# 13. 评价指标
# ============================================================
def print_metrics(res_df: pd.DataFrame) -> None:
    base = res_df["base_load_kw"].values
    opt = res_df["opt_load_kw"].values

    peak_base = float(np.max(base))
    peak_opt = float(np.max(opt))
    peak_reduction_pct = (peak_base - peak_opt) / peak_base * 100 if peak_base > 1e-9 else 0.0

    valley_base = float(np.min(base))
    valley_opt = float(np.min(opt))
    valley_fill_pct = (valley_opt - valley_base) / valley_base * 100 if valley_base > 1e-9 else 0.0

    ramp_base = float(np.mean(np.abs(np.diff(base))))
    ramp_opt = float(np.mean(np.abs(np.diff(opt))))
    ramp_reduction_pct = (ramp_base - ramp_opt) / ramp_base * 100 if ramp_base > 1e-9 else 0.0

    energy_base = float(np.sum(base) * DT_HOURS)
    energy_opt = float(np.sum(opt) * DT_HOURS)
    energy_gap_pct = (energy_opt - energy_base) / energy_base * 100 if energy_base > 1e-9 else 0.0

    print("\n==================== Evaluation Metrics ====================")
    print(f"Peak base load       : {peak_base:.2f} kW")
    print(f"Peak optimized load  : {peak_opt:.2f} kW")
    print(f"Peak reduction       : {peak_reduction_pct:.2f}%")
    print("------------------------------------------------------------")
    print(f"Valley base load     : {valley_base:.2f} kW")
    print(f"Valley optimized load: {valley_opt:.2f} kW")
    print(f"Valley filling       : {valley_fill_pct:.2f}%")
    print("------------------------------------------------------------")
    print(f"Avg ramp base        : {ramp_base:.3f} kW/step")
    print(f"Avg ramp optimized   : {ramp_opt:.3f} kW/step")
    print(f"Ramp reduction       : {ramp_reduction_pct:.2f}%")
    print("------------------------------------------------------------")
    print(f"Energy base          : {energy_base:.2f} kWh")
    print(f"Energy optimized     : {energy_opt:.2f} kWh")
    print(f"Energy gap           : {energy_gap_pct:.2f}%")
    print("============================================================\n")


# ============================================================
# 14. 主程序
# ============================================================
def run_duopoly_controller() -> pd.DataFrame:
    base_load_kw = load_base_profile(data_path)

    soc_agents = np.ones(NUM_AGENTS) * (SOC_INIT_RATIO * BATTERY_CAP_KWH)

    history = {
        "base_load_kw": [],
        "target_load_kw": [],
        "opt_load_kw": [],
        "smooth_ref_kw": [],
        "price_signal": [],
        "price_adder": [],
        "price_feedback_display": [],
        "battery_dispatch_kw": [],
        "utility_profit": []
    }

    prev_opt_load = float(base_load_kw[0])
    prev_target = None
    prev_price_adder = BASE_ADDER
    prev_dev = 0.0

    for t in range(TOTAL_STEPS):
        base_t = float(base_load_kw[t])

        # 在线目标轨迹：当前到未来 horizon 的均值 + 轻量平滑
        target_t = compute_online_target(base_load_kw, t, prev_target)

        # 第一次价格更新：用上一时刻输出负荷估计当前价格
        current_adder, _ = compute_price_adder(
            base_adder=BASE_ADDER,
            opt_load=prev_opt_load,
            target_load=target_t,
            prev_dev=prev_dev,
            prev_adder=prev_price_adder
        )
        current_price = BASE_PRICE + current_adder

        # Follower 响应
        opt_t, batt_t, soc_agents, smooth_ref_t = follower_dispatch_step(
            base_load_kw=base_t,
            target_load_kw=target_t,
            prev_opt_load_kw=prev_opt_load,
            price_adder=current_adder,
            soc_agents=soc_agents
        )

        # 第二次价格更新：用优化后的真实结果修正价格
        current_adder, current_dev = compute_price_adder(
            base_adder=BASE_ADDER,
            opt_load=opt_t,
            target_load=target_t,
            prev_dev=prev_dev,
            prev_adder=prev_price_adder
        )
        current_price = BASE_PRICE + current_adder

        profit_t = utility_profit(
            price=current_price,
            actual_load_kw=opt_t,
            target_load_kw=target_t,
            prev_load_kw=prev_opt_load
        )

        history["base_load_kw"].append(base_t)
        history["target_load_kw"].append(target_t)
        history["opt_load_kw"].append(opt_t)
        history["smooth_ref_kw"].append(smooth_ref_t)
        history["price_signal"].append(current_price)
        history["price_adder"].append(current_adder)
        history["price_feedback_display"].append(current_adder * PRICE_DISPLAY_SCALE)
        history["battery_dispatch_kw"].append(batt_t)
        history["utility_profit"].append(profit_t)

        prev_opt_load = opt_t
        prev_target = target_t
        prev_price_adder = current_adder
        prev_dev = current_dev

        if t % 12 == 0:
            print(
                f"t={t:03d} | base={base_t:6.2f} kW | "
                f"target={target_t:6.2f} kW | "
                f"opt={opt_t:6.2f} kW | "
                f"price={current_price:6.3f} | "
                f"feedbackx100={current_adder * 100:5.2f}"
            )

    res_df = pd.DataFrame(history)
    res_df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n>>> 结果已保存到: {out_csv_path}")

    print_metrics(res_df)
    return res_df


if __name__ == "__main__":
    run_duopoly_controller()