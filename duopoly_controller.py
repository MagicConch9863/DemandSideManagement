import pandas as pd
import numpy as np
from pyomo.environ import *
import os

# ==========================================
# 1. 核心参数配置
# ==========================================
NUM_AGENTS = 18       
S_MAX = 1.0           
BATTERY_CAP = 5.0     
SOC_INIT = 2.5        

ALPHA = 0.999         
BETA = 0.95           
HORIZON_N = 16        

SOLVER_NAME = 'gurobi'

# ==========================================
# 路径配置 (适配脚本在根目录的情况)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

# 脚本在根目录，data 文件夹也在根目录，所以直接拼接 "data" 即可，不需要 "../"
DATA_FILE = os.path.join(current_dir, "data", "load_profile_forecast_18bus.csv")
OUTPUT_CSV = os.path.join(current_dir, "data", "duopoly_results_18bus.csv")

def load_real_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"找不到数据文件: {DATA_FILE}\n当前查找路径为: {DATA_FILE}")
    df_raw = pd.read_csv(DATA_FILE)
    total_load_kw = df_raw.sum(axis=1) * 1000.0
    return pd.DataFrame({'total_base_load': total_load_kw})

def solve_prosumer_response(current_soc, forecast_slice, rho_incentive, target_load):
    model = ConcreteModel()
    model.I = RangeSet(0, NUM_AGENTS - 1)
    model.T = RangeSet(0, HORIZON_N - 1)
    
    model.s = Var(model.I, model.T, bounds=(-S_MAX, S_MAX)) 
    model.q = Var(model.I, model.T, bounds=(0, BATTERY_CAP)) 

    def soc_con(m, i, t):
        if t == 0: return m.q[i, t] == current_soc[i]
        return m.q[i, t] == ALPHA * m.q[i, t-1] - 0.25 * BETA * m.s[i, t-1]
    model.soc_limit = Constraint(model.I, model.T, rule=soc_con)

    def obj_rule(m):
        cost = 0
        for t in m.T:
            base_l = forecast_slice.iloc[t]
            net_l = base_l - sum(m.s[i, t] for i in m.I)
            cost += rho_incentive * (net_l - target_load)**2 + 0.5 * sum(m.s[i, t]**2 for i in m.I)
        terminal_penalty = 15.0 * sum((m.q[i, HORIZON_N-1] - SOC_INIT)**2 for i in m.I)
        return cost + terminal_penalty

    model.obj = Objective(rule=obj_rule, sense=minimize)
    
    solver = SolverFactory(SOLVER_NAME)
    solver.options['LogToConsole'] = 0
    solver.solve(model)
    return [value(model.s[i, 0]) for i in model.I]

def run_simulation():
    data = load_real_data()
    system_target_load = data['total_base_load'].mean()
    print(f"数据加载成功！系统平滑目标负荷为: {system_target_load:.2f} kW")
    
    total_steps = min(192, len(data) - HORIZON_N)
    soc_tracker = [SOC_INIT] * NUM_AGENTS
    results = []

    print(">>> [Gurobi] 正在推演 Stackelberg 博弈，导出数据...")
    for t in range(total_steps):
        horizon_slice = data['total_base_load'].iloc[t : t + HORIZON_N]
        base_load_t = horizon_slice.iloc[0]
        
        best_rho, min_grid_cost, best_s = 0, float('inf'), [0] * NUM_AGENTS
        
        for trial_rho in np.linspace(0.0, 5.0, 20):
            s_t = solve_prosumer_response(soc_tracker, horizon_slice, trial_rho, system_target_load)
            net_l = base_load_t - sum(s_t)
            
            total_cost = 2.0 * (net_l - system_target_load)**2 + 1.5 * trial_rho**2 
            if total_cost < min_grid_cost:
                min_grid_cost = total_cost
                best_rho = trial_rho
                best_s = s_t

        for i in range(NUM_AGENTS):
            soc_tracker[i] = np.clip(ALPHA * soc_tracker[i] - 0.25 * BETA * best_s[i], 0, BATTERY_CAP)

        optimized_l = base_load_t - sum(best_s)
        results.append({
            'step': t,
            'base_load': base_load_t,
            'opt_load': optimized_l,
            'price_factor': best_rho
        })
        if t % 20 == 0:
            print(f"Step {t:03d} | Base: {base_load_t:.1f} kW | Opt: {optimized_l:.1f} kW | Rho: {best_rho:.2f}")

    # ===== 保存为 CSV =====
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 计算完成！数据已成功保存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_simulation()