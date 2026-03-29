import pandas as pd
import matplotlib.pyplot as plt

# ===== 1. 读取数据 =====
file_path = r"c:\Users\Dong\Desktop\DemandSideManagement\data\stackelberg_results_18bus_fast.csv"
df = pd.read_csv(file_path)

# ===== 2. 提取数据 =====
time = df["time_step"].values
base = df["base_load_kw"].values
opt = df["opt_load_kw"].values
ref = df["ref_load_kw"].values

# ===== 3. 绘图 =====
plt.figure(figsize=(12, 5))

plt.plot(time, base, linestyle="--", label="Baseline Load", color="gray")
plt.plot(time, opt, linewidth=2, label="Stackelberg Optimized Load", color="blue")
plt.plot(time, ref, linewidth=2, label="Reference Load", color="green")

plt.xlabel("Time Step (15 min)")
plt.ylabel("Power (kW)")
plt.title("Stackelberg Load Smoothing (24h)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()