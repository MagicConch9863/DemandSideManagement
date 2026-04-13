import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import simbench

def get_simbench_profile():
    net = simbench.get_simbench_net("1-LV-urban6--0-sw")

    # 只选18个节点（和RHG一致）
    selected = net.load.index[:18]

    load = net.load.loc[selected, "p_mw"].values
    profile = net.load["profile"].values

    total = load * profile

    return total

def main():
    # 1. 读取数据 (合并了原 main 的逻辑)
    try:
        rhg_df = pd.read_csv("outputs/rhg_summary.csv")
        st_df = pd.read_csv("outputs/stackelberg_hourly_summary.csv")
    except FileNotFoundError:
        print("错误：找不到指定的 CSV 文件，请确保 outputs 目录下有 rhg_summary.csv 和 stackelberg_hourly_summary.csv")
        return

    # 提取所需列
    raw = rhg_df["raw_simbench_demand_kw"].values
    rhg = rhg_df["rhg_optimized_load_kw"].values
    stackelberg = st_df["optimized_pcc_import_kw"].values
    market_price = st_df["real_price_eur_per_kwh"].values
    leader_price = st_df["leader_price_eur_per_kwh"].values
    t = np.arange(len(raw))

    # 2. 绘图逻辑 (合并了原 plot 函数并简化)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.5]})
    
    # 上图：功率对比
    axes[0].plot(t, raw, color="gray", alpha=0.6, label="Raw SimBench demand", linestyle="--")
    axes[0].plot(t, rhg, color="green", linewidth=2, label="RHG (Trend-Following)")
    axes[0].plot(t, stackelberg, color="red", linewidth=2, label="Stackelberg Optimized")
    axes[0].set_ylabel("Power (kW)")
    axes[0].set_title("RHG vs Stackelberg Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 下图：价格对比
    axes[1].plot(t, market_price, color="tab:blue", label="Market Price")
    axes[1].plot(t, leader_price, color="tab:orange", label="Leader Price")
    axes[1].set_ylabel("Price (EUR/kWh)")
    axes[1].set_xlabel("15-min Step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    # 保存并显示
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/rhg_vs_stackelberg_comparison.png", dpi=200)
    print("对比图已保存至: outputs/rhg_vs_stackelberg_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()