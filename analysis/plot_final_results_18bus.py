import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_it():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # =========================
    # 读取RHG结果（核心）
    # =========================
    df_res = pd.read_csv(os.path.join(current_dir, "../data/simulation_results_rhg_18bus.csv"))

    base = df_res["base_load_mw"].values
    opt = df_res["opt_load_mw"].values
    t = np.arange(len(base))  # 192点

    # =========================
    # 时间轴（小时）
    # =========================
    hours = t / 4  # 15min -> hour

    # =========================
    # 创建图（适配2K屏）
    # =========================
    plt.figure(figsize=(9, 4.5))  # ✅ 不再占满屏

    # =========================
    # 曲线
    # =========================
    plt.plot(hours, base, label="Nominal Load (Baseline)",
             color="grey", linestyle="--", linewidth=1.5)

    plt.plot(hours, opt, label="RHG Optimized Load",
             color="green", linewidth=2.2)

    # =========================
    # 填充（削峰填谷）
    # =========================
    plt.fill_between(hours, base, opt,
                     where=(base > opt),
                     color="green", alpha=0.25,
                     label="Peak Shaving")

    plt.fill_between(hours, base, opt,
                     where=(base < opt),
                     color="orange", alpha=0.25,
                     label="Valley Filling")

    # =========================
    # 标题与坐标
    # =========================
    plt.title("48-Hour RHG Load Smoothing (18-Bus System)", fontsize=13)

    plt.xlabel("Time [Hours]", fontsize=11)
    plt.ylabel("Power (MW)", fontsize=11)

    # =========================
    # 坐标轴优化
    # =========================
    plt.xlim(0, 48)
    plt.xticks(np.arange(0, 49, 6))  # 每6小时一个刻度

    plt.grid(True, alpha=0.3)

    # =========================
    # 图例
    # =========================
    plt.legend(fontsize=9)

    # =========================
    # 保存
    # =========================
    output_path = os.path.join(current_dir, "../outputs/peak_shaving_18bus_48h.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    print(f"图已保存: {output_path}")

    plt.show()


if __name__ == "__main__":
    plot_it()