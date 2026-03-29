import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 学术样式配置 (字体减负)
# ==========================================
plt.rcParams.update({
    "font.family": "Times New Roman", 
    "font.size": 10,
    "axes.titlesize": 12, # 统一标题大小
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "axes.linewidth": 1.0,
    "legend.frameon": True
})

COLORS = {
    "Baseline": "#B0B0B0", "RHG": "#4C78A8", 
    "Heuristic": "#59A14F", "Stackelberg": "#E15759"
}

def plot_refined_academic_analysis():
    # ----- 路径定位 -----
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()
    project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == 'analysis' else current_dir

    paths = {
        "RHG": os.path.join(project_root, "data", "simulation_results_rhg_18bus.csv"),
        "Heuristic": os.path.join(project_root, "data", "heuristic_results_18bus.csv"),
        "Stackelberg": os.path.join(project_root, "data", "duopoly_results_18bus.csv")
    }
    
    # ----- 数据加载与计算 -----
    df_rhg, df_heu, df_duo = pd.read_csv(paths["RHG"]), pd.read_csv(paths["Heuristic"]), pd.read_csv(paths["Stackelberg"])
    min_len = min(len(df_rhg), len(df_heu), len(df_duo))
    
    base_l = df_heu['base_load_kw'].values[:min_len]
    rhg_l = df_rhg['opt_load_mw'].values[:min_len] * 1000.0
    heu_l = df_heu['opt_load_kw'].values[:min_len]
    duo_l = df_duo['opt_load'].values[:min_len]

    def get_metrics(v):
        p_red = (np.max(base_l) - np.max(v)) / np.max(base_l) * 100
        v_fill = (np.min(v) - np.min(base_l)) / np.min(base_l) * 100
        pv_red = ((np.max(base_l)-np.min(base_l)) - (np.max(v)-np.min(v))) / (np.max(base_l)-np.min(base_l)) * 100
        s_imp = (np.std(base_l) - np.std(v)) / np.std(base_l) * 100
        # 能量比例：(优化后总能量 / 基准总能量) * 100
        energy_ratio = (np.sum(v) / np.sum(base_l)) * 100
        return [p_red, v_fill, pv_red, s_imp, energy_ratio]

    metrics_names = ['Peak Shaving', 'Valley Filling', 'P-V Reduction', 'Smoothness Imp.', 'Total Energy (%)']
    m_rhg = get_metrics(rhg_l)
    m_heu = get_metrics(heu_l)
    m_duo = get_metrics(duo_l)

    # ----- 绘图开始 -----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.5), gridspec_kw={'height_ratios': [1.2, 1]})
    time_hours = np.arange(min_len) * 0.25

    # --- TOP: Load Profile ---
    ax1.plot(time_hours, base_l, color=COLORS["Baseline"], linestyle='--', linewidth=1.2, label='Nominal Load')
    ax1.plot(time_hours, rhg_l, color=COLORS["RHG"], linewidth=1.5, label='Centralized RHG')
    ax1.plot(time_hours, heu_l, color=COLORS["Heuristic"], linewidth=1.5, label='Heuristic Pricing')
    ax1.plot(time_hours, duo_l, color=COLORS["Stackelberg"], linewidth=2.5, label='Stackelberg (Proposed)')
    
    ax1.set_title("Load Smoothing Performance over 48 Hours", pad=12) # 去除加粗
    ax1.set_ylabel('Total Power (kW)')
    ax1.set_xticks(np.arange(0, 49, 12))
    ax1.set_xlim(0, 48)
    ax1.legend(loc='upper right', ncol=2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- BOTTOM: Metrics ---
    x = np.arange(len(metrics_names))
    width = 0.22
    
    # 绘制前四个指标的柱子
    rects1 = ax2.bar(x[:4] - width, m_rhg[:4], width, label='RHG', color=COLORS["RHG"], alpha=0.8)
    rects2 = ax2.bar(x[:4], m_heu[:4], width, label='Heuristic', color=COLORS["Heuristic"], alpha=0.8)
    rects3 = ax2.bar(x[:4] + width, m_duo[:4], width, label='Stackelberg', color=COLORS["Stackelberg"], alpha=0.8)
    
    # 绘制最后一个指标（Total Energy）的柱子，包括 Baseline (100%)
    ax2.bar(x[4] - 1.5*width, 100, width*0.7, color=COLORS["Baseline"], alpha=0.5) # Baseline 柱子
    ax2.bar(x[4] - 0.5*width, m_rhg[4], width*0.7, color=COLORS["RHG"], alpha=0.8)
    ax2.bar(x[4] + 0.5*width, m_heu[4], width*0.7, color=COLORS["Heuristic"], alpha=0.8)
    ax2.bar(x[4] + 1.5*width, m_duo[4], width*0.7, color=COLORS["Stackelberg"], alpha=0.8)

    ax2.set_ylabel('Improvement / Energy Ratio (%)')
    ax2.set_title("Performance Metrics & Energy Consumption", pad=12) # 去除加粗
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names)
    ax2.set_ylim(0, 115) # 给顶部留出空间

    # 为前四个指标添加标注 (标注改为灰色，不那么突兀)
    def label_metrics(rects):
        for rect in rects:
            h = rect.get_height()
            ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, color='#555555')

    label_metrics(rects1); label_metrics(rects2); label_metrics(rects3)

    # 图例放在右上角，设置背景透明度，不遮挡数据
    ax2.legend(loc='upper right', framealpha=0.8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_refined_academic_analysis()