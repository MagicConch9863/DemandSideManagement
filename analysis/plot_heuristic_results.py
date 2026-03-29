import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 学术级样式配置
# ==========================================
plt.rcParams.update({
    "font.family": "Times New Roman", 
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 120,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "axes.linewidth": 1.0,
    "legend.frameon": True
})

COLORS = {
    "Baseline": "#B0B0B0",
    "Heuristic": "#59A14F"
}

def plot_heuristic_analysis():
    # ----- 1. 智能路径定位 (修复递归寻找 data 失败的问题) -----
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(current_file_path)
    
    # 向上寻找，直到找到包含 'data' 文件夹的目录
    while not os.path.exists(os.path.join(project_root, "data")):
        parent = os.path.dirname(project_root)
        if parent == project_root: # 已经到达磁盘根目录
            break
        project_root = parent

    data_path = os.path.join(project_root, "data", "heuristic_results_18bus.csv")
    output_path = os.path.join(project_root, "outputs", "heuristic_interaction_final.png")

    print(f"🔍 正在尝试读取数据: {data_path}")

    if not os.path.exists(data_path):
        print(f"❌ 错误: 找不到数据文件。请确保 {data_path} 存在。")
        return

    # ----- 2. 读取数据 -----
    df = pd.read_csv(data_path)
    # 取数据长度，每步 15min = 0.25h
    time_hours = np.arange(len(df)) * 0.25 
    
    # 匹配你 CSV 里的真实列名
    base_l = df['base_load_kw']
    opt_l = df['opt_load_kw']
    price_s = df['price_signal']

    # ----- 3. 绘图开始 -----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True, gridspec_kw={'height_ratios': [2.5, 1.2]})

    # --- TOP: Load Profile ---
    ax1.plot(time_hours, base_l, color=COLORS["Baseline"], linestyle='--', linewidth=1.2, alpha=0.9, label='Nominal Load (Baseline)')
    ax1.plot(time_hours, opt_l, color=COLORS["Heuristic"], linewidth=1.8, label='Heuristic Optimized Load')
    
    ax1.fill_between(time_hours, base_l, opt_l, 
                    where=(base_l > opt_l), color='forestgreen', alpha=0.3, label='Peak Shaving')
    ax1.fill_between(time_hours, base_l, opt_l, 
                    where=(opt_l > base_l), color='goldenrod', alpha=0.3, label='Valley Filling')
    
    ax1.set_ylabel('Total System Power (kW)')
    ax1.set_title("Heuristic Pricing Strategy: Load Smoothing Performance", pad=12)
    ax1.legend(loc='upper right', ncol=2)
    ax1.set_xlim(0, 48)
    ax1.set_xticks(range(0, 49, 12)) 
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- BOTTOM: Price Signal ---
    ax2.step(time_hours, price_s, color=COLORS["Heuristic"], linewidth=2, where='post', label='Heuristic Price Signal ')
    ax2.fill_between(time_hours, price_s, step='post', alpha=0.1)
    
    ax2.set_ylabel('Dynamic Price ')
    ax2.set_xlabel('Time [Hours]')
    ax2.legend(loc='upper right')
    
    # 留出上下边距
    ax2.set_ylim(price_s.min() * 0.9, price_s.max() * 1.1)
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图片已成功保存至: {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_heuristic_analysis()