import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_triple_comparison():
    # ===== 1. 智能路径解析 (彻底解决路径报错) =====
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()

    if os.path.basename(current_dir) == 'analysis':
        project_root = os.path.dirname(current_dir)
    else:
        project_root = current_dir

    # 精准拼装三个算法的 CSV 路径
    rhg_path = os.path.join(project_root, "data", "simulation_results_rhg_18bus.csv")
    heu_path = os.path.join(project_root, "data", "heuristic_results_18bus.csv")
    duo_path = os.path.join(project_root, "data", "duopoly_results_18bus.csv")
    
    output_fig_path = os.path.join(project_root, "outputs", "rhg_vs_heu_vs_stackelberg_48h.png")
    
    # 检查文件是否存在
    files = [rhg_path, heu_path, duo_path]
    if not all(os.path.exists(f) for f in files):
        print("\n❌ 错误: 缺少 CSV 数据文件。请确认 RHG、Heuristic 和 Duopoly 是否都已经运行完毕。")
        for f in files: print(f"{'✅ 已找到' if os.path.exists(f) else '❌ 未找到'}: {f}")
        return

    # ===== 2. 读取数据 =====
    df_rhg = pd.read_csv(rhg_path)
    df_heu = pd.read_csv(heu_path)
    df_duo = pd.read_csv(duo_path)
    
    # ===== 3. 对齐数据长度 (防止维度报错) =====
    min_len = min(len(df_rhg), len(df_heu), len(df_duo))
    df_rhg = df_rhg.iloc[:min_len]
    df_heu = df_heu.iloc[:min_len]
    df_duo = df_duo.iloc[:min_len]
    
    # 生成时间轴：每步 15分钟 = 0.25小时
    time_hours = np.arange(min_len) * 0.25 
    
    # ===== 4. 提取并转换数据 (注意单位换算) =====
    # 基准负荷 (取自 Heuristic，单位 kW)
    base_load = df_heu['base_load_kw'] 
    
    # (1) RHG 优化负荷: 注意这里从 MW 转换为 kW
    rhg_opt_load = df_rhg['opt_load_mw'] * 1000.0
    
    # (2) Heuristic 优化负荷 (单位 kW)
    heu_opt_load = df_heu['opt_load_kw']
    
    # (3) Stackelberg/Duopoly 优化负荷 (单位 kW)
    duo_opt_load = df_duo['opt_load']

    # ===== 5. 开始美观绘图 =====
    plt.figure(figsize=(15, 8))
    
    # (0) 原始基准负荷 (灰色虚线)
    plt.plot(time_hours, base_load, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Nominal Load (Baseline)')
    
    # (1) RHG 集中式控制 (经典蓝色)
    plt.plot(time_hours, rhg_opt_load, color='#1f77b4', linewidth=2, alpha=0.9, label='Centralized RHG Control')
    
    # (2) 启发式定价 Heuristic (草绿色)
    plt.plot(time_hours, heu_opt_load, color='#2ca02c', linewidth=2, alpha=0.9, label='Heuristic Price Control')
    
    # (3) 终极博弈算法 Stackelberg (鲜艳红色，加粗强调)
    plt.plot(time_hours, duo_opt_load, color='#d62728', linewidth=3, alpha=1.0, label='Stackelberg Game (Duopoly)')

    # ===== 6. 图表装饰与格式化 =====
    plt.title('Comprehensive Load Smoothing Comparison: RHG vs. Heuristic vs. Stackelberg', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time [Hours]', fontsize=14, fontweight='bold')
    plt.ylabel('Total System Power (kW)', fontsize=14, fontweight='bold')
    
    # X 轴刻度：每 6 小时显示一个
    plt.xticks(np.arange(0, int(time_hours[-1]) + 2, 6))
    plt.xlim(0, time_hours[-1]) 
    
    # 启用网格、设置图例
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9, ncol=2) # 分两列显示图例，更美观
    
    plt.tight_layout()

    # 保存并显示
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 三重对比图已绘制成功，保存至:\n {output_fig_path}")
    plt.show()

if __name__ == "__main__":
    plot_triple_comparison()