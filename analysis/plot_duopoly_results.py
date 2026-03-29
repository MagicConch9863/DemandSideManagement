import pandas as pd
import matplotlib.pyplot as plt
import os

# 配置全局绘图字体大小
plt.rcParams.update({'font.size': 12})

def plot_duopoly_beautiful():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../data/duopoly_results_18bus.csv")
    output_img = os.path.join(current_dir, "../outputs/stackelberg_interaction_final.png")
    
    if not os.path.exists(data_path):
        print("未找到数据文件，请先运行 duopoly_controller.py")
        return

    df = pd.read_csv(data_path)
    
    # 将步数转换为小时 (每步 15分钟 = 0.25小时)
    time_hours = df['step'] * 0.25
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2.5, 1.2]})
    
    # ---- 上图：负荷平滑对比 ----
    ax1.plot(time_hours, df['base_load'], color='gray', linestyle='--', linewidth=2, alpha=0.8, label='Nominal Load (Real Data)')
    ax1.plot(time_hours, df['opt_load'], color='black', linewidth=2.5, label='Stackelberg Optimized Load')
    
    ax1.fill_between(time_hours, df['base_load'], df['opt_load'], 
                    where=(df['base_load'] - df['opt_load'] > 0.1), 
                    color='forestgreen', alpha=0.35, label='Peak Shaving')
    ax1.fill_between(time_hours, df['base_load'], df['opt_load'], 
                    where=(df['opt_load'] - df['base_load'] > 0.1), 
                    color='goldenrod', alpha=0.4, label='Valley Filling')
    
    ax1.set_ylabel('Total System Load (kW)', fontsize=13, fontweight='bold')
    # 居中、大气的美观标题
    ax1.set_title('Stackelberg Game: Prosumer Interaction & Load Smoothing', fontsize=16, pad=15, fontweight='bold')
    ax1.legend(loc='upper left', ncol=2, fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ---- 下图：动态价格策略 ----
    ax2.step(time_hours, df['price_factor'], color='crimson', linewidth=2, where='post', label='Leader Pricing Strategy (\u03C1)')
    ax2.fill_between(time_hours, df['price_factor'], step='post', color='crimson', alpha=0.15)
    
    ax2.set_ylabel('Incentive Price (\u03C1)', color='crimson', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Time [Hours]', fontsize=14, fontweight='bold')
    
    # 设置 X 轴刻度为 0, 6, 12 ... 48
    ax2.set_xticks(range(0, 49, 6))
    ax2.set_xlim([0, 48])
    ax2.set_ylim(-0.1, df['price_factor'].max() * 1.2)
    
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=300)
    print(f"✅ Stackelberg 图片已保存至: {output_img}")
    plt.show()

if __name__ == "__main__":
    plot_duopoly_beautiful()