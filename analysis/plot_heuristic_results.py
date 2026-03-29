import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "axes.linewidth": 1.0
})

def plot_heuristic_results():
    # ----- Pfade festlegen -----
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../data/heuristic_results_18bus.csv")
    output_path = os.path.join(current_dir, "../outputs/heuristic_pricing_performance.png")

    if not os.path.exists(data_path):
        print(f"Fehler: Datei {data_path} nicht gefunden.")
        return

    df = pd.read_csv(data_path)

    time_hours = np.arange(len(df)) * 0.25
    

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})


    ax1.plot(time_hours, df['base_load_kw'], color='#B0B0B0', linestyle='--', 
             linewidth=1.5, label='Nominal Load (Baseline)')
    ax1.plot(time_hours, df['opt_load_kw'], color='#59A14F', linestyle='-', 
             linewidth=2.2, label='Heuristic Optimized Load')

    ax1.fill_between(time_hours, df['base_load_kw'], df['opt_load_kw'],
                     where=(df['base_load_kw'] > df['opt_load_kw']),
                     color='#59A14F', alpha=0.3, label='Peak Shaving')
    
    ax1.fill_between(time_hours, df['base_load_kw'], df['opt_load_kw'],
                     where=(df['base_load_kw'] < df['opt_load_kw']),
                     color='#EDD08A', alpha=0.5, label='Valley Filling')


    ax1.set_ylabel("Total System Power (kW)")
    ax1.set_title("Heuristic Pricing Strategy: Load Smoothing Performance", pad=15)
    ax1.legend(loc='upper right', ncol=2, framealpha=0.8)
    

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)


    ax2.step(time_hours, df['price_signal'], color='#59A14F', linewidth=2, 
             where='post', label='Heuristic Price Signal')
    

    ax2.fill_between(time_hours, df['price_signal'], 0.20, step='post', 
                     color='#4C78A8', alpha=0.1)

    ax2.set_xlabel("Time [Hours]")
    ax2.set_ylabel("Dynamic Price")
    ax2.set_ylim(0.20, 0.40) 
    ax2.set_xlim(0, 48)
    ax2.set_xticks(np.arange(0, 49, 12))
    
    ax2.legend(loc='upper right')
    
    # Optik verbessern
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ----- Layout optimieren und speichern -----
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Grafik erfolgreich unter {output_path} gespeichert.")
    plt.show()

if __name__ == "__main__":
    plot_heuristic_results()