import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_it():

    current_dir = os.path.dirname(os.path.abspath(__file__))

    csv_path = os.path.join(current_dir, "../data/simulation_results_rhg_18bus.csv")
    df_res = pd.read_csv(csv_path)

    base = df_res["base_load_mw"].values
    opt = df_res["opt_load_mw"].values
    t = np.arange(len(base))  # 192 Zeitpunkte (48h bei 15-Min-Intervallen)

    hours = t / 4  

    plt.figure(figsize=(9, 4.5)) 

    plt.plot(hours, base, label="Nominale Last (Basislast)",
             color="grey", linestyle="--", linewidth=1.5)

    plt.plot(hours, opt, label="Optimierte Last (RHG)",
             color="green", linewidth=2.2)


    # Spitzenlastkappung (Peak Shaving)
    plt.fill_between(hours, base, opt,
                     where=(base > opt),
                     color="green", alpha=0.25,
                     label="Spitzenlastkappung")

    plt.fill_between(hours, base, opt,
                     where=(base < opt),
                     color="orange", alpha=0.25,
                     label="Lastanhebung")


    plt.title("48-Stunden RHG Lastglättung (18-Knoten-System)", fontsize=13)

    plt.xlabel("Zeit [Stunden]", fontsize=11)
    plt.ylabel("Leistung (MW)", fontsize=11)


    plt.xlim(0, 48)
    plt.xticks(np.arange(0, 49, 6))  # Markierung alle 6 Stunden

    plt.grid(True, alpha=0.3)

    plt.legend(fontsize=9)

    output_path = os.path.join(current_dir, "../outputs/RHG_interaction_final.png")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    print(f"Grafik wurde gespeichert unter: {output_path}")

    plt.show()


if __name__ == "__main__":
    plot_it()