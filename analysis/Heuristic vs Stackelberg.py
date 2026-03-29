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
    "axes.linewidth": 1.0,
    "legend.frameon": True
})

FARBEN = {
    "Baseline": "#B0B0B0", 
    "RHG": "#4C78A8", 
    "Heuristik": "#59A14F", 
    "Stackelberg": "#E15759"
}

def plot_refined_academic_analysis():
    # ----- Pfadermittlung -----
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()
    
    project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) == 'analysis' else current_dir

    pfade = {
        "RHG": os.path.join(project_root, "data", "simulation_results_rhg_18bus.csv"),
        "Heuristik": os.path.join(project_root, "data", "heuristic_results_18bus.csv"),
        "Stackelberg": os.path.join(project_root, "data", "duopoly_results_18bus.csv")
    }
    
    # ----- Daten laden und berechnen -----
    df_rhg = pd.read_csv(pfade["RHG"])
    df_heu = pd.read_csv(pfade["Heuristik"])
    df_duo = pd.read_csv(pfade["Stackelberg"])
    
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
        energy_ratio = (np.sum(v) / np.sum(base_l)) * 100
        return [p_red, v_fill, pv_red, s_imp, energy_ratio]

    metriken_namen = ['Peak Shaving', 'Valley Filling', 'P-V Reduction', 'Smoothness Imp', 'Total Energy (%)']
    m_rhg = get_metrics(rhg_l)
    m_heu = get_metrics(heu_l)
    m_duo = get_metrics(duo_l)

    # ----- Plot-Erstellung -----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.5), gridspec_kw={'height_ratios': [1.2, 1]})
    zeit_stunden = np.arange(min_len) * 0.25

    # --- OBEN: Lastprofil ---
    ax1.plot(zeit_stunden, base_l, color=FARBEN["Baseline"], linestyle='--', linewidth=1.2, label='Nominale Last (Basis)')
    ax1.plot(zeit_stunden, rhg_l, color=FARBEN["RHG"], linewidth=1.5, label='Zentralisiertes RHG')
    ax1.plot(zeit_stunden, heu_l, color=FARBEN["Heuristik"], linewidth=1.5, label='Heuristische Bepreisung')
    ax1.plot(zeit_stunden, duo_l, color=FARBEN["Stackelberg"], linewidth=2.5, label='Stackelberg (Vorgeschlagen)')
    
    ax1.set_title("Load Smoothing Performance over 48 Hours", pad=12)
    ax1.set_ylabel('Gesamtleistung (kW)')
    ax1.set_xlabel('Zeit [Stunden]')
    ax1.set_xticks(np.arange(0, 49, 12))
    ax1.set_xlim(0, 48)
    ax1.legend(loc='upper right', ncol=2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- UNTEN: Metriken ---
    x = np.arange(len(metriken_namen))
    breite = 0.22
    
    # Balken für die ersten vier Metriken
    rects1 = ax2.bar(x[:4] - breite, m_rhg[:4], breite, label='RHG', color=FARBEN["RHG"], alpha=0.8)
    rects2 = ax2.bar(x[:4], m_heu[:4], breite, label='Heuristik', color=FARBEN["Heuristik"], alpha=0.8)
    rects3 = ax2.bar(x[:4] + breite, m_duo[:4], breite, label='Stackelberg', color=FARBEN["Stackelberg"], alpha=0.8)
    
    # Letzte Metrik (Gesamtenergie) mit Baseline (100%)
    ax2.bar(x[4] - 1.5*breite, 100, breite*0.7, color=FARBEN["Baseline"], alpha=0.5, label='Referenz (100%)')
    ax2.bar(x[4] - 0.5*breite, m_rhg[4], breite*0.7, color=FARBEN["RHG"], alpha=0.8)
    ax2.bar(x[4] + 0.5*breite, m_heu[4], breite*0.7, color=FARBEN["Heuristik"], alpha=0.8)
    ax2.bar(x[4] + 1.5*breite, m_duo[4], breite*0.7, color=FARBEN["Stackelberg"], alpha=0.8)

    ax2.set_ylabel('Verbesserung / Energieverhältnis (%)')
    ax2.set_title("Leistungskennzahlen & Energieverbrauch", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(metriken_namen)
    ax2.set_ylim(0, 115)

    # Wertebeschriftung für die ersten vier Metriken
    def label_metrics(rects):
        for rect in rects:
            h = rect.get_height()
            ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8, color='#555555')

    label_metrics(rects1)
    label_metrics(rects2)
    label_metrics(rects3)

    ax2.legend(loc='upper right', framealpha=0.8, ncol=2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()

    # ----- Bild speichern -----
    ausgabe_ordner = os.path.join(project_root, "outputs")
    if not os.path.exists(ausgabe_ordner):
        os.makedirs(ausgabe_ordner)
    
    speicherpfad = os.path.join(ausgabe_ordner, "vergleich_algorithmen_analyse.png")
    plt.savefig(speicherpfad, dpi=300, bbox_inches='tight')
    print(f"Grafik erfolgreich gespeichert unter: {speicherpfad}")
    
    plt.show()

if __name__ == "__main__":
    plot_refined_academic_analysis()