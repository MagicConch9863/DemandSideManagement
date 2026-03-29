import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 12})

def plot_duopoly_beautiful():
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../data/duopoly_results_18bus.csv")
    output_img = os.path.join(current_dir, "../outputs/stackelberg_interaction_final.png")

    if not os.path.exists(data_path):
        print("Fehler: Datendatei nicht gefunden. Bitte führen Sie zuerst 'stackelberg_controller.py' aus.")
        return


    df = pd.read_csv(data_path)
    
    time_hours = df['step'] * 0.25
    
    # Layout erstellen: Zwei Diagramme untereinander
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2.5, 1.2]})
    
    # ---- Oberes Diagramm: Lastvergleich ----
    ax1.plot(time_hours, df['base_load'], color='gray', linestyle='--', linewidth=2, alpha=0.8, label='Nominale Last (Echtzeitdaten)')
    ax1.plot(time_hours, df['opt_load'], color='black', linewidth=2.5, label='Stackelberg-optimierte Last')
    
    # Flächen für Spitzenlastkappung und Talfüllung einfärben
    ax1.fill_between(time_hours, df['base_load'], df['opt_load'], 
                    where=(df['base_load'] - df['opt_load'] > 0.1), 
                    color='forestgreen', alpha=0.35, label='Spitzenlastkappung (Peak Shaving)')
    ax1.fill_between(time_hours, df['base_load'], df['opt_load'], 
                    where=(df['opt_load'] - df['base_load'] > 0.1), 
                    color='goldenrod', alpha=0.4, label='Lastanhebung (Valley Filling)')
    
    # Beschriftungen für das obere Diagramm
    ax1.set_ylabel('Gesamtsystemlast (kW)', fontsize=13, fontweight='bold')
    ax1.set_title('Stackelberg-Spiel: Prosumer-Interaktion & Lastglättung', fontsize=16, pad=15, fontweight='bold')
    ax1.legend(loc='upper left', ncol=2, fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ---- Unteres Diagramm: Dynamische Preisstrategie ----
    ax2.step(time_hours, df['price_factor'], color='crimson', linewidth=2, where='post', label='Preisstrategie des Leaders (ρ)')
    ax2.fill_between(time_hours, df['price_factor'], step='post', color='crimson', alpha=0.15)
    
    # Beschriftungen für das untere Diagramm
    ax2.set_ylabel('Anreizpreis (ρ)', color='crimson', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Zeit [Stunden]', fontsize=14, fontweight='bold')
    
    # X-Achsen-Skalierung auf 48 Stunden einstellen
    ax2.set_xticks(range(0, 49, 6))
    ax2.set_xlim([0, 48])
    ax2.set_ylim(-0.1, df['price_factor'].max() * 1.2)
    
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Layout optimieren
    plt.tight_layout()
    
    # Ordner erstellen, falls nicht vorhanden, und Bild speichern
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=300)
    print(f"✅ Stackelberg-Grafik wurde erfolgreich gespeichert unter: {output_img}")
    
    # Grafik anzeigen
    plt.show()

if __name__ == "__main__":
    plot_duopoly_beautiful()