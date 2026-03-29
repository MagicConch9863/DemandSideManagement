import simbench as sb
import pandas as pd
import numpy as np
import os

def extract_simbench_data_by_force():
    # Verbindung zur SimBench-Datenbank aufbauen
    print("Verbinde mit der SimBench-Datenbank...")
    #net = sb.get_simbench_net("1-LV-rural1--0-sw")
    #net = sb.get_simbench_net("1-LV-urban6--2-sw")
    net = sb.get_simbench_net("1-LV-urban6--0-sw")
    
    # Extraktion der Lastprofile (Einheit: MW)
    profiles = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    
    load_p_raw = profiles[('load', 'p_mw')]
    
    load_p_active = load_p_raw.loc[:, (load_p_raw != 0).any(axis=0)]
    
    time_steps = 200
    # Erstellung eines leeren Arrays für 18 Knoten (R1-R18)
    output_data = np.zeros((time_steps, 18))
    
    for i in range(18):
        src_col = i % load_p_active.shape[1]
        output_data[:, i] = load_p_active.iloc[:time_steps, src_col].values
        
    # Erstellen eines strukturierten DataFrames mit Spaltennamen (Bus R1 bis R18)
    output_df = pd.DataFrame(output_data, columns=[f"Bus R{i+1}" for i in range(18)])
    
    # Speichern der generierten Zeitreihen im CSV-Format
    os.makedirs("data", exist_ok=True)
    save_path = "data/load_profile_forecast_18bus.csv"
    output_df.to_csv(save_path, index=False)

    print("\n" + "="*50)
    print("Extraktion erfolgreich! Daten wurden für Bus R1 bis R18 zugewiesen.")
    print(f"Maximaler Lastwert: {output_df.max().max():.6f} MW")
    print(f"Vorschau (erste 5 Zeilen):\n{output_df.head(5)}")
    print("="*50)

if __name__ == "__main__":
    extract_original_simbench_data = extract_simbench_data_by_force
    extract_original_simbench_data()