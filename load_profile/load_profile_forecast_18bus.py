import pandapower as pp
import pandapower.networks as nw
import pandas as pd
import numpy as np
import os

# 1. Modell laden
net = nw.create_cigre_network_lv()

# 2. Definition der Agenten (14 aktive, 4 passive)
active_agents = [f"Bus R{i}" for i in range(1, 15)]
passive_agents = [f"Bus R{i}" for i in range(15, 19)]
all_agents = active_agents + passive_agents

# 3. Datengenerierung simulieren (Wirkleistung in MW)
time_steps = 48  
load_profile = pd.DataFrame()

for name in all_agents:
    # Index des Knotens (Bus) anhand des Namens finden
    bus_idx = net.bus[net.bus.name == name].index[0]
    # Suche nach der Last, die an diesem Knoten angeschlossen ist
    load_search = net.load[net.load.bus == bus_idx]
    
    if not load_search.empty:
        base_p = load_search.p_mw.values[0]
        # Erzeugung eines glatteren Tag-Nacht-Profils (Sinuskurve)
        t = np.linspace(0, 4*np.pi, time_steps)
        wave = 1 + 0.3 * np.sin(t - np.pi/2) # Lastspitze (Peak) zur Mittagszeit
        load_profile[name] = base_p * wave
    else:
        # Falls keine Last definiert ist, mit Nullen auffüllen
        load_profile[name] = np.zeros(time_steps)

# Daten speichern
os.makedirs("data", exist_ok=True)
load_profile.to_csv("data/load_profile_forecast_18bus.csv", index=False)
print("Referenz-Lastprofildaten wurden unter data/load_profile_forecast_18bus.csv generiert.")