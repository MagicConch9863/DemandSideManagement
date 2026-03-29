import pandapower as pp
import pandapower.networks as nw
from pyomo.environ import *
import pandas as pd
import numpy as np
import os
import sys

# ==========================================
# 1. Konfigurationsparameter
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(current_dir, "data", "load_profile_forecast_18bus.csv")

HORIZON_N = 20         # Planungshorizont für den Rolling Horizon
NUM_AGENTS = 14       # Anzahl der Agenten (Speichersysteme)
SOLVER_NAME = "gurobi"

# Erweiterung auf 48h = 192 Zeitpunkte (bei 15-Minuten-Intervallen)
TOTAL_SIM_STEPS = 192

# Physikalische Batteriekonstanten
BATTERY_CAP = 0.015    # Kapazität in MWh
SOC_INIT = 0.0075      # Initialer SoC in MWh (50%)
ALPHA = 0.999          # Selbstentladungsfaktor / Effizienz der Erhaltung
BETA = 0.95           # Lade-/Entladeeffizienz
S_MAX = 0.015          # Maximale Leistung in MW

# ==========================================
# 2. Datenvorverarbeitung: Lastprofile auf 192 Punkte erweitern
# ==========================================
def prepare_forecast_48h(df: pd.DataFrame, total_steps: int = 192) -> pd.DataFrame:
    """
    Passt die Eingangsdaten auf die gewünschte Anzahl an Zeitschritten an.
    """
    df = df.copy()

    # Falls die Länge bereits passt
    if len(df) == total_steps:
        return df.reset_index(drop=True)

    # Falls 96 Punkte (24h) vorliegen, auf 48h verdoppeln
    if len(df) == 96 and total_steps == 192:
        df_48h = pd.concat([df, df], ignore_index=True)
        return df_48h

    # Für andere Längen: Spaltenweise lineare Interpolation auf total_steps
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Keine numerischen Spalten in forecast_df gefunden, Interpolation nicht möglich.")

    x_old = np.linspace(0, 1, len(df))
    x_new = np.linspace(0, 1, total_steps)

    out = pd.DataFrame()
    for col in df.columns:
        if col in numeric_cols:
            out[col] = np.interp(x_new, x_old, df[col].values)
        else:
            # Nicht-numerische Spalten werden ignoriert oder übernommen
            pass

    return out.reset_index(drop=True)


# ==========================================
# 3. Rolling-Horizon-Fenster: Auffüllen am Ende der Daten
# ==========================================
def get_horizon_slice(df: pd.DataFrame, t_now: int, horizon_n: int) -> pd.DataFrame:
    """
    Extrahiert ein Datenfenster. Falls am Ende nicht genug Daten vorhanden sind, 
    wird die letzte Zeile zur Auffüllung wiederholt.
    """
    slice_df = df.iloc[t_now: t_now + horizon_n].copy()

    if len(slice_df) < horizon_n:
        last_row = slice_df.iloc[[-1]].copy()
        pad_count = horizon_n - len(slice_df)
        pad_df = pd.concat([last_row] * pad_count, ignore_index=True)
        slice_df = pd.concat([slice_df.reset_index(drop=True), pad_df], ignore_index=True)

    return slice_df


# ==========================================
# 4. Optimierungsmodell
# ==========================================
def solve_rhg_optimization(current_soc, forecast_slice, num_agents, N):
    model = ConcreteModel()
    model.I = RangeSet(0, num_agents - 1)
    model.T = RangeSet(0, N - 1)

    # Entscheidungsvariablen: s = Leistung (MW), q = Energieinhalt (MWh)
    model.s = Var(model.I, model.T, bounds=(-S_MAX, S_MAX))
    model.q = Var(model.I, model.T, bounds=(0, BATTERY_CAP))

    # Batterie-Dynamik (SoC-Gleichung)
    def soc_dynamics(m, i, t):
        if t == 0:
            return m.q[i, t] == current_soc[i]
        # 0.25 entspricht 15 Minuten (1/4 Stunde)
        return m.q[i, t] == ALPHA * m.q[i, t - 1] + 0.25 * BETA * m.s[i, t - 1]

    model.soc_con = Constraint(model.I, model.T, rule=soc_dynamics)

    # Zielfunktion: Glättung der Gesamtlast + Minimierung der Batterieaktivität
    def objective_rule(m):
        total_obj = 0.0
        forecast_array = forecast_slice.values
        avg_load_ref = np.sum(forecast_array) / N

        for t in m.T:
            base_load_t = np.sum(forecast_array[t, :])
            net_load = base_load_t + sum(m.s[i, t] for i in m.I)

            # Hauptziel: Peak Shaving / Annäherung an den Durchschnittswert
            total_obj += (net_load - avg_load_ref) ** 2 * 1000

            # Nebenziel: Vermeidung unnötig hoher Lade-/Entladeleistungen
            total_obj += 0.1 * sum(m.s[i, t] ** 2 for i in m.I)

        return total_obj

    model.obj = Objective(rule=objective_rule, sense=minimize)

    solver = SolverFactory(SOLVER_NAME)
    solver.solve(model, tee=False)

    # Rückgabe nur des Steuersignals für den ersten Zeitschritt (MPC-Prinzip)
    u_opt = []
    for i in range(num_agents):
        val = value(model.s[i, 0])
        if val is None:
            val = 0.0
        u_opt.append(float(val))

    return u_opt


# ==========================================
# 5. Hauptsimulationsschleife
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"Fehler: Datendatei {DATA_FILE} nicht gefunden. Bitte führen Sie zuerst das Vorbereitungsskript aus.")
        sys.exit()

    # Initialisierung des CIGRE Niederspannungsnetzes
    net = nw.create_cigre_network_lv()

    # Installation von 14 Speichersystemen an definierten Bussen
    agent_bus_indices = []
    for i in range(1, NUM_AGENTS + 1):
        idx = net.bus[net.bus.name == f"Bus R{i}"].index[0]
        agent_bus_indices.append(idx)
        pp.create_storage(net, bus=idx, p_mw=0.0, max_e_mwh=BATTERY_CAP)

    # Laden der Prognosedaten
    forecast_df_raw = pd.read_csv(DATA_FILE)

    # Erweiterung auf 48 Stunden (192 Zeitschritte)
    forecast_df = prepare_forecast_48h(forecast_df_raw, TOTAL_SIM_STEPS)

    # Startwerte für den State of Charge (SoC)
    soc_tracker = [SOC_INIT] * NUM_AGENTS

    # Datenstrukturen zur Aufzeichnung der Ergebnisse
    history = {
        "time_step": [],
        "base_load_mw": [],
        "opt_load_mw": [],
        "transformer_loading_percent": [],
        "battery_sum_mw": []
    }

    print(f">>> Starte RHG-Simulation für 48 Stunden ({TOTAL_SIM_STEPS} Zeitschritte)...")

    for t_now in range(TOTAL_SIM_STEPS):
        # Extraktion des aktuellen Planungsfensters
        horizon_slice = get_horizon_slice(forecast_df, t_now, HORIZON_N)

        # Berechnung der optimalen Steuerung für den aktuellen Zeitpunkt
        u_opt = solve_rhg_optimization(soc_tracker, horizon_slice, NUM_AGENTS, HORIZON_N)

        # Aktualisierung des physikalischen Systems
        batt_sum = 0.0
        for i in range(NUM_AGENTS):
            net.storage.at[i, "p_mw"] = float(u_opt[i])

            # SoC-Update und Berücksichtigung physikalischer Grenzen
            soc_new = ALPHA * soc_tracker[i] + 0.25 * BETA * u_opt[i]
            soc_tracker[i] = float(np.clip(soc_new, 0.0, BATTERY_CAP))

            batt_sum += float(u_opt[i])

        # Durchführung der Lastflussrechnung (Power Flow)
        pp.runpp(net)

        # Ermittlung der Basislast und der optimierten Gesamtlast
        actual_base = float(forecast_df.iloc[t_now].sum())
        optimized_load = actual_base + batt_sum

        # Ergebnisse speichern
        history["time_step"].append(t_now)
        history["base_load_mw"].append(actual_base)
        history["opt_load_mw"].append(optimized_load)
        history["transformer_loading_percent"].append(float(net.res_trafo.loading_percent.values[0]))
        history["battery_sum_mw"].append(batt_sum)

        # Statusmeldung alle 12 Schritte
        if t_now % 12 == 0:
            print(
                f"t={t_now:03d} | "
                f"Basislast: {actual_base:.4f} MW | "
                f"Optimiert: {optimized_load:.4f} MW | "
                f"Batt-Summe: {batt_sum:.4f} MW | "
                f"Trafo-Last: {history['transformer_loading_percent'][-1]:.2f}%"
            )

    # Speichern der Simulationsergebnisse in einer CSV-Datei
    res_df = pd.DataFrame(history)
    output_path = os.path.join(current_dir, "data", "simulation_results_rhg_18bus.csv")
    res_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSimulation erfolgreich abgeschlossen! Ergebnisse gespeichert unter: {output_path}")