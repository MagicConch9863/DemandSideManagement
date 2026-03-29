import pandas as pd
import numpy as np
from pyomo.environ import *
import os

# ==========================================
# 1. Kernparameter-Konfiguration
# ==========================================
NUM_AGENTS = 18       # Anzahl der Agenten (Prosumer)
S_MAX = 1.0           # Maximale Lade-/Entladeleistung (kW/MW)
BATTERY_CAP = 5.0     # Batteriekapazität
SOC_INIT = 2.5        # Initialer Ladezustand (SoC)

ALPHA = 0.999         # Selbstentladungsrate / Effizienzfaktor
BETA = 0.95           # Ladeeffizienz
HORIZON_N = 16        # Planungshorizont (Vorausschau)

SOLVER_NAME = 'gurobi' # Name des Solvers

# ==========================================
# Pfadkonfiguration
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

DATA_FILE = os.path.join(current_dir, "data", "load_profile_forecast_18bus.csv")
OUTPUT_CSV = os.path.join(current_dir, "data", "duopoly_results_18bus.csv")

def load_real_data():
    """Lädt die realen Lastdaten aus der CSV-Datei."""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Datendatei nicht gefunden: {DATA_FILE}\nAktueller Suchpfad: {DATA_FILE}")
    
    df_raw = pd.read_csv(DATA_FILE)
    # Aggregiert die Last über alle Knoten und rechnet in kW um (falls nötig)
    total_load_kw = df_raw.sum(axis=1) * 1000.0
    return pd.DataFrame({'total_base_load': total_load_kw})

def solve_prosumer_response(current_soc, forecast_slice, rho_incentive, target_load):
    """
    Optimiert die Antwort der Prosumer basierend auf dem Anreizsignal (Leader-Follower).
    Dies entspricht der unteren Ebene des Stackelberg-Spiels.
    """
    model = ConcreteModel()
    model.I = RangeSet(0, NUM_AGENTS - 1)
    model.T = RangeSet(0, HORIZON_N - 1)
    
    # Entscheidungsvariablen: s = Leistung, q = Energieinhalt
    model.s = Var(model.I, model.T, bounds=(-S_MAX, S_MAX)) 
    model.q = Var(model.I, model.T, bounds=(0, BATTERY_CAP)) 

    # Nebenbedingung für die Batteriedynamik (SoC)
    def soc_con(m, i, t):
        if t == 0: 
            return m.q[i, t] == current_soc[i]
        # 0.25 steht für ein 15-Minuten-Intervall (1/4 Stunde)
        return m.q[i, t] == ALPHA * m.q[i, t-1] - 0.25 * BETA * m.s[i, t-1]
    model.soc_limit = Constraint(model.I, model.T, rule=soc_con)

    # Zielfunktion der Prosumer: Minimierung der Kosten & Abweichung vom Zielzustand
    def obj_rule(m):
        cost = 0
        for t in m.T:
            base_l = forecast_slice.iloc[t]
            net_l = base_l - sum(m.s[i, t] for i in m.I)
            # Kosten durch Anreizfaktor + Abnutzungskosten der Batterie (quadratisch)
            cost += rho_incentive * (net_l - target_load)**2 + 0.5 * sum(m.s[i, t]**2 for i in m.I)
        
        # Strafterm für den Endzustand des SoC (Vermeidung von Tiefentladung am Ende des Horizonts)
        terminal_penalty = 15.0 * sum((m.q[i, HORIZON_N-1] - SOC_INIT)**2 for i in m.I)
        return cost + terminal_penalty

    model.obj = Objective(rule=obj_rule, sense=minimize)
    
    solver = SolverFactory(SOLVER_NAME)
    solver.options['LogToConsole'] = 0
    solver.solve(model)
    
    # Rückgabe der optimalen Leistung für den aktuellen Zeitschritt
    return [value(model.s[i, 0]) for i in model.I]

def run_simulation():
    """Hauptfunktion zur Durchführung der Simulation."""
    data = load_real_data()
    # Ziel-Lastwert (Durchschnitt zur Glättung)
    system_target_load = data['total_base_load'].mean()
    print(f"Daten erfolgreich geladen! Zielwert für Lastglättung: {system_target_load:.2f} kW")
    
    total_steps = min(192, len(data) - HORIZON_N)
    soc_tracker = [SOC_INIT] * NUM_AGENTS
    results = []

    print(">>> [Gurobi] Berechne Stackelberg-Gleichgewicht (Duopol-Szenario)...")
    for t in range(total_steps):
        horizon_slice = data['total_base_load'].iloc[t : t + HORIZON_N]
        base_load_t = horizon_slice.iloc[0]
        
        best_rho, min_grid_cost, best_s = 0, float('inf'), [0] * NUM_AGENTS
        
        # Iterative Suche nach dem optimalen Anreizfaktor (Leader-Strategie)
        for trial_rho in np.linspace(0.0, 5.0, 20):
            s_t = solve_prosumer_response(soc_tracker, horizon_slice, trial_rho, system_target_load)
            net_l = base_load_t - sum(s_t)
            
            # Kostenfunktion des Netzbetreibers (Glättungsfehler + Kosten für Anreizstellung)
            total_cost = 2.0 * (net_l - system_target_load)**2 + 1.5 * trial_rho**2 
            
            if total_cost < min_grid_cost:
                min_grid_cost = total_cost
                best_rho = trial_rho
                best_s = s_t

        # Aktualisierung der SoC-Werte für alle Agenten nach dem Schritt
        for i in range(NUM_AGENTS):
            soc_tracker[i] = np.clip(ALPHA * soc_tracker[i] - 0.25 * BETA * best_s[i], 0, BATTERY_CAP)

        optimized_l = base_load_t - sum(best_s)
        results.append({
            'step': t,
            'base_load': base_load_t,
            'opt_load': optimized_l,
            'price_factor': best_rho
        })
        
        if t % 20 == 0:
            print(f"Schritt {t:03d} | Basis: {base_load_t:.1f} kW | Opt: {optimized_l:.1f} kW | Rho: {best_rho:.2f}")

    # ===== Ergebnis-Export als CSV =====
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Berechnung abgeschlossen! Ergebnisse gespeichert unter: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_simulation()