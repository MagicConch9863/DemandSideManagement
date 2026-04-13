from dataclasses import dataclass


@dataclass
class StackelbergConfig:
    # =========================
    # Time
    # =========================
    simbench_code: str = "1-LV-urban6--0-sw"

    horizon = 96
    time_step_hours = 0.25

    rhg_window_steps = 18
    rhg_alpha = 0.999
    rhg_beta = 0.95

    rhg_weight_track = 0.1
    rhg_weight_ramp = 0.1
    rhg_weight_u = 0.05
    # =========================
    # Price CSV
    # =========================
    profile_csv_path: str = "data/price_data.csv"
    price_col: str = "price_eur_per_mwh"
    price_unit_is_eur_per_mwh: bool = True

    # =========================
    # Fixed device counts
    # =========================
    n_passive_load: int = 6
    n_pv_only: int = 6
    n_pv_battery: int = 6
    random_seed: int = 42

    # =========================
    # Battery
    # p_bat > 0 : charging
    # p_bat < 0 : discharging
    # =========================
    battery_capacity_kwh: float = 12.0
    battery_pmax_kw: float = 5.0
    soc_init_kwh: float = 6.0
    soc_min_kwh: float = 1.0
    soc_max_kwh: float = 12.0
    battery_cycle_cost_eur_per_kwh2: float = 0.01
    enforce_terminal_soc: bool = True

    # =========================
    # Leader price update
    # =========================
    price_min_eur_per_kwh: float = 0.01
    price_max_eur_per_kwh: float = 1.00
    leader_step_size: float = 0.1
    price_damping: float = 0.25
    price_smoothing_weight: float = 1.0

    # =========================
    # Leader objective
    # Lref will be computed automatically
    # from the baseline daily average PCC import
    # =========================
    target_import_kw: float | None = None
    weight_wholesale_cost: float = 1.0
    weight_peak_penalty: float = 180.0
    weight_voltage_penalty: float = 10.0
    weight_line_penalty: float = 10.0
    weight_trafo_penalty: float = 10.0

    # =========================
    # Iteration
    # =========================
    max_stackelberg_iter: int = 20
    price_convergence_tol: float = 1e-3

    # =========================
    # Solver / Output
    # =========================
    gurobi_output_flag: int = 0
    output_dir: str = "outputs"
    save_plots: bool = True
    debug_mode: bool = True

    def validate(self):
        if self.horizon <= 0:
            raise ValueError("horizon must be positive.")

        if self.time_step_hours <= 0:
            raise ValueError("time_step_hours must be positive.")

        total_nodes = self.n_passive_load + self.n_pv_only + self.n_pv_battery
        if total_nodes != 18:
            raise ValueError("The feeder must contain exactly 18 residential nodes.")

        if not (self.soc_min_kwh <= self.soc_init_kwh <= self.soc_max_kwh):
            raise ValueError("Invalid SOC bounds.")

        if not (0.0 < self.price_min_eur_per_kwh <= self.price_max_eur_per_kwh):
            raise ValueError("Invalid price bounds.")