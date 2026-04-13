import copy
import numpy as np
import pandapower as pp


def apply_time_step_net_injections(net_base, selected_bus_ids, follower_results, t):
    net_t = copy.deepcopy(net_base)

    for bus_id in selected_bus_ids:
        p_grid_kw = follower_results[bus_id].p_grid_kw[t]

        if p_grid_kw > 1e-9:
            pp.create_load(
                net_t,
                bus=bus_id,
                p_mw=p_grid_kw / 1000.0,
                q_mvar=0.0,
                name=f"load_bus_{bus_id}_t_{t}",
            )
        elif p_grid_kw < -1e-9:
            pp.create_sgen(
                net_t,
                bus=bus_id,
                p_mw=(-p_grid_kw) / 1000.0,
                q_mvar=0.0,
                name=f"sgen_bus_{bus_id}_t_{t}",
            )

    return net_t


def _safe_runpp(net_t):
    try:
        pp.runpp(net_t, init="auto", numba=False)
        return True
    except Exception as exc:
        print(f"[PowerFlow] runpp failed: {exc}")
        return False


def _extract_ext_grid_import_kw(net_t):
    if len(net_t.res_ext_grid) == 0:
        return 0.0
    p_sum_mw = float(net_t.res_ext_grid["p_mw"].sum())
    return abs(p_sum_mw) * 1000.0


def run_time_series_powerflow(net_base, selected_bus_ids, follower_results, horizon, cfg):
    pf_results = {
        "bus_vm_pu": [],
        "line_loading_percent": [],
        "trafo_loading_percent": [],
        "grid_import_from_followers_kw": [],
        "grid_export_from_followers_kw": [],
        "grid_import_from_ext_grid_kw": [],
        "powerflow_success": [],
    }

    for t in range(horizon):
        net_t = apply_time_step_net_injections(net_base, selected_bus_ids, follower_results, t)

        agg_import_kw = sum(max(follower_results[bus_id].p_grid_kw[t], 0.0) for bus_id in selected_bus_ids)
        agg_export_kw = sum(max(-follower_results[bus_id].p_grid_kw[t], 0.0) for bus_id in selected_bus_ids)

        success = _safe_runpp(net_t)

        if success:
            vm = net_t.res_bus.vm_pu.to_numpy(dtype=float)
            line_loading = net_t.res_line.loading_percent.to_numpy(dtype=float) if len(net_t.line) > 0 else np.array([])
            trafo_loading = net_t.res_trafo.loading_percent.to_numpy(dtype=float) if len(net_t.trafo) > 0 else np.array([])
            ext_grid_import_kw = _extract_ext_grid_import_kw(net_t)
        else:
            vm = np.array([])
            line_loading = np.array([])
            trafo_loading = np.array([])
            ext_grid_import_kw = agg_import_kw

        pf_results["bus_vm_pu"].append(vm)
        pf_results["line_loading_percent"].append(line_loading)
        pf_results["trafo_loading_percent"].append(trafo_loading)
        pf_results["grid_import_from_followers_kw"].append(float(agg_import_kw))
        pf_results["grid_export_from_followers_kw"].append(float(agg_export_kw))
        pf_results["grid_import_from_ext_grid_kw"].append(float(ext_grid_import_kw))
        pf_results["powerflow_success"].append(success)

        if getattr(cfg, "debug_mode", False):
            print(
                f"[PowerFlow] t={t:02d}, "
                f"followers_import={agg_import_kw:.3f} kW, "
                f"followers_export={agg_export_kw:.3f} kW, "
                f"ext_grid_import={ext_grid_import_kw:.3f} kW, "
                f"success={success}"
            )

    return pf_results


def compute_network_penalty(pf_results, cfg):
    voltage_penalty = 0.0
    line_penalty = 0.0
    trafo_penalty = 0.0
    peak_penalty = 0.0

    pcc_import_signal = pf_results["grid_import_from_ext_grid_kw"]

    # 允许 target_import_kw 在 baseline 阶段为 None
    target_import_kw = getattr(cfg, "target_import_kw", None)

    for t in range(len(pcc_import_signal)):
        vm = pf_results["bus_vm_pu"][t]
        line_loading = pf_results["line_loading_percent"][t]
        trafo_loading = pf_results["trafo_loading_percent"][t]
        pcc_import_kw = pcc_import_signal[t]

        for val in vm:
            if val < 0.95:
                voltage_penalty += (0.95 - val) ** 2
            elif val > 1.05:
                voltage_penalty += (val - 1.05) ** 2

        for val in line_loading:
            if val > 100.0:
                line_penalty += ((val - 100.0) / 100.0) ** 2

        for val in trafo_loading:
            if val > 100.0:
                trafo_penalty += ((val - 100.0) / 100.0) ** 2

        if target_import_kw is not None:
            if pcc_import_kw > target_import_kw:
                peak_penalty += ((pcc_import_kw - target_import_kw) / max(target_import_kw, 1.0)) ** 2

    total_penalty = (
        cfg.weight_voltage_penalty * voltage_penalty
        + cfg.weight_line_penalty * line_penalty
        + cfg.weight_trafo_penalty * trafo_penalty
        + cfg.weight_peak_penalty * peak_penalty
    )

    return {
        "voltage_penalty": voltage_penalty,
        "line_penalty": line_penalty,
        "trafo_penalty": trafo_penalty,
        "peak_penalty": peak_penalty,
        "total_network_penalty": total_penalty,
    }