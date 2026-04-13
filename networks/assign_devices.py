import random
import pandas as pd

from networks.build_base_network import build_base_network_and_data


def assign_devices(cfg):
    random.seed(cfg.random_seed)

    net, coords, bus_map, node_data = build_base_network_and_data(cfg)

    selected_bus_ids = node_data["bus_ids"]
    selected_bus_names = node_data["bus_names"]

    if len(selected_bus_names) != 18:
        raise ValueError("The residential feeder must contain exactly 18 nodes.")

    # fixed group sizes
    shuffled_names = selected_bus_names.copy()
    random.shuffle(shuffled_names)

    passive_names = shuffled_names[:cfg.n_passive_load]
    pv_only_names = shuffled_names[cfg.n_passive_load:cfg.n_passive_load + cfg.n_pv_only]
    active_names = shuffled_names[cfg.n_passive_load + cfg.n_pv_only:]

    if len(active_names) != cfg.n_pv_battery:
        raise ValueError("The number of PV+Battery nodes is inconsistent.")

    name_to_id = {name: bus_map[name] for name in selected_bus_names}

    for bus_name in selected_bus_names:
        bus_id = name_to_id[bus_name]

        has_pv = bus_name in pv_only_names or bus_name in active_names
        has_battery = bus_name in active_names

        if not has_pv:
            node_data["pv_kw"][bus_id] = [0.0] * cfg.horizon

        if has_battery:
            node_data["has_battery"][bus_id] = True
            node_data["battery_capacity_kwh"][bus_id] = cfg.battery_capacity_kwh
            node_data["battery_pmax_kw"][bus_id] = cfg.battery_pmax_kw
            node_data["energy_init_kwh"][bus_id] = cfg.soc_init_kwh
        else:
            node_data["has_battery"][bus_id] = False
            node_data["battery_capacity_kwh"][bus_id] = 0.0
            node_data["battery_pmax_kw"][bus_id] = 0.0
            node_data["energy_init_kwh"][bus_id] = 0.0

    records = []
    for bus_name in selected_bus_names:
        bus_id = name_to_id[bus_name]

        has_pv = 1 if (bus_name in pv_only_names or bus_name in active_names) else 0
        has_battery = 1 if bus_name in active_names else 0

        if bus_name in active_names:
            node_type = "active"
        elif bus_name in pv_only_names:
            node_type = "pv_only"
        else:
            node_type = "passive"

        records.append(
            {
                "bus_idx": bus_id,
                "bus_name": bus_name,
                "has_load": 1,
                "has_pv": has_pv,
                "has_battery": has_battery,
                "battery_kw": node_data["battery_pmax_kw"][bus_id],
                "battery_kwh": node_data["battery_capacity_kwh"][bus_id],
                "node_type": node_type,
            }
        )

    device_df = pd.DataFrame(records)

    groups = {
        "upstream": ["Bus 0", "Bus R0"],
        "passive": passive_names,
        "pv_only": pv_only_names,
        "active": active_names,
    }

    if cfg.debug_mode:
        print("\nDevice assignment:")
        print(f"Passive load nodes   : {passive_names}")
        print(f"PV-only nodes        : {pv_only_names}")
        print(f"PV+Battery nodes     : {active_names}")
        print(device_df[["bus_name", "has_pv", "has_battery", "node_type"]])

    return net, coords, groups, node_data, device_df, bus_map