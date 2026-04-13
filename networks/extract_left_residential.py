import pandapower as pp
import pandapower.toolbox as tb

from networks.build_base_network import load_simbench_network


def extract_left_residential(simbench_code: str = "1-LV-rural1--0-sw"):
    """
    Extract the left residential feeder from the SimBench LV feeder.
    Keeps:
        - Bus 0
        - Bus R0
        - Bus R1 ... Bus R18
    """
    net = load_simbench_network(simbench_code)

    keep_prefixes = ["Bus 0", "Bus R"]
    keep_bus_idx = net.bus[
        net.bus["name"].fillna("").apply(
            lambda x: any(str(x).startswith(prefix) for prefix in keep_prefixes)
        )
    ].index.tolist()

    drop_bus_idx = [idx for idx in net.bus.index if idx not in keep_bus_idx]
    tb.drop_buses(net, drop_bus_idx, drop_elements=True)

    try:
        pp.runpp(net)
    except Exception as exc:
        print(f"Power flow failed on extracted residential subnetwork: {exc}")

    return net
