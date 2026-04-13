from configs.stackelberg_config import StackelbergConfig
from simulation.simulation_runner import run_stackelberg_simulation


def main():
    cfg = StackelbergConfig()
    cfg.validate()
    run_stackelberg_simulation(cfg)


if __name__ == "__main__":
    main()