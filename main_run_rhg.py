from __future__ import annotations

from configs.stackelberg_config import StackelbergConfig
from simulation.rhg_runner import run_rhg_simulation


def main():
    cfg = StackelbergConfig()
    cfg.validate()
    run_rhg_simulation(cfg)


if __name__ == "__main__":
    main()