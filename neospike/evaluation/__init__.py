"""Search evaluation: one scoring core, a real-data and a synthetic-fixture source.

Modules: core (metrics, paired bootstrap, gate, threshold rule), lanes (product search lanes, sweep),
adapters (RealSource, SyntheticSource), protocol (seal, ledger), command (`neospike eval ...`).
"""
from .command import main, register_eval_cli, run_eval_cli

__all__ = ["main", "register_eval_cli", "run_eval_cli"]
