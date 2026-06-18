"""
ALFWorld text environment wrapper for FAILGROUND.

Wraps ALFWorld TextWorld env into the simple interface expected by
FAILGROUNDReActAgent: reset() -> str, step(action) -> str, success -> bool.
"""

from __future__ import annotations
import re
from typing import Optional


class ALFWorldEnv:
    """
    Thin wrapper around alfworld.agents.environment.AlfredTWEnv.
    Falls back to a stub mode if alfworld is not installed.
    """

    def __init__(self, split: str = "eval_out_of_distribution", batch_size: int = 1):
        self.split = split
        self.batch_size = batch_size
        self._env = None
        self._obs = ""
        self._info = {}
        self.success = False
        self._task_text = ""
        self._done = False

    def _lazy_init(self):
        if self._env is not None:
            return
        try:
            import alfworld.agents.environment as alf_env
            import yaml, os
            config_path = os.path.join(
                os.path.dirname(__file__),
                "../../benchmarks/alfworld/configs/base_config.yaml"
            )
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            # point at downloaded game data
            cfg["dataset"]["data_path"] = os.path.expanduser("~/.cache/alfworld/json_2.1.1")
            EnvClass = alf_env.get_environment("AlfredTWEnv")
            env = EnvClass(cfg, train_eval=self.split)
            self._env = env.init_env(batch_size=self.batch_size)
        except Exception as e:
            self._env = "stub"
            print(f"[ALFWorldEnv] alfworld unavailable ({e}), using stub mode")

    def reset(self) -> str:
        self._lazy_init()
        self.success = False
        self._done = False
        if self._env == "stub":
            self._task_text = "pick up the apple and put it in the fridge"
            self._obs = f"You are in a kitchen. Task: {self._task_text}"
            self._admissible: list[str] = []
            return self._obs
        obs, info = self._env.reset()
        self._obs = obs[0]
        self._info = info
        self._admissible = list(info.get("admissible_commands", [[]])[0])
        # Extract task from admissible commands or obs
        self._task_text = self._obs.split("\n")[0]
        return self._obs

    def step(self, action: str) -> str:
        if self._env == "stub":
            self._done = "apple" in action.lower() and "fridge" in action.lower()
            self.success = self._done
            self._admissible: list[str] = []
            return "Task completed." if self.success else f"You tried: {action}. Nothing happened."
        obs, scores, dones, info = self._env.step([action])
        self._obs = obs[0]
        self._done = bool(dones[0])
        self.success = self._done and float(scores[0]) > 0
        self._admissible = list(info.get("admissible_commands", [[]])[0])
        return self._obs

    def grounding_context(self) -> dict:
        return {"type": "alfworld", "admissible": getattr(self, "_admissible", [])}

    @property
    def task_text(self) -> str:
        return self._task_text
