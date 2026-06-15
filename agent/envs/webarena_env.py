"""
WebArena-Lite environment wrapper for FAILGROUND.

WebArena provides web navigation tasks grounded in factual entity knowledge.
This wrapper connects to a running WebArena server or falls back to stub mode.
"""

from __future__ import annotations
import json
import os
from typing import Optional


STUB_TASKS = [
    {
        "task_id": 0,
        "intent": "Find the capital city of France on the website",
        "sites": ["wikipedia"],
        "answer": "Paris",
    },
    {
        "task_id": 1,
        "intent": "Find who directed the film Inception on the movie database",
        "sites": ["imdb"],
        "answer": "Christopher Nolan",
    },
]


class WebArenaLiteEnv:
    """
    Wrapper for WebArena-Lite.
    Requires a running WebArena server (base_url from config).
    Falls back to stub mode if server not reachable.
    """

    def __init__(self, base_url: str = "http://localhost:7780", headless: bool = True):
        self.base_url = base_url
        self.headless = headless
        self._tasks = STUB_TASKS
        self._idx = 0
        self._current_task = {}
        self.success = False
        self._stub = True
        self._browser = None
        self._try_connect()

    def _try_connect(self):
        try:
            import requests
            resp = requests.get(f"{self.base_url}/health", timeout=3)
            if resp.status_code == 200:
                self._stub = False
                self._load_tasks()
        except Exception:
            print(f"[WebArenaEnv] Server not reachable at {self.base_url}, using stub mode")

    def _load_tasks(self):
        tasks_path = os.path.join(
            os.path.dirname(__file__),
            "../../benchmarks/visualwebarena/config_files/test.json"
        )
        if os.path.exists(tasks_path):
            with open(tasks_path) as f:
                self._tasks = json.load(f)

    def reset(self, idx: int = 0) -> str:
        self.success = False
        self._idx = idx % len(self._tasks)
        self._current_task = self._tasks[self._idx]
        intent = self._current_task.get("intent", "")
        return f"Task: {intent}\nURL: {self.base_url}"

    def step(self, action: str) -> str:
        if self._stub:
            answer = self._current_task.get("answer", "")
            self.success = answer.lower() in action.lower()
            return f"[stub] Action: {action}\nResult: {'success' if self.success else 'continue'}"
        # Real mode: use Playwright to execute action
        return self._execute_browser_action(action)

    def _execute_browser_action(self, action: str) -> str:
        # Placeholder for Playwright integration
        return f"Executed browser action: {action}"

    @property
    def task_text(self) -> str:
        return self._current_task.get("intent", "")

    def __len__(self):
        return len(self._tasks)
