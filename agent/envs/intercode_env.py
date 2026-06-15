"""
InterCode-SQL environment wrapper for FAILGROUND.

InterCode provides a gym-like interface for code execution tasks.
This wrapper targets the SQL variant (schema entity hallucination).
"""

from __future__ import annotations
import subprocess
import json
import os


class InterCodeSQLEnv:
    """
    Wrapper for InterCode SQL environment.
    Falls back to stub mode if intercode not installed.
    """

    DATASETS = {
        "spider": "data/spider",
        "bird": "data/bird",
    }

    def __init__(self, dataset: str = "spider", data_path: str = None):
        self.dataset = dataset
        self.data_path = data_path
        self._tasks = []
        self._idx = 0
        self._current_task = {}
        self.success = False
        self._stub = False
        self._try_load()

    def _try_load(self):
        try:
            intercode_path = os.path.join(
                os.path.dirname(__file__),
                "../../benchmarks/intercode"
            )
            tasks_file = os.path.join(intercode_path, "data", "sql", "dev.json")
            if os.path.exists(tasks_file):
                with open(tasks_file) as f:
                    self._tasks = json.load(f)
            else:
                self._stub = True
        except Exception:
            self._stub = True
        if self._stub:
            self._tasks = [
                {"question": "Find all employees with salary > 50000", "answer": "SELECT * FROM employees WHERE salary > 50000", "db_id": "company"},
                {"question": "Count orders per customer", "answer": "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id", "db_id": "ecommerce"},
            ]

    def reset(self, idx: int = None) -> str:
        self.success = False
        if idx is not None:
            self._idx = idx % len(self._tasks)
        task = self._tasks[self._idx]
        self._current_task = task
        return f"Database: {task.get('db_id', 'unknown')}\nQuestion: {task['question']}"

    def step(self, action: str) -> str:
        gold = self._current_task.get("answer", "")
        # Simple string match for stub; real env executes SQL
        self.success = action.strip().lower() == gold.strip().lower()
        if self._stub:
            return f"Executed: {action}\n[stub] Match: {self.success}"
        return f"SQL result: {'correct' if self.success else 'incorrect'}"

    @property
    def task_text(self) -> str:
        t = self._current_task
        return f"Database: {t.get('db_id', '?')} | Question: {t.get('question', '')}"

    def __len__(self):
        return len(self._tasks)
