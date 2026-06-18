"""
InterCode-SQL environment wrapper for FAILGROUND.

InterCode provides a gym-like interface for code execution tasks.
This wrapper targets the SQL variant (schema entity hallucination).
"""

from __future__ import annotations
import subprocess
import json
import os
import sqlite3

_DBS_DIR = os.path.join(
    os.path.dirname(__file__), "../../benchmarks/intercode/data/sql/spider/dbs"
)


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
        intercode_path = os.path.join(os.path.dirname(__file__), "../../benchmarks/intercode")
        candidates = [
            os.path.join(intercode_path, "data", "sql", "spider", "ic_spider_dev.json"),
            os.path.join(intercode_path, "data", "sql", "wikisql", "ic_wikisql_dev.json"),
            os.path.join(intercode_path, "data", "sql", "bird", "ic_bird.json"),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        raw = json.load(f)
                    self._tasks = [
                        {
                            "question": t.get("query", t.get("question", "")),
                            "answer":   t.get("gold",  t.get("answer", "")),
                            "db":       t.get("db",    t.get("db_id", "unknown")),
                            "db_tables": t.get("db_tables", {}),
                        }
                        for t in raw
                    ]
                    self._stub = False
                    print(f"[InterCode] loaded {len(self._tasks)} tasks from {os.path.basename(path)}")
                    return
                except Exception:
                    continue
        self._stub = True
        self._tasks = [
            {"question": "Find employees with salary > 50000",
             "answer": "SELECT * FROM employees WHERE salary > 50000", "db": "company"},
            {"question": "Count orders per customer",
             "answer": "SELECT customer_id, COUNT(*) FROM orders GROUP BY customer_id", "db": "ecommerce"},
        ]

    def reset(self, idx: int = None) -> str:
        self.success = False
        if idx is not None:
            self._idx = idx % len(self._tasks)
        task = self._tasks[self._idx]
        self._current_task = task
        return f"Database: {task.get('db', 'unknown')}\nQuestion: {task['question']}"

    def _execute(self, db_name: str, sql: str):
        """Run sql against the db's sqlite file. Returns (rows_set, error_str)."""
        db_path = os.path.join(_DBS_DIR, f"{db_name}.sqlite")
        if not os.path.exists(db_path):
            return None, f"no sqlite db for '{db_name}'"
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(sql)
            rows = frozenset(tuple(r) for r in cur.fetchall())
            conn.close()
            return rows, None
        except Exception as e:
            return None, str(e)

    def step(self, action: str) -> str:
        db_name = self._current_task.get("db", "unknown")
        gold = self._current_task.get("answer", "")

        agent_rows, agent_err = self._execute(db_name, action)
        if agent_err:
            self.success = False
            return f"SQL error: {agent_err}"

        gold_rows, gold_err = self._execute(db_name, gold)
        if gold_err:
            # gold itself can't run against converted db (rare schema-conversion gap);
            # fall back to exact-string match so the episode isn't silently unwinnable
            self.success = action.strip().lower() == gold.strip().lower()
        else:
            self.success = agent_rows == gold_rows

        return f"SQL executed: {action}\nRows returned: {len(agent_rows)}\nMatch: {self.success}"

    def grounding_context(self) -> dict:
        schema = self._current_task.get("db_tables", {})
        return {"type": "sql", "schema": schema}

    @property
    def task_text(self) -> str:
        t = self._current_task
        return f"Database: {t.get('db', '?')} | Question: {t.get('question', '')}"

    def __len__(self):
        return len(self._tasks)
