"""
Adaptive Failure Memory (AFM).

When |FailureNodes| exceeds budget B, evict by composite priority:

  priority(fn) = alpha * freq(fn)
               + beta  * exp(-lambda * delta_ep)
               + gamma * task_sim_max(fn)

Evicted nodes enter ghost state (is_active=False) with edges preserved.
They reactivate if the same (h,r) failure recurs — preventing cold-start.

Ablation A4: FIFO eviction vs. composite priority.
Ablation A5: gamma=0 (drop task-similarity term).
"""

from __future__ import annotations
import math
from typing import Optional

from fakg.fakg import FAKG
from fakg.fpt import FPT


class AFM:
    def __init__(
        self,
        fakg: FAKG,
        budget: int = 500,
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        decay_lambda: float = 0.01,
        fifo_mode: bool = False,
    ):
        self.fakg = fakg
        self.budget = budget
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.decay_lambda = decay_lambda
        self.fifo_mode = fifo_mode
        self._insertion_order: list[str] = []

    def on_add(self, fpt: FPT, current_episode: int, task_embedding: list[float]):
        self._insertion_order.append(fpt.fpt_id)
        self._update_priority(fpt, current_episode, task_embedding)
        if len(self.fakg.active_failure_nodes()) > self.budget:
            self._evict(current_episode, task_embedding)

    def _update_priority(self, fpt: FPT, current_episode: int, task_embedding: list[float]):
        freq = self.fakg.get_confidence(fpt.h, fpt.r)
        delta_ep = max(0, current_episode - fpt.t_ep)
        recency = math.exp(-self.decay_lambda * delta_ep)
        sim = fpt.task_similarity(task_embedding)
        fpt.priority = self.alpha * freq + self.beta * recency + self.gamma * sim
        if self.fakg.G.has_node(fpt.fpt_id):
            self.fakg.G.nodes[fpt.fpt_id]["priority"] = fpt.priority

    def _evict(self, current_episode: int, task_embedding: list[float]):
        active = self.fakg.active_failure_nodes()
        if not active:
            return

        if self.fifo_mode:
            # Evict oldest insertion
            for fpt_id in self._insertion_order:
                fpt = next((f for f in active if f.fpt_id == fpt_id), None)
                if fpt:
                    self.fakg.deactivate(fpt.fpt_id)
                    return
        else:
            # Refresh priorities then evict lowest
            for fpt in active:
                self._update_priority(fpt, current_episode, task_embedding)
            victim = min(active, key=lambda f: f.priority)
            self.fakg.deactivate(victim.fpt_id)

    def reactivate_if_seen(self, h: str, r: str):
        """Called when (h,r) fails again — reactivate matching ghost nodes."""
        self.fakg.reactivate(h, r)

    def stats(self) -> dict:
        active = self.fakg.active_failure_nodes()
        total = len(self.fakg._fpt_index)
        return {
            "active": len(active),
            "ghost": total - len(active),
            "total": total,
            "budget": self.budget,
            "churn_rate": (total - len(active)) / max(total, 1),
        }
