"""
Failure-Provenance Triple (FPT) — core data structure.

FPT = (h, r, t, gamma, theta_task, t_ep, c, a_correct)

Fields:
  h           — head entity (wrong claim subject)
  r           — relation
  t           — claimed (wrong) tail entity
  gamma       — MAST failure mode label (one of 14 modes in GAMMA)
  theta_task  — Sentence-BERT embedding of task at failure time
  t_ep        — episode index for temporal decay in AFM
  c           — confidence = #times(h,r) failed / #times attempted
  a_correct   — correct tail entity from Wikidata / ConceptNet
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import numpy as np
import uuid

GAMMA = [
    "hallucinated_entity",
    "wrong_relation",
    "wrong_attribute",
    "temporal_confusion",
    "spatial_confusion",
    "causal_reversal",
    "quantitative_error",
    "missing_context",
    "role_confusion",
    "negation_failure",
    "coreference_error",
    "schema_violation",
    "out_of_scope",
    "unverifiable",
]


@dataclass
class FPT:
    h: str
    r: str
    t: str
    gamma: str
    theta_task: list[float]
    t_ep: int
    c: float
    a_correct: str
    fpt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    is_active: bool = True
    priority: float = 0.0
    wikidata_qid_h: Optional[str] = None
    wikidata_qid_correct: Optional[str] = None
    conceptnet_uri_h: Optional[str] = None

    def __post_init__(self):
        assert self.gamma in GAMMA, f"Unknown failure mode: {self.gamma}"
        assert 0.0 <= self.c <= 1.0, f"Confidence must be in [0,1], got {self.c}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["theta_task"] = list(self.theta_task)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FPT":
        d["theta_task"] = list(d["theta_task"])
        return cls(**d)

    def task_similarity(self, query_embedding: list[float]) -> float:
        a = np.array(self.theta_task)
        b = np.array(query_embedding)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> "FPT":
        return cls.from_dict(json.loads(line))
