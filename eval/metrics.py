"""
Evaluation metrics for FAILGROUND.

  TSR  — Task Success Rate: binary task completion
  SHR  — Step Hallucination Rate: % steps with Wikidata/ConceptNet-contradicting claim
  GCS  — Grounded Coverage Score: fraction of entities per step covered by SPARQL check
  CP   — Correction Precision: % injected corrections that prevented downstream failure
  FAKG Growth Rate    — new FailureNodes per episode
  Memory Churn Rate   — ghost nodes / total nodes
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    hallucinated_steps: int = 0
    total_entities_checked: int = 0
    entities_covered: int = 0
    corrections_injected: int = 0
    corrections_effective: int = 0


@dataclass
class BenchmarkMetrics:
    tsr: float = 0.0
    shr: float = 0.0
    gcs: float = 0.0
    cp: float = 0.0
    fakg_growth_rate: float = 0.0
    memory_churn_rate: float = 0.0
    n_episodes: int = 0
    results: list[EpisodeResult] = field(default_factory=list)

    def __str__(self):
        return (
            f"TSR={self.tsr:.3f}  SHR={self.shr:.3f}  "
            f"GCS={self.gcs:.3f}  CP={self.cp:.3f}  "
            f"FAKG_growth={self.fakg_growth_rate:.2f}  churn={self.memory_churn_rate:.3f}"
        )


def compute_metrics(results: list[EpisodeResult], fakg_stats: dict) -> BenchmarkMetrics:
    n = len(results)
    if n == 0:
        return BenchmarkMetrics()

    tsr = sum(r.success for r in results) / n

    all_steps = sum(r.steps for r in results)
    shr = sum(r.hallucinated_steps for r in results) / max(all_steps, 1)

    gcs = sum(r.entities_covered for r in results) / max(
        sum(r.total_entities_checked for r in results), 1
    )

    total_injected = sum(r.corrections_injected for r in results)
    cp = sum(r.corrections_effective for r in results) / max(total_injected, 1)

    return BenchmarkMetrics(
        tsr=tsr,
        shr=shr,
        gcs=gcs,
        cp=cp,
        fakg_growth_rate=fakg_stats.get("growth_rate", 0.0),
        memory_churn_rate=fakg_stats.get("churn_rate", 0.0),
        n_episodes=n,
        results=results,
    )
