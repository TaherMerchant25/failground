"""
Coverage-Complete Subgraph Retrieval (CCSR).

Before executing action a'_t:
  1. Extract entities E from the action text.
  2. For each e in E, retrieve 1-hop world-knowledge neighbors and all
     active FailureNodes where CAUSED_BY -> WorldNode(e).
  3. Compute Failure Coverage Score (FCS) = |{e with active failure neighbor}| / |E|.
  4. If FCS > theta AND task-similarity > delta, inject correction prompt.

Coverage guarantee: every e in E is checked if WorldNode(e) exists in FAKG.
"""

from __future__ import annotations
from dataclasses import dataclass

from fakg.fakg import FAKG
from fakg.fpt import FPT

FCS_THRESHOLD = 0.3
SIM_THRESHOLD = 0.7


@dataclass
class RetrievalResult:
    entities: list[str]
    fcs: float
    matched_fpts: list[FPT]
    inject_correction: bool
    correction_prompt: str


class CCSR:
    def __init__(
        self,
        fakg: FAKG,
        fcs_threshold: float = FCS_THRESHOLD,
        sim_threshold: float = SIM_THRESHOLD,
        retrieval_mode: str = "coverage",   # "coverage" | "topk" (A3)
        use_task_sim_filter: bool = True,   # False disables sim gate (A1, A6)
        use_provenance: bool = True,        # False -> flat (h)-only lookup (A2)
        disabled: bool = False,             # True -> never inject (baseline)
        topk_k: int = 3,
    ):
        self.fakg = fakg
        self.fcs_threshold = fcs_threshold
        self.sim_threshold = sim_threshold
        self.retrieval_mode = retrieval_mode
        self.use_task_sim_filter = use_task_sim_filter
        self.use_provenance = use_provenance
        self.disabled = disabled
        self.topk_k = topk_k

    def retrieve(
        self,
        action_text: str,
        entities: list[str],
        task_embedding: list[float],
    ) -> RetrievalResult:
        if self.disabled:
            return RetrievalResult(entities, 0.0, [], False, "")
        if not entities:
            return RetrievalResult([], 0.0, [], False, "")
        if self.retrieval_mode == "topk":
            return self._retrieve_topk(action_text, entities, task_embedding)
        return self._retrieve_coverage(action_text, entities, task_embedding)

    def _retrieve_coverage(
        self,
        action_text: str,
        entities: list[str],
        task_embedding: list[float],
    ) -> RetrievalResult:
        covered_entities: list[str] = []
        matched_fpts: list[FPT] = []

        for entity in entities:
            fpts = (
                self.fakg.failure_nodes_for_entity(entity)
                if self.use_provenance
                else self.fakg.flat_lookup(entity)
            )
            if fpts:
                covered_entities.append(entity)
                if self.use_task_sim_filter:
                    similar = [
                        fpt for fpt in fpts
                        if fpt.task_similarity(task_embedding) >= self.sim_threshold
                    ]
                else:
                    similar = fpts
                matched_fpts.extend(similar)

        fcs = len(covered_entities) / len(entities)
        inject = fcs >= self.fcs_threshold and len(matched_fpts) > 0

        prompt = self._build_correction_prompt(matched_fpts, action_text) if inject else ""

        return RetrievalResult(
            entities=entities,
            fcs=fcs,
            matched_fpts=matched_fpts,
            inject_correction=inject,
            correction_prompt=prompt,
        )

    def _retrieve_topk(
        self,
        action_text: str,
        entities: list[str],
        task_embedding: list[float],
    ) -> RetrievalResult:
        """A3 ablation: dense top-k over all active FPTs by task-embedding
        similarity, ignoring entity coverage / graph structure entirely."""
        all_fpts = self.fakg.active_failure_nodes()
        if not all_fpts:
            return RetrievalResult(entities, 0.0, [], False, "")
        scored = sorted(
            ((fpt, fpt.task_similarity(task_embedding)) for fpt in all_fpts),
            key=lambda x: -x[1],
        )
        top = [fpt for fpt, score in scored[: self.topk_k] if score >= self.sim_threshold]
        inject = len(top) > 0
        prompt = self._build_correction_prompt(top, action_text) if inject else ""
        return RetrievalResult(
            entities=entities,
            fcs=1.0 if inject else 0.0,
            matched_fpts=top,
            inject_correction=inject,
            correction_prompt=prompt,
        )

    def _build_correction_prompt(self, fpts: list[FPT], action_text: str) -> str:
        lines = [
            "FAILGROUND WARNING — Past failure patterns detected for this action:",
            f"  Action: {action_text}",
            "",
            "Known failure patterns:",
        ]
        seen: set[tuple] = set()
        for fpt in fpts:
            key = (fpt.h, fpt.r, fpt.t)
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"  - [{fpt.gamma}] '{fpt.h} {fpt.r} {fpt.t}' is WRONG. "
                f"Correct: '{fpt.h} {fpt.r} {fpt.a_correct}' "
                f"(confidence={fpt.c:.2f})"
            )
        lines += ["", "Verify these claims before proceeding."]
        return "\n".join(lines)
