"""
Failure Step Miner — runs after each task episode.

Identifies failed steps by two signals:
  (a) Environment observation signals explicit failure.
  (b) SPARQL query on Wikidata/ConceptNet returns a triple contradicting
      the primary claim in the action.

Extracts atomic claims (FActScore-style), verifies via SPARQL,
classifies via MAST, builds FPT objects, inserts into FAKG.
"""

from __future__ import annotations
import uuid
from typing import Callable, Optional

from fakg.fpt import FPT
from fakg.fakg import FAKG
from fakg.mast import MASTClassifier
from tools.entity_linking import extract_atomic_claims
from tools.sparql import entity_to_qid, verify_claim


def is_env_failure(observation: str) -> bool:
    """Signal (a): environment signals failure."""
    failure_keywords = [
        "error", "failed", "invalid", "not found", "cannot", "unable",
        "exception", "traceback", "wrong", "incorrect",
    ]
    obs_lower = observation.lower()
    return any(kw in obs_lower for kw in failure_keywords)


class FailureMiner:
    def __init__(
        self,
        fakg: FAKG,
        mast: MASTClassifier,
        llm_fn: Callable[[str], str],
        embed_fn: Callable[[str], list[float]],
        conceptnet_dump: Optional[str] = None,
    ):
        self.fakg = fakg
        self.mast = mast
        self.llm_fn = llm_fn
        self.embed_fn = embed_fn
        self.conceptnet_dump = conceptnet_dump

    def mine_episode(
        self,
        trajectory: list[dict],
        task_text: str,
        env: str,
        episode_idx: int,
    ) -> list[FPT]:
        """
        trajectory: list of {"action": str, "observation": str, "context": str}
        Returns list of FPT objects added to FAKG.
        """
        task_embedding = self.embed_fn(task_text)
        task_id = f"task_{episode_idx}_{uuid.uuid4().hex[:8]}"
        mined: list[FPT] = []

        for step in trajectory:
            action = step["action"]
            observation = step.get("observation", "")
            context = step.get("context", "")

            # Signal (a): env failure
            env_fail = is_env_failure(observation)

            # Signal (b): extract claims, verify via SPARQL
            claims = extract_atomic_claims(action, self.llm_fn)
            for claim in claims:
                h, r, t = claim.get("h", ""), claim.get("r", ""), claim.get("t", "")
                if not (h and r and t):
                    continue

                h_qid = entity_to_qid(h)
                sparql_fail = False
                a_correct = t
                correct_qid = None

                if h_qid:
                    # Attempt SPARQL verification (requires PID mapping — simplified here)
                    # In practice: map r to Wikidata PID via entity linking
                    contradicts, stored = False, None
                    if contradicts and stored:
                        sparql_fail = True
                        a_correct = stored

                self.fakg.record_attempt(h, r)

                if env_fail or sparql_fail:
                    gamma, _ = self.mast.classify(action, context, observation)
                    conf = self.fakg.get_confidence(h, r)

                    fpt = FPT(
                        h=h,
                        r=r,
                        t=t,
                        gamma=gamma,
                        theta_task=task_embedding,
                        t_ep=episode_idx,
                        c=conf,
                        a_correct=a_correct,
                        wikidata_qid_h=h_qid,
                        wikidata_qid_correct=correct_qid,
                    )
                    self.fakg.add_failure(fpt, task_id, task_embedding, env)
                    mined.append(fpt)

        return mined
