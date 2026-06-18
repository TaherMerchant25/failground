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
from tools.entity_linking import extract_atomic_claims, extract_entities
from tools.schema_grounding import verify_step, is_adjudicable


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
        ablation: str = "none",
    ):
        self.fakg = fakg
        self.mast = mast
        self.llm_fn = llm_fn
        self.embed_fn = embed_fn
        self.conceptnet_dump = conceptnet_dump
        self.ablation = ablation

    def mine_episode(
        self,
        trajectory: list[dict],
        task_text: str,
        env: str,
        episode_idx: int,
    ) -> tuple[list[FPT], dict]:
        """
        trajectory: list of {"action": str, "observation": str, "context": str}
        Returns (fpts, stats) where stats keys: failed_steps, total_claims, sparql_covered.
        """
        task_embedding = self.embed_fn(task_text)
        task_id = f"task_{episode_idx}_{uuid.uuid4().hex[:8]}"
        mined: list[FPT] = []
        failed_steps = 0
        total_claims = 0
        schema_covered = 0
        schema_hallucinated_steps = 0

        for step in trajectory:
            action = step["action"]
            observation = step.get("observation", "")
            context = step.get("context", "")
            env_ctx = step.get("env_ctx", {})

            # use explicit flag from agent if available, else fall back to keyword heuristic
            env_fail = step.get("step_failed", is_env_failure(observation))
            if env_fail:
                failed_steps += 1

            # schema-grounded hallucination detection
            schema_flags = verify_step(action, env_ctx)
            schema_fail = len(schema_flags) > 0
            if is_adjudicable(env_ctx):
                schema_covered += 1
            if schema_fail:
                schema_hallucinated_steps += 1

            # action-level entities for MENTIONS indexing
            action_entities = extract_entities(action)

            claims = extract_atomic_claims(action, self.llm_fn)
            total_claims += len(claims)

            for claim in claims:
                if not isinstance(claim, dict):
                    continue
                h, r, t = claim.get("h", ""), claim.get("r", ""), claim.get("t", "")
                if not (h and r and t):
                    continue
                h, r, t = str(h), str(r), str(t)

                # pick correction hint from schema flags if available
                a_correct = t
                for flag in schema_flags:
                    if flag.get("h", "").lower() == h.lower() and flag.get("correct"):
                        a_correct = flag["correct"]
                        break

                self.fakg.record_attempt(h, r)

                if env_fail or schema_fail:
                    conf = self.fakg.get_confidence(h, r)
                    if self.ablation == "A1":
                        # plain (h,r,t): no failure-mode classification, no correction hint
                        gamma = "unverifiable"
                        a_correct = t
                    else:
                        gamma, _ = self.mast.classify(action, context, observation)

                    fpt = FPT(
                        h=h,
                        r=r,
                        t=t,
                        gamma=gamma,
                        theta_task=task_embedding,
                        t_ep=episode_idx,
                        c=conf,
                        a_correct=a_correct,
                    )
                    self.fakg.add_failure(
                        fpt, task_id, task_embedding, env,
                        mention_entities=action_entities,
                    )
                    mined.append(fpt)

        stats = {
            "failed_steps": failed_steps,
            "total_claims": total_claims,
            "schema_covered": schema_covered,
            "schema_hallucinated_steps": schema_hallucinated_steps,
        }
        return mined, stats
