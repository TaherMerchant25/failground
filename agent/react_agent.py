"""
ReAct agent with FAILGROUND step-level correction.

On each step:
  1. Extract entities from planned action.
  2. Run CCSR against FAKG.
  3. If correction injected, prepend correction prompt to LLM input.
  4. Execute action, get observation.
  5. Post-episode: run FailureMiner, update FAKG, run AFM pruning.
"""

from __future__ import annotations
from typing import Callable, Optional

from fakg.fakg import FAKG
from fakg.ccsr import CCSR
from fakg.afm import AFM
from tools.entity_linking import extract_entities


class FAILGROUNDReActAgent:
    def __init__(
        self,
        fakg: FAKG,
        ccsr: CCSR,
        afm: AFM,
        llm_fn: Callable[[str], str],
        embed_fn: Callable[[str], list[float]],
        max_steps: int = 20,
    ):
        self.fakg = fakg
        self.ccsr = ccsr
        self.afm = afm
        self.llm_fn = llm_fn
        self.embed_fn = embed_fn
        self.max_steps = max_steps
        self.trajectory: list[dict] = []

    def run(self, task: str, env) -> dict:
        """Run one episode. env must implement .step(action) -> observation and .success -> bool."""
        self.trajectory = []
        task_embedding = self.embed_fn(task)
        observation = env.reset()
        history = f"Task: {task}\nObservation: {observation}\n"

        for step_idx in range(self.max_steps):
            # 1. Plan action
            planned_action = self.llm_fn(history + "Action:")

            # 2. CCSR lookup
            entities = extract_entities(planned_action)
            retrieval = self.ccsr.retrieve(planned_action, entities, task_embedding)

            # 3. Inject correction if warranted
            if retrieval.inject_correction:
                corrected_prompt = (
                    history
                    + retrieval.correction_prompt
                    + "\nGiven the above warnings, reconsider and provide action:"
                )
                planned_action = self.llm_fn(corrected_prompt)

            # 4. Execute
            observation = env.step(planned_action)
            self.trajectory.append({
                "action": planned_action,
                "observation": observation,
                "context": history[-500:],
                "ccsr_fcs": retrieval.fcs,
                "correction_injected": retrieval.inject_correction,
            })
            history += f"Action: {planned_action}\nObservation: {observation}\n"

            if env.success:
                break

        return {
            "success": env.success,
            "steps": len(self.trajectory),
            "trajectory": self.trajectory,
        }
