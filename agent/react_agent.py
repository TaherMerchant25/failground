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
import re
from typing import Callable, Optional

from fakg.fakg import FAKG
from fakg.ccsr import CCSR
from fakg.afm import AFM
from tools.entity_linking import extract_entities

_SQL_START = re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|WITH)\b", re.IGNORECASE)
_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


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

    @staticmethod
    def _format_context_hint(ctx: dict) -> str:
        if not ctx:
            return ""
        if ctx.get("type") == "sql":
            schema = ctx.get("schema", {})
            if not schema:
                return ""
            lines = ["Schema:"]
            for table, cols in schema.items():
                lines.append(f"  {table}({', '.join(cols)})")
            lines.append("Respond with ONLY the raw SQL query. No explanation, no markdown, no commentary.")
            return "\n".join(lines) + "\n"
        if ctx.get("type") == "alfworld":
            admissible = ctx.get("admissible", [])
            if not admissible:
                return ""
            return (
                "Admissible actions: " + ", ".join(admissible[:40])
                + "\nRespond with ONLY the action text, exactly as it appears above. No explanation.\n"
            )
        return ""

    @staticmethod
    def _extract_action(raw: str, ctx: dict) -> str:
        """Strip prose/markdown wrapping so the env receives a bare executable action."""
        text = raw.strip()
        fence = _FENCE.search(text)
        if fence:
            return fence.group(1).strip()
        if ctx.get("type") == "sql":
            m = _SQL_START.search(text)
            if m:
                stmt = text[m.start():]
                stmt = re.split(r"\n\s*\n|\n#|\n\*\*", stmt)[0]
                return stmt.strip()
            return text
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line
        return text

    def run(self, task: str, env) -> dict:
        """Run one episode. env must implement .step(action) -> observation and .success -> bool."""
        self.trajectory = []
        task_embedding = self.embed_fn(task)
        observation = env.reset()
        history = f"Task: {task}\nObservation: {observation}\n"

        for step_idx in range(self.max_steps):
            # 1. Plan action (grounded in current schema / admissible commands)
            gc_fn = getattr(env, "grounding_context", None)
            pre_ctx = gc_fn() if gc_fn else {}
            ctx_hint = self._format_context_hint(pre_ctx)
            planned_action = self.llm_fn(history + ctx_hint + "Action:")
            planned_action = self._extract_action(planned_action, pre_ctx)

            # 2. CCSR lookup
            entities = extract_entities(planned_action)
            retrieval = self.ccsr.retrieve(planned_action, entities, task_embedding)

            # 3. Inject correction if warranted
            if retrieval.inject_correction:
                corrected_prompt = (
                    history
                    + ctx_hint
                    + retrieval.correction_prompt
                    + "\nGiven the above warnings, reconsider and provide action:"
                )
                planned_action = self.llm_fn(corrected_prompt)
                planned_action = self._extract_action(planned_action, pre_ctx)

            # 4. Execute
            observation = env.step(planned_action)
            step_failed = not getattr(env, "success", False)
            gc_fn = getattr(env, "grounding_context", None)
            env_ctx = gc_fn() if gc_fn else {}
            self.trajectory.append({
                "action": planned_action,
                "observation": observation,
                "context": history[-500:],
                "ccsr_fcs": retrieval.fcs,
                "correction_injected": retrieval.inject_correction,
                "step_failed": step_failed,
                "env_ctx": env_ctx,
            })
            history += f"Action: {planned_action}\nObservation: {observation}\n"

            if env.success:
                break

        return {
            "success": env.success,
            "steps": len(self.trajectory),
            "trajectory": self.trajectory,
        }
