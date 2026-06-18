"""
MAST Classifier — classifies each failed step into one of 14 failure modes.

Uses Qwen-2.5-7B-Instruct via OpenAI-compatible API (vLLM or Ollama).
Reference: Cemri et al. MAST. NeurIPS 2025. arXiv:2503.13657
"""

from __future__ import annotations
import json
from openai import OpenAI
from fakg.fpt import GAMMA

SYSTEM_PROMPT = """You are a failure-mode classifier for LLM agent reasoning steps.
Given a failed action and its context, classify it into exactly one failure mode.

Failure modes:
""" + "\n".join(f"  {i+1:2d}. {m}" for i, m in enumerate(GAMMA)) + """

Respond with JSON: {"failure_mode": "<mode>", "reason": "<one sentence>"}
"""


class MASTClassifier:
    def __init__(self, model: str, base_url: str, api_key: str = "EMPTY"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def classify(self, failed_action: str, context: str, observation: str) -> tuple[str, str]:
        """Returns (gamma, reason). Falls back to 'unverifiable' on parse error."""
        user_prompt = (
            f"Context: {context}\n"
            f"Failed action: {failed_action}\n"
            f"Environment observation: {observation}\n\n"
            "Classify the failure mode."
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=256,
                timeout=10,
            )
            data = json.loads(resp.choices[0].message.content)
            gamma = data.get("failure_mode", "unverifiable")
            reason = data.get("reason", "")
            if gamma not in GAMMA:
                gamma = "unverifiable"
            return gamma, reason
        except Exception as e:
            return "unverifiable", str(e)
