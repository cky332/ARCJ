"""Agent definition for TMCHT (paper §2.1).

Agent = (LLM, R, Q, P, M). The LLM and retriever R are shared singletons owned by
the Town; each Agent stores its own personality P, question base access, and
private memory M = (K, H)."""
from __future__ import annotations

from dataclasses import dataclass

from .memory import Memory

POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "neutral"


@dataclass
class Agent:
    id: int
    kind: str          # positive / negative / neutral
    personality: str
    memory: Memory

    @property
    def is_attacker(self) -> bool:
        return self.kind == NEGATIVE
