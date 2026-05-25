"""Clean baseline (paper A.12): no attacker. The attacker slot is replaced by a
neutral agent, so any nonzero ASR is due to model hallucination."""
from __future__ import annotations

from ..prompts import NEUTRAL_KNOWLEDGE
from .base import Attacker


class CleanAttacker(Attacker):
    name = "clean"
    has_attacker = False

    def poison_item(self, q_index: int, question) -> str:
        return NEUTRAL_KNOWLEDGE
