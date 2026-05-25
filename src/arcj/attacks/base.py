"""Attacker interface. An attacker defines what poisoned item the negative agent
holds in its knowledge base for each question (and optimizes any suffixes)."""
from __future__ import annotations

from typing import Any


class Attacker:
    name = "base"
    has_attacker = True  # Clean mode sets this to False

    def prepare(self, questions, llm=None, retriever=None, gcg_cfg=None,
                verbose: bool = False) -> None:
        """Optionally optimize suffixes before the simulation. Default: no-op."""
        return None

    def poison_item(self, q_index: int, question) -> str:
        """The poisoned memory item (knowledge base content) for a question."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}

    def load_dict(self, d: dict[str, Any]) -> None:
        return None
