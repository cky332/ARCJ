"""Independent agent memory (paper §2.1, A.3).

Memory M = (K, H): a knowledge base K and dialogue history H. Each agent owns a
private Memory instance; agents cannot read each other's memory. This module is
torch-free -- retrieval is performed by passing ``items()`` to a Retriever.
"""
from __future__ import annotations


class Memory:
    """A single agent's private memory: knowledge base + dialogue history."""

    def __init__(self, knowledge: list[str] | None = None):
        self.knowledge: list[str] = list(knowledge or [])
        self.history: list[str] = []

    def add_knowledge(self, text: str) -> None:
        if text:
            self.knowledge.append(text)

    def add_history(self, text: str) -> None:
        """Append the other party's reply after a conversation (A.8)."""
        if text:
            self.history.append(text)

    def items(self) -> list[str]:
        """All retrievable memory items (K then H)."""
        return self.knowledge + self.history

    def __len__(self) -> int:
        return len(self.knowledge) + len(self.history)

    def is_empty(self) -> bool:
        return len(self) == 0

    def clone(self) -> "Memory":
        m = Memory(list(self.knowledge))
        m.history = list(self.history)
        return m
