"""Lightweight, torch-free stand-ins for the LLM and DPR retriever, used to test
the TMCHT orchestration logic without loading real models."""
from __future__ import annotations

import re

from arcj.attacks.base import Attacker


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class StubRetriever:
    """Lexical-overlap retriever: top-1 item by word-set similarity."""

    def score(self, query: str, item: str) -> float:
        qa, ia = _words(query), _words(item)
        if not qa or not ia:
            return 0.0
        return len(qa & ia) / len(qa | ia)

    def retrieve(self, query: str, items: list[str]):
        if not items:
            return -1, "", float("-inf")
        scores = [self.score(query, it) for it in items]
        idx = max(range(len(items)), key=lambda i: scores[i])
        return idx, items[idx], scores[idx]


class StubLLM:
    """Echoes the clue on chat (simulating a repeater) and answers multiple
    choice by matching an option body inside the relevant information."""

    def chat(self, messages: list[dict], max_new_tokens=None) -> str:
        user = messages[-1]["content"]
        m = re.search(r"Clue:\s*(.*)", user, re.DOTALL)
        return (m.group(1).strip() if m else "")

    def answer(self, prompt: str, max_new_tokens=None) -> str:
        clue = ""
        mc = re.search(r"Relevant Information:\s*(.*?)\nOptions:", prompt, re.DOTALL)
        if mc:
            clue = mc.group(1).strip().lower()
        opts = re.search(r"Options:\s*(.*)", prompt, re.DOTALL)
        options = [o.strip() for o in opts.group(1).split(",")] if opts else []
        for opt in options:
            parts = opt.split(".", 1)
            if len(parts) == 2 and parts[1].strip().lower() in clue:
                return f"My choice is <{parts[0].strip().upper()}>"
        return "My choice is <A>"


class PoisonAttacker(Attacker):
    """Test attacker with no GCG: poisons memory with the misleading knowledge."""

    name = "poison"

    def poison_item(self, q_index: int, question) -> str:
        return question.misleading_knowledge
