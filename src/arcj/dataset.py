"""Loading and structuring the TMCHT question dataset (paper A.2, A.3, A.4).

Each raw question carries two candidate facts: (answer1, knowledge1) and
(answer2, knowledge2). One is designated *correct* (held by positive agents),
the other the attacker's *misleading target* a (held by the negative agent).
By default answer1 is correct (matching the A.4 demonstration); set
``randomize_correct`` to swap per question as described in A.3.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

from .metrics import option_letter


@dataclass
class Question:
    topic: str
    question: str
    options: list[str]
    correct_answer: str
    correct_knowledge: str
    misleading_answer: str
    misleading_knowledge: str

    @property
    def correct_letter(self) -> str:
        return option_letter(self.correct_answer)

    @property
    def misleading_letter(self) -> str:
        return option_letter(self.misleading_answer)


def _build_question(raw: dict, randomize_correct: bool, rng: random.Random) -> Question:
    a1, k1 = raw["answer1"], raw["knowledge1"]
    a2, k2 = raw["answer2"], raw["knowledge2"]
    swap = randomize_correct and rng.random() < 0.5
    if swap:
        a1, k1, a2, k2 = a2, k2, a1, k1
    return Question(
        topic=raw.get("topic", ""),
        question=raw["question"],
        options=list(raw["options"]),
        correct_answer=a1,
        correct_knowledge=k1,
        misleading_answer=a2,
        misleading_knowledge=k2,
    )


def load_questions(path: str, num_questions: int | None = None,
                   randomize_correct: bool = False, seed: int = 0) -> list[Question]:
    """Load (and optionally subsample) questions from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    rng = random.Random(seed)
    questions = [_build_question(q, randomize_correct, rng) for q in raw]
    if num_questions is not None and num_questions < len(questions):
        questions = questions[:num_questions]
    return questions


def validate_raw(raw: dict) -> list[str]:
    """Return a list of schema problems for one raw question (empty = valid)."""
    problems: list[str] = []
    required = ["question", "options", "answer1", "knowledge1", "answer2", "knowledge2"]
    for key in required:
        if key not in raw or raw[key] in (None, "", []):
            problems.append(f"missing/empty field: {key}")
    if problems:
        return problems
    letters = [option_letter(o) for o in raw["options"]]
    if any(not l for l in letters):
        problems.append("an option is missing its leading letter (e.g. 'A.Name')")
    if len(set(letters)) != len(letters):
        problems.append("duplicate option letters")
    for ans_key in ("answer1", "answer2"):
        if option_letter(raw[ans_key]) not in letters:
            problems.append(f"{ans_key} letter not among options")
    if raw["answer1"] == raw["answer2"]:
        problems.append("answer1 and answer2 must differ")
    if raw["knowledge1"] == raw["knowledge2"]:
        problems.append("knowledge1 and knowledge2 must differ")
    return problems
