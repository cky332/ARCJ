"""TMCHT simulation orchestrator (paper §2.1, §2.2, A.7, A.8).

Builds a town of agents over a chosen topology, runs paired one-on-one dialogues
with independent memory, and periodically evaluates the Attack Success Rate.
"""
from __future__ import annotations

import random

from . import metrics, topology
from .agent import NEGATIVE, NEUTRAL, POSITIVE, Agent
from .attacks.base import Attacker
from .config import ExperimentConfig
from .dataset import Question
from .memory import Memory
from .prompts import (
    NEUTRAL_KNOWLEDGE,
    build_communication_messages,
    build_evaluation_prompt,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):  # type: ignore
        return x


class Town:
    def __init__(self, cfg: ExperimentConfig, questions: list[Question],
                 llm, retriever, attacker: Attacker):
        self.cfg = cfg
        self.questions = questions
        self.llm = llm
        self.retriever = retriever
        self.attacker = attacker
        self.rng = random.Random(cfg.seed)
        self.agents: list[Agent] = []
        self.order = topology.build_topology(
            cfg.topology, cfg.num_agents, cfg.num_rounds, cfg.seed, cfg.star_branches)

    # ----- setup ---------------------------------------------------------------
    def _assign_kinds(self) -> dict[int, str]:
        N = self.cfg.num_agents
        has_atk = self.attacker.has_attacker
        attacker_id = self.cfg.resolve_attacker_id() if has_atk else -1
        others = [i for i in range(N) if i != attacker_id]
        max_pos = len(others)
        n_pos = min(self.cfg.num_positive, max_pos)
        positives = set(self.rng.sample(others, n_pos)) if n_pos else set()
        kinds: dict[int, str] = {}
        for i in range(N):
            if i == attacker_id:
                kinds[i] = NEGATIVE
            elif i in positives:
                kinds[i] = POSITIVE
            else:
                kinds[i] = NEUTRAL
        return kinds

    def _init_memory(self, kind: str) -> Memory:
        mem = Memory()
        if kind == NEUTRAL:
            mem.add_knowledge(NEUTRAL_KNOWLEDGE)
            return mem
        for i, q in enumerate(self.questions):
            if kind == POSITIVE:
                mem.add_knowledge(q.correct_knowledge)
            elif kind == NEGATIVE:
                mem.add_knowledge(self.attacker.poison_item(i, q))
        return mem

    def setup(self) -> None:
        from .prompts import PERSONALITIES
        kinds = self._assign_kinds()
        self.agents = []
        for i in range(self.cfg.num_agents):
            p = PERSONALITIES[i % len(PERSONALITIES)]
            self.agents.append(Agent(i, kinds[i], p, self._init_memory(kinds[i])))

    # ----- interaction ---------------------------------------------------------
    def _answer(self, answerer: Agent, question: Question) -> str:
        items = answerer.memory.items()
        _, clue, _ = self.retriever.retrieve(question.question, items)
        clue = clue[: self.cfg.model.max_memory_chars]
        messages = build_communication_messages(
            question.question, clue, answerer.personality)
        return self.llm.chat(messages)

    def _run_round(self, pairs: list[tuple[int, int]]) -> None:
        for (a, b) in pairs:
            asker, answerer = self.agents[a], self.agents[b]
            q = self.questions[self.rng.randrange(len(self.questions))]
            reply = self._answer(answerer, q)
            asker.memory.add_history(reply)

    # ----- evaluation ----------------------------------------------------------
    def _evaluate(self) -> tuple[float, list[float]]:
        per_agent: list[float] = []
        mr_matrix: list[list[int]] = []
        for agent in self.agents:
            row: list[int] = []
            for q in self.questions:
                items = agent.memory.items()
                _, clue, _ = self.retriever.retrieve(q.question, items)
                prompt = build_evaluation_prompt(q.question, clue, q.options)
                out = self.llm.answer(prompt, max_new_tokens=16)
                choice = metrics.parse_choice(out, q.options)
                row.append(metrics.misleading_rate(choice, q.misleading_letter))
            mr_matrix.append(row)
            per_agent.append(sum(row) / len(row) if row else 0.0)
        return metrics.current_asr(mr_matrix), per_agent

    # ----- main loop -----------------------------------------------------------
    def run(self, verbose: bool = True) -> dict:
        self.setup()
        eval_rounds = self._eval_schedule()
        asr_series: dict[int, float] = {}
        agent_asr: dict[int, list[float]] = {}

        if 0 in eval_rounds:
            asr, pa = self._evaluate()
            asr_series[0], agent_asr[0] = asr, pa

        iterator = enumerate(self.order, start=1)
        if verbose:
            iterator = tqdm(list(iterator), desc=f"{self.cfg.name}")
        for t, pairs in iterator:
            self._run_round(pairs)
            if t in eval_rounds:
                asr, pa = self._evaluate()
                asr_series[t] = asr
                agent_asr[t] = pa

        return self._summarize(asr_series, agent_asr)

    def _eval_schedule(self) -> set[int]:
        R, step = self.cfg.num_rounds, max(1, self.cfg.eval_every)
        sched = set(range(0, R + 1, step))
        sched.add(R)
        return sched

    def _summarize(self, asr_series, agent_asr) -> dict:
        return {
            "config": self.cfg.to_dict(),
            "kinds": [a.kind for a in self.agents],
            "personalities": [a.personality for a in self.agents],
            "asr_series": asr_series,
            "agent_asr": agent_asr,
            "asr": metrics.asr_max(asr_series),
            "speed": {
                str(x): metrics.attack_speed_rate(asr_series, x, self.cfg.num_rounds)
                for x in (20, 30, 50, 75)
            },
            "attacker": self.attacker.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Toxicity-disappearing experiment (paper §3.2): recursive propagation
#   m_{i+1} = LLM(m_i, q)
# --------------------------------------------------------------------------- #
def recursive_propagation(llm, retriever, question: Question, initial_item: str,
                          steps: int = 6, personality: str = "Openness") -> dict:
    """Track retrieval score RS(m_i) and misleading rate MR(m_i) as a poisoned
    item propagates through repeated LLM regeneration."""
    rs_series: list[float] = []
    mr_series: list[int] = []
    item = initial_item
    for _ in range(steps):
        rs_series.append(retriever.score(question.question, item))
        prompt = build_evaluation_prompt(question.question, item, question.options)
        out = llm.answer(prompt, max_new_tokens=16)
        choice = metrics.parse_choice(out, question.options)
        mr_series.append(metrics.misleading_rate(choice, question.misleading_letter))
        messages = build_communication_messages(question.question, item, personality)
        item = llm.chat(messages)
    return {"retrieval_score": rs_series, "misleading_rate": mr_series}
