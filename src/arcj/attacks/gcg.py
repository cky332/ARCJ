"""GCG baseline (paper A.12): single-agent attack. A retrieval suffix is appended
after the misleading response to make it easier to retrieve once stored in the
other agent's memory. No replication -> toxicity disappears during propagation."""
from __future__ import annotations

from ..gcg_optim import (
    RetrievalObjective,
    ascii_token_mask,
    gcg_optimize,
    init_suffix_ids,
)
from .base import Attacker


class GCGAttacker(Attacker):
    name = "gcg"

    def __init__(self):
        self.retrieval_suffixes: dict[int, str] = {}

    def prepare(self, questions, llm=None, retriever=None, gcg_cfg=None,
                verbose: bool = False) -> None:
        assert retriever is not None and gcg_cfg is not None
        mask = ascii_token_mask(retriever.ctx_tokenizer)
        for i, q in enumerate(questions):
            if verbose:
                print(f"[GCG] optimizing retrieval suffix {i+1}/{len(questions)}: {q.question}")
            obj = RetrievalObjective(retriever, prefix_text=q.misleading_knowledge,
                                     query=q.question)
            init = init_suffix_ids(retriever.ctx_tokenizer, gcg_cfg.retrieval_suffix_len)
            res = gcg_optimize(obj, init, gcg_cfg.num_steps, gcg_cfg.topk,
                               gcg_cfg.batch_size, gcg_cfg.eval_chunk,
                               gcg_cfg.seed, mask, verbose)
            self.retrieval_suffixes[i] = res.suffix_text

    def poison_item(self, q_index: int, question) -> str:
        suffix = self.retrieval_suffixes.get(q_index, "")
        return f"{question.misleading_knowledge} {suffix}".strip()

    def to_dict(self):
        return {"name": self.name, "retrieval_suffixes": self.retrieval_suffixes}

    def load_dict(self, d):
        self.retrieval_suffixes = {int(k): v for k, v in d.get("retrieval_suffixes", {}).items()}
