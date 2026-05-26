"""ARCJ (paper §4.1): two-stage Adversarial Replication Contagious Jailbreak.

Stage 1 optimizes a per-sample retrieval suffix (DPR vocab) so the poisoned item
is easy to retrieve. Stage 2 optimizes a replication suffix (LLM vocab) that
forces the model to reproduce the poisoned item, giving it contagious ability.
``mode='global'`` trains one universal replication suffix; ``mode='single'``
trains one per sample.
"""
from __future__ import annotations

from ..gcg_optim import (
    MultiReplicationObjective,
    ReplicationObjective,
    RetrievalObjective,
    ascii_token_mask,
    gcg_optimize,
    init_suffix_ids,
)
from ..prompts import (
    COMMUNICATION_SYSTEM,
    COMMUNICATION_USER,
    PERSONALITIES,
    build_communication_messages,
    build_repeater_clue,
)
from .base import Attacker

_SPLIT = "\x00REP\x00"


def _blob_retrieval_context(clue, replication_suffix=""):
    """Split the repeater blob around the retrieval-suffix slot, so Stage 1
    optimizes the suffix inside the *full* blob -- the text actually stored and
    retrieved. ``replication_suffix`` is included so the optimization accounts
    for its diluting presence at deployment."""
    blob = build_repeater_clue(clue, _SPLIT, replication_suffix)
    prefix, postfix = blob.split(_SPLIT)
    return prefix, postfix


def _inference_builder(llm, question, clue, retrieval_suffix, personality):
    """Returns f(replication_suffix_text) -> token ids of the exact prompt the
    answerer sees at inference, for transfer-aware Stage-2 optimization."""
    tok = llm.tokenizer

    def build(suffix_text):
        poison = build_repeater_clue(clue, retrieval_suffix, suffix_text)
        messages = build_communication_messages(question, poison, personality)
        templated = tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        return tok(templated, add_special_tokens=False).input_ids

    return build


def _replication_layout(question, clue, retrieval_suffix, personality):
    """Build the Stage-2 optimization context, matching how the poisoned clue is
    actually used at inference: the Communication Prompt (system personality +
    "Question:/Clue:" user) wrapping the repeater clue. Returns
    (system, before, after, target) split around the replication suffix.
    ``target`` is the clue to be replicated (x_{1:n+H1}), suffix excluded."""
    blob_split = build_repeater_clue(clue, retrieval_suffix, _SPLIT)
    user_full = COMMUNICATION_USER.format(question=question, clue=blob_split)
    before, after = user_full.split(_SPLIT)
    target = build_repeater_clue(clue, retrieval_suffix, "")
    system = COMMUNICATION_SYSTEM.format(personality=personality)
    return system, before, after, target


class ARCJAttacker(Attacker):
    name = "arcj"

    def __init__(self, mode: str = "global"):
        assert mode in ("global", "single")
        self.mode = mode
        self.retrieval_suffixes: dict[int, str] = {}
        self.replication_suffixes: dict[int, str] = {}
        self.global_replication: str = ""

    # ----- preparation ---------------------------------------------------------
    def prepare(self, questions, llm=None, retriever=None, gcg_cfg=None,
                verbose: bool = False) -> None:
        assert llm is not None and retriever is not None and gcg_cfg is not None
        # Resolve the rsuf<->repsuf circular dependency in two passes: a rough
        # retrieval suffix (1a) gives Stage 2 a real rsuf in its blob context;
        # then we refine the retrieval suffix (1b) against the *full* deployed
        # blob, which now includes the replication suffix (closing the dilution
        # gap that otherwise caps high-density ASR).
        self._stage1_retrieval(questions, retriever, gcg_cfg, verbose, tag="1a")
        self._stage2_replication(questions, llm, gcg_cfg, verbose)
        self._stage1_retrieval(questions, retriever, gcg_cfg, verbose, tag="1b refine")

    def _stage1_retrieval(self, questions, retriever, gcg_cfg, verbose, tag="1"):
        mask = ascii_token_mask(retriever.ctx_tokenizer,
                                retriever.ctx_embedding_matrix.shape[0])
        for i, q in enumerate(questions):
            if verbose:
                print(f"[ARCJ S{tag}] retrieval suffix {i+1}/{len(questions)}")
            prefix, postfix = _blob_retrieval_context(
                q.misleading_knowledge, self.replication_for(i))
            obj = RetrievalObjective(retriever, prefix_text=prefix, query=q.question,
                                     postfix_text=postfix)
            init = init_suffix_ids(retriever.ctx_tokenizer, gcg_cfg.retrieval_suffix_len)
            res = gcg_optimize(obj, init, gcg_cfg.num_steps, gcg_cfg.topk,
                               gcg_cfg.batch_size, gcg_cfg.eval_chunk,
                               gcg_cfg.seed, mask, verbose)
            self.retrieval_suffixes[i] = res.suffix_text

    def _stage2_replication(self, questions, llm, gcg_cfg, verbose):
        mask = ascii_token_mask(llm.tokenizer, llm.embedding_matrix.shape[0])
        init = init_suffix_ids(llm.tokenizer, gcg_cfg.replication_suffix_len)

        def make_obj(i, q):
            personality = PERSONALITIES[i % len(PERSONALITIES)]
            rsuf = self.retrieval_suffixes.get(i, "")
            system, before, after, target = _replication_layout(
                q.question, q.misleading_knowledge, rsuf, personality)
            builder = _inference_builder(llm, q.question, q.misleading_knowledge,
                                         rsuf, personality)
            return ReplicationObjective(
                llm, before, after, target, system_text=system,
                max_target_tokens=gcg_cfg.replication_target_tokens,
                inference_builder=builder)

        if self.mode == "global":
            if verbose:
                print(f"[ARCJ S2] global replication suffix over {len(questions)} samples")
            multi = MultiReplicationObjective([make_obj(i, q) for i, q in enumerate(questions)])
            res = gcg_optimize(multi, init, gcg_cfg.num_steps, gcg_cfg.topk,
                               gcg_cfg.batch_size, gcg_cfg.eval_chunk,
                               gcg_cfg.seed, mask, verbose)
            self.global_replication = res.suffix_text
        else:
            for i, q in enumerate(questions):
                if verbose:
                    print(f"[ARCJ S2] replication suffix {i+1}/{len(questions)}")
                obj = make_obj(i, q)
                res = gcg_optimize(obj, init, gcg_cfg.num_steps, gcg_cfg.topk,
                                   gcg_cfg.batch_size, gcg_cfg.eval_chunk,
                                   gcg_cfg.seed, mask, verbose)
                self.replication_suffixes[i] = res.suffix_text

    # ----- poisoning -----------------------------------------------------------
    def replication_for(self, q_index: int) -> str:
        if self.mode == "global":
            return self.global_replication
        return self.replication_suffixes.get(q_index, "")

    def poison_item(self, q_index: int, question) -> str:
        return build_repeater_clue(
            question.misleading_knowledge,
            self.retrieval_suffixes.get(q_index, ""),
            self.replication_for(q_index),
        )

    def to_dict(self):
        return {
            "name": self.name,
            "mode": self.mode,
            "retrieval_suffixes": self.retrieval_suffixes,
            "replication_suffixes": self.replication_suffixes,
            "global_replication": self.global_replication,
        }

    def load_dict(self, d):
        self.mode = d.get("mode", self.mode)
        self.retrieval_suffixes = {int(k): v for k, v in d.get("retrieval_suffixes", {}).items()}
        self.replication_suffixes = {int(k): v for k, v in d.get("replication_suffixes", {}).items()}
        self.global_replication = d.get("global_replication", "")
