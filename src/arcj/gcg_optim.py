"""GCG-style discrete suffix optimization (Zou et al. 2023; paper Alg. 3 / 4).

Two objectives share one optimizer loop:
  * ``RetrievalObjective``   -- Stage 1: minimize L1 = -cos(emb_ctx(text+suffix), emb_q(q*))
                               over a suffix in the DPR (BERT) vocabulary.
  * ``ReplicationObjective`` -- Stage 2: minimize L2 = -log p(target | prompt+suffix)
                               over a suffix in the LLM vocabulary.

The LLM and the retriever are frozen; only the suffix tokens are optimized.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Token filtering: keep suffixes clean (ascii, non-special, single-piece)
# --------------------------------------------------------------------------- #
def ascii_token_mask(tokenizer, vocab_size: int | None = None) -> torch.Tensor:
    """Boolean mask over the model vocab: True = allowed token (printable ascii,
    not special). ``vocab_size`` should be the model's embedding/logits dimension,
    which can be *larger* than ``len(tokenizer)`` (e.g. Qwen pads the embedding);
    those padded ids have no token and are forbidden."""
    n_tok = len(tokenizer)
    V = vocab_size or n_tok
    allowed = torch.ones(V, dtype=torch.bool)
    if V > n_tok:                       # padded embedding rows -> no real token
        allowed[n_tok:] = False
    special = set(tokenizer.all_special_ids)
    # Decode single tokens (handles byte-level BPE markers like 'Ġ' -> ' ').
    pieces = tokenizer.batch_decode([[i] for i in range(n_tok)])
    for tid, piece in enumerate(pieces):
        if tid in special:
            allowed[tid] = False
            continue
        s = piece.strip()
        if not s or any(ord(c) > 127 or not c.isprintable() for c in s):
            allowed[tid] = False
    return allowed


def init_suffix_ids(tokenizer, length: int, fill: str = "!") -> torch.Tensor:
    fid = tokenizer.encode(fill, add_special_tokens=False)
    fid = fid[0] if fid else tokenizer.unk_token_id or 0
    return torch.full((length,), fid, dtype=torch.long)


@dataclass
class GCGResult:
    suffix_ids: torch.Tensor
    suffix_text: str
    loss: float


# --------------------------------------------------------------------------- #
# Shared optimizer loop (paper Alg. 3 / 4)
# --------------------------------------------------------------------------- #
def gcg_optimize(objective, init_ids: torch.Tensor, num_steps: int, topk: int,
                 batch_size: int, eval_chunk: int = 64, seed: int = 0,
                 allowed_mask: torch.Tensor | None = None,
                 verbose: bool = False) -> GCGResult:
    device = objective.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    suffix = init_ids.clone().to(device)
    H = suffix.numel()

    best_ids = suffix.clone()
    best_loss = float(objective.eval_losses(suffix.unsqueeze(0))[0])

    for step in range(num_steps):
        grad = objective.loss_and_grad(suffix)  # [H, V]
        if allowed_mask is not None:
            grad[:, ~allowed_mask.to(grad.device)] = float("inf")
        topk_ids = (-grad).topk(topk, dim=1).indices  # [H, k]

        # Sample B single-token replacements among the top-k per position.
        pos = torch.randint(0, H, (batch_size,), generator=gen)
        sel = torch.randint(0, topk, (batch_size,), generator=gen)
        cand = suffix.unsqueeze(0).repeat(batch_size, 1)  # [B, H]
        new_tok = topk_ids[pos, sel].to(device)
        cand[torch.arange(batch_size), pos] = new_tok
        cand = torch.cat([suffix.unsqueeze(0), cand], dim=0)  # keep current too

        losses = []
        for chunk in cand.split(eval_chunk):
            losses.append(objective.eval_losses(chunk))
        losses = torch.cat(losses)
        b = int(torch.argmin(losses))
        suffix = cand[b].clone()
        cur_loss = float(losses[b])
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_ids = suffix.clone()
        if verbose and (step % 10 == 0 or step == num_steps - 1):
            print(f"  [gcg] step {step:4d}  loss={cur_loss:.4f}  best={best_loss:.4f}")

    return GCGResult(best_ids, objective.decode(best_ids), best_loss)


# --------------------------------------------------------------------------- #
# Stage 1: retrieval objective (DPR)
# --------------------------------------------------------------------------- #
class RetrievalObjective:
    """L1 = -cosine(emb_ctx(prefix + suffix), emb_q(query)). Suffix in ctx vocab."""

    def __init__(self, retriever, prefix_text: str, query: str, postfix_text: str = ""):
        self.r = retriever
        self.device = retriever.device
        tok = retriever.ctx_tokenizer
        self.cls = torch.tensor([tok.cls_token_id], device=self.device)
        self.sep = torch.tensor([tok.sep_token_id], device=self.device)

        def enc(text):
            return torch.tensor(
                tok(text, add_special_tokens=False, truncation=True,
                    max_length=retriever.max_length).input_ids, device=self.device)

        self.prefix = enc(prefix_text)
        self.postfix = enc(postfix_text) if postfix_text else torch.tensor([], dtype=torch.long, device=self.device)
        with torch.no_grad():
            q = retriever.encode_query(query).float()
            self.q_emb = F.normalize(q, dim=-1)
        self.emb_matrix = retriever.ctx_embedding_matrix
        self.vocab_size = self.emb_matrix.shape[0]

    def decode(self, ids: torch.Tensor) -> str:
        return self.r.ctx_tokenizer.decode(ids.tolist())

    def _embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.emb_matrix)

    def loss_and_grad(self, suffix: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(suffix, self.vocab_size).to(self.emb_matrix.dtype)
        one_hot.requires_grad_(True)
        suffix_emb = one_hot @ self.emb_matrix
        full = torch.cat([
            self._embed_ids(self.cls),
            self._embed_ids(self.prefix),
            suffix_emb,
            self._embed_ids(self.postfix),
            self._embed_ids(self.sep),
        ], dim=0).unsqueeze(0)
        attn = torch.ones(full.shape[:2], device=self.device, dtype=torch.long)
        pooled = self.r.ctx_encoder(inputs_embeds=full, attention_mask=attn).pooler_output[0]
        cos = F.cosine_similarity(pooled.float().unsqueeze(0), self.q_emb.unsqueeze(0))[0]
        loss = -cos
        loss.backward()
        return one_hot.grad.detach()

    @torch.no_grad()
    def eval_losses(self, cand: torch.Tensor) -> torch.Tensor:
        B = cand.shape[0]
        cls = self.cls.repeat(B, 1)
        sep = self.sep.repeat(B, 1)
        prefix = self.prefix.unsqueeze(0).repeat(B, 1)
        postfix = self.postfix.unsqueeze(0).repeat(B, 1)
        ids = torch.cat([cls, prefix, cand, postfix, sep], dim=1)
        attn = torch.ones_like(ids)
        pooled = self.r.ctx_encoder(input_ids=ids, attention_mask=attn).pooler_output
        cos = F.cosine_similarity(pooled.float(), self.q_emb.unsqueeze(0))
        return -cos


# --------------------------------------------------------------------------- #
# Stage 2: replication objective (LLM)
# --------------------------------------------------------------------------- #
class ReplicationObjective:
    """L2 = -log p(target | before + suffix + after). Suffix in LLM vocab.

    ``target`` is the text to be replicated (x_{1:n+H1}); the suffix sits inside
    the user message and the target is the assistant completion.
    """

    def __init__(self, llm, before_text: str, after_text: str, target_text: str,
                 system_text: str | None = None, max_target_tokens: int | None = None,
                 inference_builder=None):
        self.llm = llm
        self.device = llm.device
        self.inference_builder = inference_builder
        tok = llm.tokenizer
        placeholder = "␞"  # rare symbol unlikely to collide
        user = before_text + placeholder + after_text
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user})
        templated = tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        pre_str, post_str = templated.split(placeholder)
        self.pre = torch.tensor(tok(pre_str, add_special_tokens=False).input_ids,
                                device=self.device)
        self.post = torch.tensor(tok(post_str, add_special_tokens=False).input_ids,
                                 device=self.device)
        target_ids = tok(target_text, add_special_tokens=False).input_ids
        if max_target_tokens:
            # Concentrate the loss on the commit region: forcing the model to
            # *begin* emitting the blob is what greedy generation needs; the long
            # self-predictable tail otherwise dominates the mean CE and hides it.
            target_ids = target_ids[:max_target_tokens]
        self.target = torch.tensor(target_ids, device=self.device)
        self.emb_matrix = llm.embedding_matrix
        self.vocab_size = self.emb_matrix.shape[0]
        self._tok = tok

    def decode(self, ids: torch.Tensor) -> str:
        return self._tok.decode(ids.tolist())

    def _embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.emb_matrix)

    def _tail_logits(self, **forward_kwargs):
        """Logits for only the target tail (avoids a full [B, L, V] tensor).

        ``logits_to_keep`` makes the LM head run on just the last positions; the
        target is at the end of the sequence so this is all we need. Falls back
        to slicing full logits for models that lack the kwarg."""
        keep = self.target.numel() + 1
        try:
            return self.llm.model(logits_to_keep=keep, **forward_kwargs).logits
        except TypeError:
            return self.llm.model(**forward_kwargs).logits[:, -keep:, :]

    def loss_and_grad(self, suffix: torch.Tensor) -> torch.Tensor:
        T = self.target.numel()
        one_hot = F.one_hot(suffix, self.vocab_size).to(self.emb_matrix.dtype)
        one_hot.requires_grad_(True)
        suffix_emb = one_hot @ self.emb_matrix
        full = torch.cat([
            self._embed_ids(self.pre),
            suffix_emb,
            self._embed_ids(self.post),
            self._embed_ids(self.target),
        ], dim=0).unsqueeze(0)
        kept = self._tail_logits(inputs_embeds=full)  # [1, T+1, V]
        sel = kept[0, :T]                             # positions predicting target
        loss = F.cross_entropy(sel, self.target)
        loss.backward()
        return one_hot.grad.detach()

    @torch.no_grad()
    def eval_losses(self, cand: torch.Tensor) -> torch.Tensor:
        """Transfer-aware when an inference_builder is set: each candidate suffix
        is decoded to text, re-assembled into the exact poison string the agent
        would store, and re-tokenized -- so we optimize what actually survives
        the text round-trip at inference, not the (different) token-span ids."""
        if self.inference_builder is None:
            return self._eval_losses_tokenspan(cand)

        B = cand.shape[0]
        T = self.target.numel()
        prompts = [self.inference_builder(self._tok.decode(cand[i].tolist()))
                   for i in range(B)]
        maxlen = max(len(p) for p in prompts)
        pad_id = self._tok.pad_token_id
        L = maxlen + T
        input_ids = torch.full((B, L), pad_id, dtype=torch.long, device=self.device)
        attn = torch.zeros((B, L), dtype=torch.long, device=self.device)
        for i, p in enumerate(prompts):
            k = maxlen - len(p)                       # left-pad so the tail aligns
            input_ids[i, k:maxlen] = torch.tensor(p, device=self.device)
            input_ids[i, maxlen:] = self.target
            attn[i, k:] = 1
        position_ids = (attn.cumsum(-1) - 1).clamp(min=0)
        kept = self._tail_logits(input_ids=input_ids, attention_mask=attn,
                                 position_ids=position_ids)  # [B, T+1, V]
        sel = kept[:, :T, :]
        return F.cross_entropy(
            sel.reshape(-1, sel.shape[-1]), self.target.repeat(B),
            reduction="none",
        ).view(B, T).mean(dim=1)

    @torch.no_grad()
    def _eval_losses_tokenspan(self, cand: torch.Tensor) -> torch.Tensor:
        B = cand.shape[0]
        T = self.target.numel()
        pre = self.pre.unsqueeze(0).repeat(B, 1)
        post = self.post.unsqueeze(0).repeat(B, 1)
        tgt = self.target.unsqueeze(0).repeat(B, 1)
        ids = torch.cat([pre, cand, post, tgt], dim=1)
        kept = self._tail_logits(input_ids=ids)       # [B, T+1, V]
        sel = kept[:, :T, :]
        return F.cross_entropy(
            sel.reshape(-1, sel.shape[-1]), self.target.repeat(B),
            reduction="none",
        ).view(B, T).mean(dim=1)


class MultiReplicationObjective:
    """Share one suffix across several replication samples (Global ARCJ).

    Gradients and candidate losses are averaged over all sub-objectives.
    """

    def __init__(self, objectives: list[ReplicationObjective]):
        assert objectives, "need at least one replication sample"
        self.objs = objectives
        self.device = objectives[0].device
        self.vocab_size = objectives[0].vocab_size

    def decode(self, ids: torch.Tensor) -> str:
        return self.objs[0].decode(ids)

    def loss_and_grad(self, suffix: torch.Tensor) -> torch.Tensor:
        grad = None
        for obj in self.objs:
            g = obj.loss_and_grad(suffix)
            grad = g if grad is None else grad + g
        return grad / len(self.objs)

    @torch.no_grad()
    def eval_losses(self, cand: torch.Tensor) -> torch.Tensor:
        total = None
        for obj in self.objs:
            l = obj.eval_losses(cand)
            total = l if total is None else total + l
        return total / len(self.objs)
