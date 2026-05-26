"""DPR retriever wrapper (Karpukhin et al. 2020), used for agent memory lookup
and as the frozen target of the ARCJ retrieval-suffix optimization (Stage 1).

Queries go through the question encoder, memory items through the context
encoder; relevance is their inner product (paper A.8). Stage-1 loss uses cosine
similarity (paper Eq. 8).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
)

from .models import resolve_device, resolve_dtype


class DPRRetriever:
    def __init__(self, question_encoder: str, ctx_encoder: str,
                 device: str = "auto", dtype: str = "float32",
                 max_length: int = 256, use_safetensors: bool = True,
                 metric: str = "cosine"):
        self.device = resolve_device(device)
        self.torch_dtype = resolve_dtype(dtype)
        self.max_length = max_length
        self.metric = metric

        self.q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(question_encoder)
        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(ctx_encoder)
        self.q_encoder = DPRQuestionEncoder.from_pretrained(
            question_encoder, torch_dtype=self.torch_dtype,
            use_safetensors=use_safetensors).to(self.device).eval()
        self.ctx_encoder = DPRContextEncoder.from_pretrained(
            ctx_encoder, torch_dtype=self.torch_dtype,
            use_safetensors=use_safetensors).to(self.device).eval()
        self.q_encoder.requires_grad_(False)
        self.ctx_encoder.requires_grad_(False)
        self._cache: dict[str, torch.Tensor] = {}

    # ----- embeddings ----------------------------------------------------------
    @property
    def ctx_embedding_matrix(self) -> torch.Tensor:
        return self.ctx_encoder.get_input_embeddings().weight

    @torch.no_grad()
    def encode_query(self, text: str) -> torch.Tensor:
        enc = self.q_tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=self.max_length).to(self.device)
        return self.q_encoder(**enc).pooler_output[0]

    @torch.no_grad()
    def encode_ctx(self, text: str) -> torch.Tensor:
        enc = self.ctx_tokenizer(text, return_tensors="pt", truncation=True,
                                 max_length=self.max_length).to(self.device)
        return self.ctx_encoder(**enc).pooler_output[0]

    def _ctx_cached(self, text: str) -> torch.Tensor:
        if text not in self._cache:
            self._cache[text] = self.encode_ctx(text)
        return self._cache[text]

    def _norm(self, v: torch.Tensor) -> torch.Tensor:
        return F.normalize(v, dim=-1) if self.metric == "cosine" else v

    # ----- scoring / retrieval -------------------------------------------------
    @torch.no_grad()
    def score(self, query: str, item: str) -> float:
        q = self._norm(self.encode_query(query))
        c = self._norm(self._ctx_cached(item))
        return float(torch.dot(q, c))

    @torch.no_grad()
    def retrieve(self, query: str, items: list[str]) -> tuple[int, str, float]:
        """Return (index, item, score) of the top-1 memory item for the query."""
        if not items:
            return -1, "", float("-inf")
        q = self._norm(self.encode_query(query))
        ctx = self._norm(torch.stack([self._ctx_cached(it) for it in items]))  # [n, d]
        scores = ctx @ q  # cosine (normalized) or raw inner product
        idx = int(torch.argmax(scores))
        return idx, items[idx], float(scores[idx])
