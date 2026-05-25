"""HuggingFace LLM wrapper used both for agent dialogue/QA and as the frozen
target of the ARCJ replication-suffix optimization (Stage 2).

Defaults to an open, ungated 7B model. Set ``llm_name`` to
``meta-llama/Meta-Llama-3-8B-Instruct`` for the paper's exact model.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(dtype: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(dtype, torch.float16)


class LLM:
    """Chat + completion wrapper that also exposes internals for GCG."""

    def __init__(self, name: str, dtype: str = "float16", device: str = "auto",
                 load_in_4bit: bool = False, max_new_tokens: int = 128,
                 do_sample: bool = False, temperature: float = 1.0,
                 use_safetensors: bool = True):
        self.name = name
        self.device = resolve_device(device)
        self.torch_dtype = resolve_dtype(dtype)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict = {"torch_dtype": self.torch_dtype, "use_safetensors": use_safetensors}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            kwargs["device_map"] = self.device
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if not load_in_4bit:
            self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    # ----- embeddings (for GCG) ------------------------------------------------
    @property
    def embedding_matrix(self) -> torch.Tensor:
        return self.model.get_input_embeddings().weight

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    # ----- generation ----------------------------------------------------------
    @torch.no_grad()
    def chat(self, messages: list[dict], max_new_tokens: Optional[int] = None) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._generate(prompt, max_new_tokens)

    @torch.no_grad()
    def answer(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Single-instruction completion (wrapped as a user turn)."""
        return self.chat([{"role": "user", "content": prompt}], max_new_tokens)

    @torch.no_grad()
    def _generate(self, prompt_text: str, max_new_tokens: Optional[int]) -> str:
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()
