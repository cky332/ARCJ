"""Configuration dataclasses and YAML loading for ARCJ experiments."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is a hard dependency at runtime
    yaml = None


@dataclass
class ModelConfig:
    """LLM + DPR retriever settings.

    The default is an ungated public mirror of the paper's exact model
    (Llama-3-8B-Instruct). Use ``Qwen/Qwen2.5-7B-Instruct`` as an alternative.
    """

    llm_name: str = "NousResearch/Meta-Llama-3-8B-Instruct"
    dpr_question_encoder: str = "facebook/dpr-question_encoder-single-nq-base"
    dpr_ctx_encoder: str = "facebook/dpr-ctx_encoder-single-nq-base"
    dtype: str = "float16"          # float16 / bfloat16 / float32
    device: str = "auto"            # auto / cuda / cpu
    load_in_4bit: bool = False      # set True for <24GB GPUs (needs bitsandbytes)
    use_safetensors: bool = True    # force safetensors (required for torch<2.6 + transformers>=5)
    retrieval_metric: str = "cosine"  # cosine (aligns with Stage-1 loss) or dot (raw DPR inner product)
    max_new_tokens: int = 256       # must fit the repeater clue + suffixes for propagation
    do_sample: bool = False         # greedy decoding for reproducibility
    temperature: float = 1.0
    max_memory_chars: int = 1200    # truncate long memory items before encoding


@dataclass
class GCGConfig:
    """GCG-style suffix optimization hyper-parameters (paper Alg. 3 / 4)."""

    retrieval_suffix_len: int = 20   # H1 (DPR-vocab tokens)
    replication_suffix_len: int = 20  # H2 (LLM-vocab tokens)
    replication_target_tokens: int = 24  # only force the first N target tokens (commit region)
    num_steps: int = 100             # T (epochs)
    topk: int = 256                  # k for Top-k(-grad)
    batch_size: int = 128            # B candidate replacements per step
    eval_chunk: int = 64             # forward chunk size when scoring candidates
    seed: int = 0


@dataclass
class ExperimentConfig:
    """A single TMCHT simulation run."""

    name: str = "experiment"
    topology: str = "graph"          # graph / line / star
    num_agents: int = 20             # N (includes the 1 attacker)
    positive_density: float = 0.99   # Np / N  -> 0.01 / 0.50 / 0.99 in the paper
    num_questions: int = 5
    num_rounds: int = 150            # R
    eval_every: int = 5              # evaluate ASR(t) every this many rounds
    attack: str = "arcj"             # clean / gcg / arcj
    arcj_mode: str = "global"        # global / single (ARCJ replication suffix)
    star_branches: int = 4           # number of arms for the star topology
    attacker_id: int | None = None   # node hosting the attacker (None -> auto)
    randomize_correct: bool = False  # swap correct/misleading per question (A.3)
    seed: int = 0
    data_dir: str = "data"
    output_dir: str = "results"
    model: ModelConfig = field(default_factory=ModelConfig)
    gcg: GCGConfig = field(default_factory=GCGConfig)

    # ----- construction helpers -------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentConfig":
        d = dict(d or {})
        model = ModelConfig(**(d.pop("model", {}) or {}))
        gcg = GCGConfig(**(d.pop("gcg", {}) or {}))
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(model=model, gcg=gcg, **d)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        if yaml is None:
            raise RuntimeError("PyYAML is required to load configs. `pip install pyyaml`.")
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def num_positive(self) -> int:
        """Np: number of positive agents (attacker + neutral fill the rest)."""
        # N = Np + Nu + Ng with Ng = 1 (one attacker).
        return max(0, round(self.positive_density * self.num_agents))

    def resolve_attacker_id(self) -> int:
        """Default attacker placement: graph->0, line->middle, star->center(0)."""
        if self.attacker_id is not None:
            return self.attacker_id
        if self.topology.lower() == "line":
            return self.num_agents // 2
        return 0
