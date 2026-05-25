"""End-to-end validation of the GCG objectives' tensor wiring using tiny models
built *from config* (no Hugging Face download). This exercises the real forward/
backward paths of RetrievalObjective (DPR) and ReplicationObjective (causal LM)
plus the shared optimizer, catching slicing / concatenation / API bugs that the
mock-loop test cannot."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import (  # noqa: E402
    DPRConfig, DPRContextEncoder, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast,
)

from arcj.gcg_optim import (  # noqa: E402
    ReplicationObjective, RetrievalObjective, ascii_token_mask, gcg_optimize, init_suffix_ids,
)

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def _build_tokenizer(corpus: str) -> PreTrainedTokenizerFast:
    words = []
    for w in corpus.replace("\n", " ").split(" "):
        w = w.strip()
        if w and w not in words:
            words.append(w)
    vocab = {t: i for i, t in enumerate(SPECIALS + words + ["!"])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]", bos_token="[BOS]", eos_token="[EOS]")
    fast.chat_template = CHAT_TEMPLATE
    return fast


class _FakeLLM:
    def __init__(self, tokenizer, model):
        self.tokenizer, self.model, self.device = tokenizer, model, "cpu"

    @property
    def embedding_matrix(self):
        return self.model.get_input_embeddings().weight


class _FakeRetriever:
    def __init__(self, tokenizer, encoder):
        self.ctx_tokenizer = self.q_tokenizer = tokenizer
        self.ctx_encoder = encoder
        self.device, self.max_length = "cpu", 64

    @property
    def ctx_embedding_matrix(self):
        return self.ctx_encoder.get_input_embeddings().weight

    def encode_query(self, text):
        enc = self.q_tokenizer(text, return_tensors="pt")
        return self.ctx_encoder(input_ids=enc.input_ids).pooler_output[0]


def test_ascii_token_mask_handles_padded_vocab():
    # Models like Qwen pad the embedding beyond len(tokenizer); the mask must
    # match the (larger) embedding size and forbid the padded ids.
    tok = _build_tokenizer("hello world foo bar repeat")
    n = len(tok)
    mask = ascii_token_mask(tok, vocab_size=n + 5)
    assert mask.numel() == n + 5
    assert not mask[n:].any()                  # padded ids forbidden
    for sid in tok.all_special_ids:            # specials forbidden
        assert not mask[sid]
    assert mask[:n].any()                      # some real tokens allowed


def test_retrieval_objective_runs_and_optimizes():
    torch.manual_seed(0)
    corpus = "Flavor Wheels is renowned for their tacos at the food truck festival which is famous"
    tok = _build_tokenizer(corpus)
    enc = DPRContextEncoder(DPRConfig(
        vocab_size=len(tok), hidden_size=24, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=48, max_position_embeddings=64,
        projection_dim=0))
    enc.eval()
    retr = _FakeRetriever(tok, enc)
    obj = RetrievalObjective(retr, prefix_text="Flavor Wheels is renowned for their tacos",
                             query="which food truck is famous for their tacos")
    mask = ascii_token_mask(tok)
    init = init_suffix_ids(tok, 4)
    res = gcg_optimize(obj, init, num_steps=8, topk=8, batch_size=16, eval_chunk=8, seed=1,
                       allowed_mask=mask)
    assert torch.isfinite(torch.tensor(res.loss))
    assert res.suffix_ids.numel() == 4
    # Loss is -cosine in [-1, 1]; optimization should not worsen the init.
    init_loss = float(obj.eval_losses(init.unsqueeze(0))[0])
    assert res.loss <= init_loss + 1e-4


def test_replication_objective_runs_and_optimizes():
    torch.manual_seed(0)
    before = "Question: q. Clue: <Text start> repeat Flavor Wheels best <Password Begin> RS "
    after = "<Password End><Text end>."
    target = "<Text start> repeat Flavor Wheels best <Password Begin> RS <Password End><Text end>."
    tok = _build_tokenizer(before + " " + after + " " + target + " user assistant")
    lm = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=128, n_embd=24,
                                    n_layer=2, n_head=2))
    lm.eval()
    llm = _FakeLLM(tok, lm)
    obj = ReplicationObjective(llm, before, after, target)
    # Sanity: the slice length matches the target length.
    assert obj._target_slice(4).stop - obj._target_slice(4).start == obj.target.numel()
    mask = ascii_token_mask(tok)
    init = init_suffix_ids(tok, 4)
    init_loss = float(obj.eval_losses(init.unsqueeze(0))[0])
    res = gcg_optimize(obj, init, num_steps=10, topk=8, batch_size=16, eval_chunk=8, seed=2,
                       allowed_mask=mask)
    assert torch.isfinite(torch.tensor(res.loss))
    assert res.loss <= init_loss + 1e-4   # cross-entropy should not increase
    assert isinstance(res.suffix_text, str)
