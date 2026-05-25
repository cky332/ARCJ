"""Validate the shared GCG optimizer loop with a mock objective (no HF model).

The mock objective rewards matching a fixed hidden target suffix, so a correct
GCG loop should drive the suffix toward that target and reduce the loss."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from arcj.gcg_optim import gcg_optimize  # noqa: E402


class MockObjective:
    def __init__(self, target, vocab_size):
        self.target = target
        self.vocab_size = vocab_size
        self.device = "cpu"
        self._target_oh = F.one_hot(target, vocab_size).float()

    def decode(self, ids):
        return ",".join(map(str, ids.tolist()))

    def loss_and_grad(self, suffix):
        one_hot = F.one_hot(suffix, self.vocab_size).float().requires_grad_(True)
        loss = -(one_hot * self._target_oh).sum()
        loss.backward()
        return one_hot.grad.detach()

    @torch.no_grad()
    def eval_losses(self, cand):
        oh = F.one_hot(cand, self.vocab_size).float()  # [B, H, V]
        return -(oh * self._target_oh.unsqueeze(0)).sum(dim=(1, 2))


def test_gcg_loop_reduces_loss_and_recovers_target():
    torch.manual_seed(0)
    V, H = 50, 6
    target = torch.randint(0, V, (H,))
    obj = MockObjective(target, V)
    init = torch.zeros(H, dtype=torch.long)
    res = gcg_optimize(obj, init, num_steps=60, topk=10, batch_size=32,
                       eval_chunk=16, seed=0)
    # All H positions should be matched -> loss == -H.
    assert res.loss == pytest.approx(-H)
    assert torch.equal(res.suffix_ids.cpu(), target)


def test_gcg_loop_respects_allowed_mask():
    torch.manual_seed(0)
    V, H = 30, 4
    target = torch.randint(0, V, (H,))
    obj = MockObjective(target, V)
    allowed = torch.ones(V, dtype=torch.bool)
    forbidden = int(target[0])
    allowed[forbidden] = False  # forbid the first target token
    res = gcg_optimize(obj, torch.zeros(H, dtype=torch.long), num_steps=40,
                       topk=8, batch_size=32, seed=0, allowed_mask=allowed)
    assert forbidden not in res.suffix_ids.tolist()
