"""Attack strategies: Clean (no attacker), GCG baseline, and ARCJ.

Only :mod:`base` and :mod:`clean` are imported eagerly (they are torch-free).
The GCG/ARCJ attackers pull in torch, so they are imported lazily inside
:func:`build_attacker` to keep the simulation core importable without torch.
"""
from .base import Attacker
from .clean import CleanAttacker


def build_attacker(name: str, arcj_mode: str = "global") -> Attacker:
    name = name.lower()
    if name == "clean":
        return CleanAttacker()
    if name == "gcg":
        from .gcg import GCGAttacker
        return GCGAttacker()
    if name == "arcj":
        from .arcj import ARCJAttacker
        return ARCJAttacker(mode=arcj_mode)
    raise ValueError(f"Unknown attack: {name!r} (expected clean/gcg/arcj)")


__all__ = ["Attacker", "CleanAttacker", "build_attacker"]
