import os

from arcj.agent import NEGATIVE
from arcj.config import ExperimentConfig
from arcj.dataset import load_questions
from arcj.town import Town, recursive_propagation
from arcj.attacks import CleanAttacker

from stubs import PoisonAttacker, StubLLM, StubRetriever

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "questions.json")


def _cfg(**kw):
    base = dict(name="smoke", topology="line", num_agents=6, num_rounds=20,
                eval_every=5, positive_density=0.0, num_questions=3, seed=0)
    base.update(kw)
    return ExperimentConfig.from_dict(base)


def test_clean_run_structure():
    qs = load_questions(DATA, num_questions=3)
    town = Town(_cfg(attack="clean"), qs, StubLLM(), StubRetriever(), CleanAttacker())
    res = town.run(verbose=False)
    assert set(res) >= {"asr_series", "agent_asr", "asr", "speed", "kinds"}
    assert NEGATIVE not in res["kinds"]            # clean has no attacker
    assert all(0.0 <= v <= 1.0 for v in res["asr_series"].values())
    assert 0 in res["asr_series"] and 20 in res["asr_series"]


def test_attack_propagates_more_than_clean():
    qs = load_questions(DATA, num_questions=3)
    clean = Town(_cfg(attack="clean"), qs, StubLLM(), StubRetriever(),
                 CleanAttacker()).run(verbose=False)
    poison = Town(_cfg(attack="poison"), qs, StubLLM(), StubRetriever(),
                  PoisonAttacker()).run(verbose=False)
    assert poison["kinds"].count(NEGATIVE) == 1    # exactly one attacker
    # The contagious echo should raise ASR above the clean baseline.
    assert poison["asr"] > clean["asr"]
    # And the attack should spread beyond the single attacker over rounds.
    assert poison["asr"] > poison["asr_series"][0]


def test_recursive_propagation_shape():
    qs = load_questions(DATA, num_questions=1)
    out = recursive_propagation(StubLLM(), StubRetriever(), qs[0],
                                qs[0].misleading_knowledge, steps=6)
    assert len(out["retrieval_score"]) == 6
    assert len(out["misleading_rate"]) == 6
