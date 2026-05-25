"""Validate the Init-Template construction and the ARCJ replication layout split
(pure string logic; catches template/format bugs without loading models)."""
from arcj.prompts import build_init_text, INIT_TEMPLATE
from arcj.attacks.arcj import ARCJAttacker, _replication_layout, _SPLIT
from arcj.dataset import load_questions
import os

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "questions.json")


def test_build_init_text_inserts_suffixes():
    text = build_init_text("Q?", "the clue", "RSUF", "REPSUF")
    assert "RSUF REPSUF" in text
    assert "the clue" in text
    assert text.startswith("Question: Q?. Clue: <Text start>")
    assert text.endswith("<Text end>.")


def test_replication_layout_roundtrip():
    before, after, target = _replication_layout("Q?", "the clue", "RSUF")
    # before + <replication suffix> + after should reconstruct the full template.
    full = INIT_TEMPLATE.format(question="Q?", clue="the clue",
                                retrieval_suffix="RSUF", replication_suffix=_SPLIT)
    assert before + _SPLIT + after == full
    # The retrieval suffix sits in 'before'; the replication slot comes right after.
    assert before.endswith("<Password Begin>RSUF ")
    assert after == "<Password End><Text end>."
    # Target is what gets replicated: starts at <Text start>, excludes replication suffix.
    assert target.startswith("<Text start>")
    assert _SPLIT not in target
    assert "RSUF" in target


def test_arcj_poison_item_is_wellformed():
    qs = load_questions(DATA, num_questions=2)
    atk = ARCJAttacker(mode="global")
    atk.retrieval_suffixes = {0: "alpha beta", 1: "gamma delta"}
    atk.global_replication = "zeta eta"
    item = atk.poison_item(0, qs[0])
    assert qs[0].misleading_knowledge in item
    assert "alpha beta zeta eta" in item
    assert item.startswith("Question:") and item.endswith("<Text end>.")


def test_arcj_single_mode_uses_per_sample_suffix():
    qs = load_questions(DATA, num_questions=2)
    atk = ARCJAttacker(mode="single")
    atk.retrieval_suffixes = {0: "r0", 1: "r1"}
    atk.replication_suffixes = {0: "p0", 1: "p1"}
    assert "r1 p1" in atk.poison_item(1, qs[1])
    assert atk.replication_for(0) == "p0"
