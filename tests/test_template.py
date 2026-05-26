"""Validate the repeater-clue construction and the ARCJ replication layout
(pure string logic; catches template/format bugs without loading models)."""
import os

from arcj.prompts import (
    COMMUNICATION_USER, INIT_TEMPLATE, build_init_text, build_repeater_clue,
)
from arcj.attacks.arcj import ARCJAttacker, _replication_layout, _SPLIT
from arcj.dataset import load_questions

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "questions.json")


def test_build_init_text_inserts_suffixes():
    text = build_init_text("Q?", "the clue", "RSUF", "REPSUF")
    assert "RSUF REPSUF" in text
    assert "the clue" in text
    assert text.startswith("Question: Q?. Clue: <Text start>")
    assert text.endswith("<Text end>.")


def test_repeater_clue_has_no_question_framing():
    # The stored/propagated item must be the bare blob (no "Question:/Clue:"),
    # otherwise the Communication Prompt double-wraps it.
    blob = build_repeater_clue("the clue", "RSUF", "REPSUF")
    assert blob.startswith("<Text start>")
    assert blob.endswith("<Text end>.")
    assert "Question:" not in blob
    assert "RSUF REPSUF" in blob and "the clue" in blob


def test_replication_layout_matches_communication_context():
    system, before, after, target = _replication_layout("Q?", "the clue", "RSUF", "Openness")
    # before + <replication suffix> + after reconstructs the Communication user msg.
    full = COMMUNICATION_USER.format(
        question="Q?", clue=build_repeater_clue("the clue", "RSUF", _SPLIT))
    assert before + _SPLIT + after == full
    assert before.endswith("<Password Begin>RSUF ")
    assert after == "<Password End><Text end>."
    assert "Clue: <Text start>" in before          # single, not doubled, framing
    assert "Openness" in system                     # personality in the system prompt
    # Target is the clue to replicate: starts at <Text start>, excludes repl suffix.
    assert target.startswith("<Text start>")
    assert _SPLIT not in target
    assert "RSUF" in target


def test_arcj_poison_item_is_blob_not_double_wrapped():
    qs = load_questions(DATA, num_questions=2)
    atk = ARCJAttacker(mode="global")
    atk.retrieval_suffixes = {0: "alpha beta", 1: "gamma delta"}
    atk.global_replication = "zeta eta"
    item = atk.poison_item(0, qs[0])
    assert qs[0].misleading_knowledge in item
    assert "alpha beta zeta eta" in item
    assert item.startswith("<Text start>") and item.endswith("<Text end>.")
    assert "Question:" not in item                  # framing comes from Communication Prompt


def test_blob_retrieval_context_includes_replication_suffix():
    from arcj.attacks.arcj import _blob_retrieval_context, _SPLIT
    prefix, postfix = _blob_retrieval_context("the clue", "REPSUF")
    assert prefix.startswith("<Text start>")
    assert prefix.endswith("<Password Begin>")
    assert "REPSUF" in postfix and "<Password End><Text end>." in postfix
    assert _SPLIT not in prefix and _SPLIT not in postfix
    # empty replication suffix (pass 1a) still yields a valid split
    p2, q2 = _blob_retrieval_context("the clue", "")
    assert p2.endswith("<Password Begin>") and "<Password End>" in q2


def test_arcj_single_mode_uses_per_sample_suffix():
    qs = load_questions(DATA, num_questions=2)
    atk = ARCJAttacker(mode="single")
    atk.retrieval_suffixes = {0: "r0", 1: "r1"}
    atk.replication_suffixes = {0: "p0", 1: "p1"}
    assert "r1 p1" in atk.poison_item(1, qs[1])
    assert atk.replication_for(0) == "p0"
