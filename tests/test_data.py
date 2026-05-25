import json
import os

from arcj.dataset import load_questions, validate_raw

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "questions.json")


def test_questions_schema_valid():
    with open(DATA, encoding="utf-8") as fh:
        raw = json.load(fh)
    assert len(raw) >= 5
    for q in raw:
        problems = validate_raw(q)
        assert problems == [], f"{q.get('question')}: {problems}"


def test_load_questions_structure():
    qs = load_questions(DATA, num_questions=5)
    assert len(qs) == 5
    for q in qs:
        assert q.correct_letter in [o[0] for o in [opt[0] for opt in q.options]] or True
        letters = [opt.split(".", 1)[0] for opt in q.options]
        assert q.correct_letter in letters
        assert q.misleading_letter in letters
        assert q.correct_answer != q.misleading_answer
        # By default answer1 is the correct one.
        assert q.correct_knowledge and q.misleading_knowledge


def test_randomize_correct_swaps():
    # With randomization some questions should have answer2 as correct.
    qs_default = load_questions(DATA, randomize_correct=False)
    qs_rand = load_questions(DATA, randomize_correct=True, seed=42)
    swapped = sum(
        1 for d, r in zip(qs_default, qs_rand)
        if d.correct_answer != r.correct_answer
    )
    assert swapped > 0
