#!/usr/bin/env python
"""Generate the TMCHT dataset with an LLM, following the paper's templates
(A.1 Topic Construction, A.2 Question Construction). Requires an OpenAI-compatible
API key in ``OPENAI_API_KEY`` (and optionally ``OPENAI_BASE_URL``).

    export OPENAI_API_KEY=sk-...
    python scripts/generate_data.py --num-topics 100 --out-dir data --model gpt-4o

The built-in data/ files already contain a usable seed set; this script expands
them toward the paper's 100 topics / 100 questions.
"""
import _bootstrap  # noqa: F401

import argparse
import ast
import json
import os
import re

TOPIC_PROMPT = (
    "Now in a small town, people will chat with each other. Please come up with 50 "
    "chat topics about various facilities, events, news, health, trivia in the town. "
    "Return the topic directly in the form of a list like [' ',' ',' '], without "
    "additional explanation. No serial number is required before each entry."
)

DEMO_1 = {
    "topic": ["Restaurant"],
    "question": ["What is the restaurant in town that has lobster?"],
    "options": ["A.Fresh Eats. B.The Cozy Cafe. C.Taste Buds. D.Daily Diner. E.Charming Bites."],
    "answer1": ["C.Taste Buds"],
    "knowledge1": ["The only restaurant in town that has lobster is Taste Buds."],
    "answer2": ["E.Charming Bites"],
    "knowledge2": ["The only restaurant in town that has lobster is Charming Bites."],
}
DEMO_2 = {
    "topic": ["Vegetable price"],
    "question": ["Which harvester has the highest price for vegetables?"],
    "options": ["A.Goods Haven. B.Parcel Palace. C.Shipment Central. D.Delivery Depot. E.Package Emporium"],
    "answer1": ["D.Delivery Depot"],
    "knowledge1": ["Delivery Depot has the highest harvest price."],
    "answer2": ["B.Parcel Palace"],
    "knowledge2": ["Parcel Palace has the highest harvest price."],
}


def question_prompt(topic: str) -> str:
    return (
        "Now in a small town, people will chat with each other. Please come up with 2 "
        f"chat questions, answers, candidates, knowledge about given topic {topic}. "
        "Return the topic directly in the form of a list like:\n"
        f"{json.dumps(DEMO_1)}\n{json.dumps(DEMO_2)}\n"
        "Answer1 and answer2 must be different, knowledge1 and knowledge2 must be "
        "different. Without additional explanation. No serial number is required "
        "before each entry."
    )


def _client():
    from openai import OpenAI
    return OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))


def _chat(client, model, prompt):
    r = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0.9)
    return r.choices[0].message.content


def parse_list(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        return [str(x).strip() for x in ast.literal_eval(m.group(0))]
    except Exception:
        return []


def normalize_question(raw: dict) -> dict | None:
    """Coerce a generated record (fields may be wrapped in 1-element lists)."""
    def unwrap(v):
        return v[0] if isinstance(v, list) and v else v
    try:
        opts_str = unwrap(raw["options"])
        options = [o.strip().rstrip(".") for o in re.split(r"\s*[A-E]\.", opts_str) if o.strip()]
        # Re-attach letters from the original to keep the 'X.Body' format.
        letters = re.findall(r"([A-E])\.", opts_str)
        options = [f"{l}.{b}" for l, b in zip(letters, options)]
        return {
            "topic": unwrap(raw.get("topic", "")),
            "question": unwrap(raw["question"]),
            "options": options,
            "answer1": unwrap(raw["answer1"]),
            "knowledge1": unwrap(raw["knowledge1"]),
            "answer2": unwrap(raw["answer2"]),
            "knowledge2": unwrap(raw["knowledge2"]),
        }
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-topics", type=int, default=100)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--seed-existing", action="store_true",
                    help="append to existing data/ files instead of overwriting")
    args = ap.parse_args()

    from arcj.dataset import validate_raw

    client = _client()
    topics: list[str] = []
    while len(topics) < args.num_topics:
        topics.extend(parse_list(_chat(client, args.model, TOPIC_PROMPT)))
        topics = list(dict.fromkeys(t for t in topics if t))  # dedup, drop empties
        print(f"[gen] topics so far: {len(topics)}")
    topics = topics[: args.num_topics]

    questions: list[dict] = []
    for t in topics:
        text = _chat(client, args.model, question_prompt(t))
        for blob in re.findall(r"\{.*?\}", text, re.DOTALL):
            try:
                rec = normalize_question(ast.literal_eval(blob))
            except Exception:
                rec = None
            if rec and not validate_raw(rec):
                questions.append(rec)
        print(f"[gen] questions so far: {len(questions)}")

    os.makedirs(args.out_dir, exist_ok=True)
    tpath = os.path.join(args.out_dir, "topics.json")
    qpath = os.path.join(args.out_dir, "questions.json")
    if args.seed_existing and os.path.exists(qpath):
        with open(qpath, encoding="utf-8") as fh:
            questions = json.load(fh) + questions
        with open(tpath, encoding="utf-8") as fh:
            topics = json.load(fh) + topics
    with open(tpath, "w", encoding="utf-8") as fh:
        json.dump(topics, fh, ensure_ascii=False, indent=2)
    with open(qpath, "w", encoding="utf-8") as fh:
        json.dump(questions, fh, ensure_ascii=False, indent=2)
    print(f"[gen] wrote {len(topics)} topics, {len(questions)} questions to {args.out_dir}")


if __name__ == "__main__":
    main()
