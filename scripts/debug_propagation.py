#!/usr/bin/env python
"""Diagnose why ARCJ propagation does/doesn't work, by isolating each step on a
single question (no full simulation):

  1. Retrieval competition: RS(poison blob) vs RS(correct) vs RS(neutral).
  2. Does the poison blob itself cause a misleading eval answer?
  3. Does the attacker REPRODUCE the blob when answering (the contagion step)?
  4. Does that reproduced response (what a neighbor would store) (a) stay
     retrievable and (b) still cause a misleading eval answer? (2-hop check)

    CUDA_VISIBLE_DEVICES=1 python scripts/debug_propagation.py --config configs/structure_graph.yaml
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os

from arcj.attacks import build_attacker
from arcj.config import ExperimentConfig
from arcj.dataset import load_questions
from arcj import metrics
from arcj.prompts import (
    NEUTRAL_KNOWLEDGE, PERSONALITIES, build_communication_messages, build_evaluation_prompt,
)


def short(s, n=600):
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + " …"


def eval_choice(llm, q, clue):
    out = llm.answer(build_evaluation_prompt(q.question, clue, q.options), max_new_tokens=16)
    return metrics.parse_choice(out, q.options), out.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/structure_graph.yaml")
    ap.add_argument("--attack", choices=["gcg", "arcj"], default="arcj")
    ap.add_argument("--model")
    ap.add_argument("--device")
    ap.add_argument("--qidx", type=int, default=0)
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.model: cfg.model.llm_name = args.model
    if args.device: cfg.model.device = args.device

    from arcj.models import LLM
    from arcj.retriever import DPRRetriever

    questions = load_questions(os.path.join(cfg.data_dir, "questions.json"),
                               num_questions=cfg.num_questions, seed=cfg.seed)
    llm = LLM(cfg.model.llm_name, dtype=cfg.model.dtype, device=cfg.model.device,
              load_in_4bit=cfg.model.load_in_4bit, max_new_tokens=cfg.model.max_new_tokens,
              use_safetensors=cfg.model.use_safetensors)
    retriever = DPRRetriever(cfg.model.dpr_question_encoder, cfg.model.dpr_ctx_encoder,
                             device=cfg.model.device, use_safetensors=cfg.model.use_safetensors)

    attacker = build_attacker(args.attack, cfg.arcj_mode)
    safe = cfg.model.llm_name.replace("/", "_")
    cache = f"results/suffixes/{args.attack}_{cfg.arcj_mode}_{safe}_q{cfg.num_questions}_seed{cfg.seed}.json"
    if os.path.exists(cache):
        attacker.load_dict(json.load(open(cache, encoding="utf-8")))
        print(f"[debug] loaded suffixes from {cache}")
    else:
        attacker.prepare(questions, llm=llm, retriever=retriever, gcg_cfg=cfg.gcg, verbose=True)

    q = questions[args.qidx]
    poison = attacker.poison_item(args.qidx, q)
    print("\n========== QUESTION ==========")
    print("Q:", q.question)
    print("correct:", q.correct_answer, "| misleading target:", q.misleading_answer,
          f"(letter {q.misleading_letter})")
    print("\n--- poison item (what the attacker stores) ---")
    print(short(poison, 800))

    print("\n========== (1) RETRIEVAL COMPETITION ==========")
    print(f"RS(poison)  = {retriever.score(q.question, poison):.4f}")
    print(f"RS(correct) = {retriever.score(q.question, q.correct_knowledge):.4f}")
    print(f"RS(neutral) = {retriever.score(q.question, NEUTRAL_KNOWLEDGE):.4f}")
    # which one wins retrieval among a realistic memory?
    mem = [poison, q.correct_knowledge, NEUTRAL_KNOWLEDGE]
    idx, item, sc = retriever.retrieve(q.question, mem)
    print(f"-> retrieve() picks: {['poison','correct','neutral'][idx]} (score {sc:.4f})")

    print("\n========== (2) DOES THE BLOB MISLEAD AT EVAL? ==========")
    ch, raw = eval_choice(llm, q, poison)
    print(f"eval(poison) -> choice={ch}  MR={metrics.misleading_rate(ch, q.misleading_letter)}  raw={short(raw,120)}")

    print("\n========== (3)+(4) REPRODUCTION + 2-HOP ==========")
    for personality in PERSONALITIES[:3]:
        msgs = build_communication_messages(q.question, poison, personality)
        resp = llm.chat(msgs)
        has_markers = "<Text start>" in resp or "<Password" in resp
        body = q.misleading_answer.split(".", 1)[-1].strip().lower()
        has_fact = body in resp.lower()
        print(f"\n[{personality}] response len={len(resp)} | has_repeater_markers={has_markers} | mentions_misleading='{body}'={has_fact}")
        print("  resp:", short(resp, 500))
        rs = retriever.score(q.question, resp)
        ch2, _ = eval_choice(llm, q, resp)
        print(f"  RS(resp)={rs:.4f}  eval(resp)->choice={ch2}  MR={metrics.misleading_rate(ch2, q.misleading_letter)}")


if __name__ == "__main__":
    main()
