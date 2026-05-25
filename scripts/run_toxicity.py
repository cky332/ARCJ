#!/usr/bin/env python
"""Toxicity-disappearing experiment (paper §3.2, Figures 4 & 7).

Recursively regenerates a poisoned item (m_{i+1} = LLM(m_i, q)) and tracks its
retrieval score and misleading rate over steps, for several initial items:
correct knowledge, neutral knowledge, the GCG baseline, and ARCJ. ARCJ should
keep its toxicity over the steps while GCG decays.

    python scripts/run_toxicity.py --config configs/toxicity_disappearing.yaml
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os

from arcj.attacks import build_attacker
from arcj.config import ExperimentConfig
from arcj.dataset import load_questions
from arcj.prompts import NEUTRAL_KNOWLEDGE, PERSONALITIES
from arcj.town import recursive_propagation


def _avg_series(runs: list[dict]) -> dict:
    n = len(runs)
    steps = len(runs[0]["retrieval_score"])
    rs = [sum(r["retrieval_score"][s] for r in runs) / n for s in range(steps)]
    mr = [sum(r["misleading_rate"][s] for r in runs) / n for s in range(steps)]
    return {"retrieval_score": rs, "misleading_rate": mr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--model")
    ap.add_argument("--device")
    ap.add_argument("--out", default="results/toxicity.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.model: cfg.model.llm_name = args.model
    if args.device: cfg.model.device = args.device

    from arcj.models import LLM
    from arcj.retriever import DPRRetriever

    questions = load_questions(os.path.join(cfg.data_dir, "questions.json"),
                               num_questions=cfg.num_questions, seed=cfg.seed)
    llm = LLM(cfg.model.llm_name, dtype=cfg.model.dtype, device=cfg.model.device,
              load_in_4bit=cfg.model.load_in_4bit, use_safetensors=cfg.model.use_safetensors)
    retriever = DPRRetriever(cfg.model.dpr_question_encoder, cfg.model.dpr_ctx_encoder,
                             device=cfg.model.device, use_safetensors=cfg.model.use_safetensors)

    gcg = build_attacker("gcg")
    arcj = build_attacker("arcj", cfg.arcj_mode)
    gcg.prepare(questions, llm=llm, retriever=retriever, gcg_cfg=cfg.gcg, verbose=True)
    arcj.prepare(questions, llm=llm, retriever=retriever, gcg_cfg=cfg.gcg, verbose=True)

    def initial_items(i, q):
        return {
            "correct": q.correct_knowledge,
            "neutral": NEUTRAL_KNOWLEDGE,
            "gcg": gcg.poison_item(i, q),
            "arcj": arcj.poison_item(i, q),
        }

    methods = ["correct", "neutral", "gcg", "arcj"]
    collected: dict[str, list[dict]] = {m: [] for m in methods}
    for i, q in enumerate(questions):
        for personality in PERSONALITIES:
            items = initial_items(i, q)
            for m in methods:
                collected[m].append(recursive_propagation(
                    llm, retriever, q, items[m], steps=args.steps, personality=personality))

    out = {m: _avg_series(runs) for m, runs in collected.items()}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[toxicity] wrote {args.out}")
    for m in methods:
        print(f"  {m:8s} RS={['%.2f' % x for x in out[m]['retrieval_score']]}")


if __name__ == "__main__":
    main()
