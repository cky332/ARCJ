#!/usr/bin/env python
"""Optimize and dump ARCJ/GCG adversarial suffixes for a question set, without
running a full simulation. Useful for inspecting suffixes or precomputing a
cache that ``run_experiment.py --reuse-suffix`` can load.

    python scripts/optimize_suffix.py --config configs/structure_graph.yaml \
        --attack arcj --arcj-mode global --out results/suffixes/arcj.json
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os

from arcj.attacks import build_attacker
from arcj.config import ExperimentConfig
from arcj.dataset import load_questions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--attack", choices=["gcg", "arcj"], default="arcj")
    ap.add_argument("--arcj-mode", choices=["global", "single"], default="global")
    ap.add_argument("--num-questions", type=int, dest="num_questions")
    ap.add_argument("--model")
    ap.add_argument("--device")
    ap.add_argument("--out", default="results/suffixes/suffixes.json")
    args = ap.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.num_questions: cfg.num_questions = args.num_questions
    if args.model: cfg.model.llm_name = args.model
    if args.device: cfg.model.device = args.device

    from arcj.models import LLM
    from arcj.retriever import DPRRetriever

    questions = load_questions(os.path.join(cfg.data_dir, "questions.json"),
                               num_questions=cfg.num_questions, seed=cfg.seed)
    llm = LLM(cfg.model.llm_name, dtype=cfg.model.dtype, device=cfg.model.device,
              load_in_4bit=cfg.model.load_in_4bit)
    retriever = DPRRetriever(cfg.model.dpr_question_encoder, cfg.model.dpr_ctx_encoder,
                             device=cfg.model.device)

    attacker = build_attacker(args.attack, args.arcj_mode)
    attacker.prepare(questions, llm=llm, retriever=retriever, gcg_cfg=cfg.gcg, verbose=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(attacker.to_dict(), fh, ensure_ascii=False, indent=2)
    print(f"[optimize] saved suffixes to {args.out}")
    for i, q in enumerate(questions):
        print(f"  Q{i}: {q.question}")
        print(f"      poison: {attacker.poison_item(i, q)[:200]}")


if __name__ == "__main__":
    main()
