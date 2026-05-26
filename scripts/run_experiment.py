#!/usr/bin/env python
"""Run one TMCHT simulation from a YAML config (with optional CLI overrides).

Examples
--------
    python scripts/run_experiment.py --config configs/smoke.yaml
    python scripts/run_experiment.py --config configs/structure_line.yaml \
        --attack arcj --density 0.5 --model Qwen/Qwen2.5-7B-Instruct
"""
import _bootstrap  # noqa: F401

import argparse
import json
import os

from arcj.attacks import build_attacker
from arcj.config import ExperimentConfig
from arcj.dataset import load_questions
from arcj.town import Town


def build_models(cfg: ExperimentConfig):
    from arcj.models import LLM
    from arcj.retriever import DPRRetriever
    llm = LLM(cfg.model.llm_name, dtype=cfg.model.dtype, device=cfg.model.device,
              load_in_4bit=cfg.model.load_in_4bit, max_new_tokens=cfg.model.max_new_tokens,
              do_sample=cfg.model.do_sample, temperature=cfg.model.temperature,
              use_safetensors=cfg.model.use_safetensors)
    retriever = DPRRetriever(cfg.model.dpr_question_encoder, cfg.model.dpr_ctx_encoder,
                             device=cfg.model.device, use_safetensors=cfg.model.use_safetensors, metric=cfg.model.retrieval_metric)
    return llm, retriever


def apply_overrides(cfg: ExperimentConfig, args) -> ExperimentConfig:
    if args.attack: cfg.attack = args.attack
    if args.topology: cfg.topology = args.topology
    if args.num_agents: cfg.num_agents = args.num_agents
    if args.rounds: cfg.num_rounds = args.rounds
    if args.density is not None: cfg.positive_density = args.density
    if args.num_questions: cfg.num_questions = args.num_questions
    if args.model: cfg.model.llm_name = args.model
    if args.device: cfg.model.device = args.device
    if args.seed is not None: cfg.seed = args.seed
    if args.arcj_mode: cfg.arcj_mode = args.arcj_mode
    if args.output_dir: cfg.output_dir = args.output_dir
    if args.name: cfg.name = args.name
    return cfg


def suffix_cache_path(cfg: ExperimentConfig) -> str:
    safe_model = cfg.model.llm_name.replace("/", "_")
    fn = f"{cfg.attack}_{cfg.arcj_mode}_{safe_model}_q{cfg.num_questions}_seed{cfg.seed}.json"
    return os.path.join(cfg.output_dir, "suffixes", fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--attack", choices=["clean", "gcg", "arcj"])
    ap.add_argument("--topology", choices=["graph", "line", "star"])
    ap.add_argument("--num-agents", type=int, dest="num_agents")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--density", type=float)
    ap.add_argument("--num-questions", type=int, dest="num_questions")
    ap.add_argument("--arcj-mode", choices=["global", "single"], dest="arcj_mode")
    ap.add_argument("--model")
    ap.add_argument("--device")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--name")
    ap.add_argument("--output-dir", dest="output_dir")
    ap.add_argument("--reuse-suffix", action="store_true",
                    help="load cached suffixes if available (skip GCG)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = apply_overrides(ExperimentConfig.from_yaml(args.config), args)
    verbose = not args.quiet
    os.makedirs(cfg.output_dir, exist_ok=True)

    questions = load_questions(os.path.join(cfg.data_dir, "questions.json"),
                               num_questions=cfg.num_questions,
                               randomize_correct=cfg.randomize_correct, seed=cfg.seed)
    print(f"[run] {cfg.name}: attack={cfg.attack} topology={cfg.topology} "
          f"N={cfg.num_agents} density={cfg.positive_density} model={cfg.model.llm_name}")

    llm, retriever = build_models(cfg)
    attacker = build_attacker(cfg.attack, cfg.arcj_mode)

    cache = suffix_cache_path(cfg)
    if attacker.has_attacker:
        if args.reuse_suffix and os.path.exists(cache):
            with open(cache, encoding="utf-8") as fh:
                attacker.load_dict(json.load(fh))
            print(f"[run] loaded cached suffixes from {cache}")
        else:
            attacker.prepare(questions, llm=llm, retriever=retriever,
                             gcg_cfg=cfg.gcg, verbose=verbose)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "w", encoding="utf-8") as fh:
                json.dump(attacker.to_dict(), fh, ensure_ascii=False, indent=2)
            print(f"[run] saved suffixes to {cache}")

    town = Town(cfg, questions, llm, retriever, attacker)
    results = town.run(verbose=verbose)

    out = os.path.join(cfg.output_dir, f"{cfg.name}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"[run] ASR={results['asr']:.4f}  speed={results['speed']}  -> {out}")


if __name__ == "__main__":
    main()
