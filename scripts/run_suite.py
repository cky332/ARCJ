#!/usr/bin/env python
"""Run the TMCHT experiment suite in ONE process.

The LLM + DPR retriever are loaded once and reused across every
(attack, topology, density, scale) combination, and each attack's adversarial
suffix is optimized once and reused everywhere. This avoids run_all.sh reloading
the 8B model 50+ times.

Examples
--------
    # Full structure table (Table 1): graph/line/star x {1%,50%,99%} x {clean,gcg,arcj}
    CUDA_VISIBLE_DEVICES=0 python scripts/run_suite.py --config configs/structure_graph.yaml \
        --suite structure

    # Fast sanity sweep: one density, fewer rounds
    python scripts/run_suite.py --config configs/structure_graph.yaml \
        --suite structure --densities 0.5 --rounds 80

    # Full paper suite (Table 1 + Table 2). Heavy -- run in tmux / nohup.
    python scripts/run_suite.py --config configs/structure_graph.yaml --suite both
"""
import _bootstrap  # noqa: F401

import argparse
import copy
import json
import os
import time

from arcj.attacks import build_attacker
from arcj.config import ExperimentConfig
from arcj.dataset import load_questions
from arcj.town import Town
from run_experiment import build_models, suffix_cache_path


def _float_list(s):
    return [float(x) for x in s.split(",")] if s else None


def _int_list(s):
    return [int(x) for x in s.split(",")] if s else None


def build_jobs(args):
    """Return a list of job dicts; dedup identical (topology,N,density,attack)."""
    densities = args.densities or [0.01, 0.5, 0.99]
    attacks = args.attacks or ["clean", "gcg", "arcj"]
    jobs, seen = [], set()

    def add(topology, n, density, attack, rounds, name):
        key = (topology, n, round(density, 4), attack)
        if key in seen:
            return
        seen.add(key)
        jobs.append(dict(topology=topology, num_agents=n, density=density,
                         attack=attack, rounds=rounds, name=name))

    if args.suite in ("structure", "both"):
        for topo in (args.topologies or ["graph", "line", "star"]):
            for atk in attacks:
                for d in densities:
                    add(topo, 20, d, atk, args.rounds,
                        f"structure_{topo}_{atk}_d{d}")
    if args.suite in ("scale", "both"):
        for n in (args.scales or [6, 20, 100]):
            r = args.scale_rounds if (n >= 100 and args.scale_rounds) else args.rounds
            for atk in attacks:
                for d in densities:
                    add("graph", n, d, atk, r, f"scale_{n}_{atk}_d{d}")
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--suite", choices=["structure", "scale", "both"], default="structure")
    ap.add_argument("--attacks", type=lambda s: s.split(","))
    ap.add_argument("--topologies", type=lambda s: s.split(","))
    ap.add_argument("--densities", type=_float_list)
    ap.add_argument("--scales", type=_int_list)
    ap.add_argument("--rounds", type=int, help="rounds for all sims (default: config)")
    ap.add_argument("--scale-rounds", type=int, dest="scale_rounds",
                    help="separate (smaller) rounds for 100-agent sims")
    ap.add_argument("--num-questions", type=int, dest="num_questions")
    ap.add_argument("--model")
    ap.add_argument("--device")
    ap.add_argument("--output-dir", dest="output_dir", default="results")
    ap.add_argument("--reoptimize", action="store_true",
                    help="ignore cached suffixes and re-optimize")
    args = ap.parse_args()

    base = ExperimentConfig.from_yaml(args.config)
    if args.model: base.model.llm_name = args.model
    if args.device: base.model.device = args.device
    if args.num_questions: base.num_questions = args.num_questions
    if args.rounds: base.num_rounds = args.rounds
    base.output_dir = args.output_dir
    os.makedirs(base.output_dir, exist_ok=True)

    jobs = build_jobs(args)
    print(f"[suite] {len(jobs)} simulations: suite={args.suite} "
          f"model={base.model.llm_name}")

    # ---- load models once ----------------------------------------------------
    questions = load_questions(os.path.join(base.data_dir, "questions.json"),
                               num_questions=base.num_questions,
                               randomize_correct=base.randomize_correct, seed=base.seed)
    llm, retriever = build_models(base)

    # ---- optimize each attack's suffix once ----------------------------------
    attackers = {}
    for atk_name in sorted({j["attack"] for j in jobs}):
        attacker = build_attacker(atk_name, base.arcj_mode)
        if attacker.has_attacker:
            cfg_for_path = copy.copy(base)
            cfg_for_path.attack = atk_name
            cache = suffix_cache_path(cfg_for_path)
            if (not args.reoptimize) and os.path.exists(cache):
                attacker.load_dict(json.load(open(cache, encoding="utf-8")))
                print(f"[suite] {atk_name}: loaded cached suffixes from {cache}")
            else:
                print(f"[suite] {atk_name}: optimizing suffixes ...")
                attacker.prepare(questions, llm=llm, retriever=retriever,
                                 gcg_cfg=base.gcg, verbose=True)
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                json.dump(attacker.to_dict(), open(cache, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=2)
                print(f"[suite] {atk_name}: saved suffixes to {cache}")
        attackers[atk_name] = attacker

    # ---- run every job, reusing models + suffixes ----------------------------
    summary = []
    t0 = time.time()
    for i, job in enumerate(jobs, 1):
        cfg = copy.deepcopy(base)
        cfg.topology = job["topology"]
        cfg.num_agents = job["num_agents"]
        cfg.positive_density = job["density"]
        cfg.attack = job["attack"]
        cfg.name = job["name"]
        if job["rounds"]:
            cfg.num_rounds = job["rounds"]
        out = os.path.join(cfg.output_dir, f"{cfg.name}.json")
        if os.path.exists(out):
            res = json.load(open(out, encoding="utf-8"))
            print(f"[suite] ({i}/{len(jobs)}) {cfg.name}: exists, skip "
                  f"(ASR={res['asr']:.3f})")
            summary.append((cfg.name, res["asr"], res["speed"]))
            continue
        elapsed = (time.time() - t0) / 60
        print(f"\n[suite] ({i}/{len(jobs)}) {cfg.name} "
              f"N={cfg.num_agents} rounds={cfg.num_rounds} | elapsed {elapsed:.0f}m")
        town = Town(cfg, questions, llm, retriever, attackers[cfg.attack])
        res = town.run(verbose=True)
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[suite] -> ASR={res['asr']:.4f} speed={res['speed']}")
        summary.append((cfg.name, res["asr"], res["speed"]))

    # ---- summary -------------------------------------------------------------
    sumpath = os.path.join(base.output_dir, f"suite_{args.suite}_summary.txt")
    with open(sumpath, "w", encoding="utf-8") as fh:
        header = f"{'name':32s} {'ASR':>7s}  R(20/30/50/75)"
        fh.write(header + "\n" + "-" * len(header) + "\n")
        for name, asr, speed in summary:
            sp = "/".join(str(speed.get(x, "-")) for x in ("20", "30", "50", "75"))
            fh.write(f"{name:32s} {asr*100:>6.2f}%  {sp}\n")
    print(f"\n[suite] done in {(time.time()-t0)/60:.0f}m. summary -> {sumpath}")
    print(open(sumpath, encoding="utf-8").read())


if __name__ == "__main__":
    main()
