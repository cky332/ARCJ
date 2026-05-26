#!/usr/bin/env python
"""Plot and tabulate TMCHT results (paper Figures 2/3/6/4/7 and Tables 1/2).

Subcommands
-----------
  curves    ASR(t) curves for several result files (Fig. 2 / 3)
  heatmap   per-agent ASR(agent,t) heatmap for one result file (Fig. 6)
  toxicity  retrieval-score & misleading-rate over steps (Fig. 4 / 7)
  table     print an ASR / R(x) comparison table (Table 1 / 2)
"""
import _bootstrap  # noqa: F401

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _series(res):
    items = sorted((int(k), v) for k, v in res["asr_series"].items())
    return [t for t, _ in items], [v for _, v in items]


def cmd_curves(args):
    plt.figure(figsize=(6, 4))
    labels = args.labels or [f"run{i}" for i in range(len(args.inputs))]
    for path, label in zip(args.inputs, labels):
        t, v = _series(_load(path))
        plt.plot(t, v, marker="o", markersize=3, label=label)
    plt.xlabel("Round")
    plt.ylabel("ASR(t)")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[plot] wrote {args.out}")


def cmd_heatmap(args):
    res = _load(args.input)
    rounds = sorted(int(k) for k in res["agent_asr"])
    mat = [[res["agent_asr"][str(r)][i] for r in rounds]
           for i in range(len(res["kinds"]))]
    plt.figure(figsize=(7, 5))
    plt.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1,
               extent=[rounds[0], rounds[-1], len(mat), 0])
    plt.colorbar(label="ASR(agent, t)")
    plt.xlabel("Round")
    plt.ylabel("Agent ID")
    plt.title(res["config"].get("name", "attack"))
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[plot] wrote {args.out}")


def cmd_toxicity(args):
    data = _load(args.input)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for method, series in data.items():
        steps = range(1, len(series["retrieval_score"]) + 1)
        ax1.plot(steps, series["retrieval_score"], marker="o", label=method)
        ax2.plot(steps, series["misleading_rate"], marker="o", label=method)
    ax1.set(title="Retrieval Score", xlabel="Step", ylabel="RS(m_i)")
    ax2.set(title="Misleading Rate", xlabel="Step", ylabel="MR(m_i)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[plot] wrote {args.out}")


def cmd_table(args):
    rows = []
    for path in args.inputs:
        try:
            res = _load(path)
        except FileNotFoundError:
            print(f"[warn] skipping missing file: {path}")
            continue
        c = res["config"]
        rows.append((c.get("name", path), c.get("attack"), c.get("topology"),
                     c.get("num_agents"), c.get("positive_density"),
                     res["asr"], res["speed"]))
    hdr = f"{'name':24s} {'attack':6s} {'topo':6s} {'N':>4s} {'dens':>5s} {'ASR':>7s}  R(20/30/50/75)"
    print(hdr)
    print("-" * len(hdr))
    for name, atk, topo, N, dens, asr, speed in rows:
        sp = "/".join(str(speed.get(x, "-")) for x in ("20", "30", "50", "75"))
        print(f"{str(name):24s} {str(atk):6s} {str(topo):6s} {str(N):>4s} "
              f"{dens:>5.2f} {asr*100:>6.2f}% {sp}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("curves"); p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+"); p.add_argument("--out", default="results/curves.png")
    p.set_defaults(func=cmd_curves)

    p = sub.add_parser("heatmap"); p.add_argument("--input", required=True)
    p.add_argument("--out", default="results/heatmap.png"); p.set_defaults(func=cmd_heatmap)

    p = sub.add_parser("toxicity"); p.add_argument("--input", required=True)
    p.add_argument("--out", default="results/toxicity.png"); p.set_defaults(func=cmd_toxicity)

    p = sub.add_parser("table"); p.add_argument("--inputs", nargs="+", required=True)
    p.set_defaults(func=cmd_table)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
