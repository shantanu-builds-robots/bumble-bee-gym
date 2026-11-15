"""PPO vs NMPC on the race gates, under the same conditions: nominal, model mismatch, and random shoves.

    python -m bumble_bee_gym.mpc.compare [--seeds 5] [--out outputs/mpc]

Run from a scratch directory: the NMPC is compiled to C once (gcc, about two minutes) and its build files land in the
current directory. Each condition is flown by both controllers from the start line for 15 s (race.fly); the
mismatch is applied to the simulated drone only, never to the controllers' models. Writes compare.json, compare.md (a
table) and compare.png (gates passed per condition).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .. import course as course_mod, params as params_mod
from ..policy_np import Policy
from .controller import NMPC, Plan
from .model import Model
from .plan import OUT
from .race import PPO_POLICY, fly

CONDITIONS = [  # (name, mass scale, motor scale, shove [m/s], random?)
    ("nominal", 1.0, 1.0, 0.0, False),
    ("mass +10%", 1.1, 1.0, 0.0, False),
    ("mass +20%", 1.2, 1.0, 0.0, False),
    ("motors -10%", 1.0, 0.9, 0.0, False),
    ("shoves 1 m/s", 1.0, 1.0, 1.0, True),
    ("shoves 2 m/s", 1.0, 1.0, 2.0, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="flights per shove condition")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--plan", default=str(OUT / "plan.npz"))
    ap.add_argument("--policy", default=str(PPO_POLICY))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    c, p = course_mod.load("race_gates"), params_mod.load()
    mod, plan, policy = Model(p), Plan(a.plan), Policy(a.policy)
    t0 = time.time()
    nmpc = NMPC(mod, plan, jit=True)
    print(f"NMPC compiled in {time.time() - t0:.0f} s", flush=True)

    rows = []
    for name, mass, motor, push, random in CONDITIONS:
        for ctl in ("ppo", "mpc"):
            runs = []
            for s in range(a.seeds if random else 1):
                r = fly(ctl, c, p, mod, a.seconds, mass, motor, push, s, plan, policy, ctrl=nmpc if ctl == "mpc" else None)
                runs.append({k: v for k, v in r.items() if k not in ("path", "thrusts")})
                if s == 0:
                    np.savez(out / f"cmp_{ctl}_{name.replace(' ', '_').replace('%', 'pct').replace('/', '')}.npz",
                             path=r["path"], thrusts=r["thrusts"])
            g = np.array([r["gates"] for r in runs])
            crashed = np.array([r["fail"] is not None for r in runs])
            t33 = [r["t_33_gates"] for r in runs if "t_33_gates" in r]
            row = {"condition": name, "controller": ctl, "flights": len(runs), "gates_mean": float(g.mean()),
                   "gates_min": int(g.min()), "crashed": int(crashed.sum()),
                   "t_33_gates": round(float(np.mean(t33)), 2) if t33 else None,
                   "fails": [r["fail"] for r in runs if r["fail"]],
                   "solve_ms": runs[0].get("solve_ms")}
            rows.append(row)
            print(f"{name:13s} {ctl}: gates {g.mean():5.1f} (min {g.min()}), crashed {crashed.sum()}/{len(runs)}"
                  + (f", 33 gates at {row['t_33_gates']} s" if t33 else "")
                  + (f", fails {row['fails']}" if row["fails"] else ""), flush=True)

    with open(out / "compare.json", "w") as f:
        json.dump(rows, f, indent=1)
    lines = ["| condition | controller | flights | gates in 15 s (mean / min) | crashed | time to 33 gates |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['condition']} | {r['controller'].upper()} | {r['flights']} | {r['gates_mean']:.1f} / "
                     f"{r['gates_min']} | {r['crashed']}/{r['flights']} | "
                     f"{str(r['t_33_gates']) + ' s' if r['t_33_gates'] else '-'} |")
    (out / "compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n, *_ in CONDITIONS]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    xs = np.arange(len(names))
    for i, (ctl, col) in enumerate((("ppo", "#2a78d6"), ("mpc", "#d6612a"))):
        vals = [next(r["gates_mean"] for r in rows if r["condition"] == n and r["controller"] == ctl) for n in names]
        ax.bar(xs + (i - 0.5) * 0.38, vals, 0.38, color=col, label={"ppo": "PPO (learned)", "mpc": "NMPC + time-optimal plan"}[ctl])
    ax.axhline(33, color="0.6", lw=0.8, ls="--")
    ax.set_xticks(xs, names)
    ax.set_ylabel("gates passed in 15 s (mean)")
    ax.set_title("Race gates: learned policy vs model predictive control, same drone, same start")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "compare.png", dpi=130)
    print(out / "compare.png")


if __name__ == "__main__":
    main()
