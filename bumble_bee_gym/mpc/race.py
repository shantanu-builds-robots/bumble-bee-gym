"""Race the gates in MuJoCo (CPU) with the NMPC or the PPO policy, judged by the same rules as training.

    python -m bumble_bee_gym.mpc.race --controller mpc ppo [--seconds 15] [--mass 1.0] [--motor 1.0] [--push 0]
                                      [--video] [--plan outputs/mpc/plan.npz] [--policy outputs/gates_02/...npz]

Both fly the MuJoCo twin at 2 ms physics, 100 Hz control, from the start line at rest (the PPO race film's start).
The judge is a numpy copy of GatesEnv's: gate k counts when the drone's centre crosses its plane inside the opening
without any of its collision spheres touching any gate's frame; the flight ends on a frame touch, a miss (crossing
the target's plane outside the opening, within miss_radius), flipping over, the ground, or straying too far.

Robustness knobs, applied to the simulated drone only (the controllers keep their nominal model): --mass scales its
mass and inertia, --motor scales every rotor's thrust, --push gives a random shove of this speed [m/s] every 1 to 3 s
(as push_01 / the sprout push tests).

Writes outputs/mpc/race_<tag>.json (and .npz paths, and with --video a side-by-side chase-cam mp4).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from .. import course as course_mod, hoop, params as params_mod
from ..policy_np import Policy
from .controller import NMPC, Plan
from .judge import Judge
from .model import Model, quat_to_mat_np
from .pid import PathPID
from .plan import OUT, start_state

REPO = Path(__file__).resolve().parents[2]
PPO_POLICY = REPO / "outputs" / "gates_02" / "policy_000024576000.npz"


def judge_plan(c, p, mod: Model, plan: Plan, dt: float = 0.002) -> tuple[int, str | None, float]:
    """Run the judge along the plan itself (sampled every dt): (gates passed, failure or None, time it ended)."""
    j = Judge(c, p)
    ts = np.arange(0.0, plan.total, dt)
    xs, _ = plan(ts)
    prev = mod.body_origin(xs[0])[0]
    for t, x in zip(ts, xs):
        pos = mod.body_origin(x)[0]
        fail = j.step(prev, pos, quat_to_mat_np(x[6:10]))
        prev = pos
        if fail:
            return j.passed, fail, float(t)
        if j.finished:
            break
    return j.passed, None, float(ts[-1])


def ppo_obs(j: Judge, pos, vel, R, rates, last, lidar=None, lookahead: int = 1) -> np.ndarray:
    """GatesEnv's observation: state in the target's frame, the next `lookahead` gates (past the last gate of a
    point-to-point course, points 5 m apart beyond it), the lidar, the target one-hot."""
    k, n = j.k, j.n
    Rg = j.R[k]
    ahead = []
    for i in range(1, lookahead + 1):
        if j.loop:
            nc, na = j.C[(k + i) % n], j.R[(k + i) % n][:, 0]
        else:
            past, nxt = k + i - (n - 1), min(k + i, n - 1)
            nc = j.C[n - 1] + 5.0 * past * j.R[n - 1][:, 0] if past > 0 else j.C[nxt]
            na = j.R[nxt][:, 0]
        ahead += [Rg.T @ (nc - j.C[k]), Rg.T @ na]
    scan = [lidar.numpy(pos, R)] if lidar is not None else []
    return np.hstack([Rg.T @ (pos - j.C[k]), Rg.T @ vel, (Rg.T @ R).ravel(), rates, last, *ahead, *scan,
                      np.eye(n)[k]])


def policy_layout(policy: Policy, c) -> tuple[int, object]:
    """(lookahead, lidar Scene or None) a policy was trained with: from its run's config.json when there is one,
    else guessed from its input size."""
    from ..learn import lidar as lidar_mod
    cfg = Path(getattr(policy, "path", "")).parent / "config.json"
    ov = json.loads(cfg.read_text()).get("env_overrides") or {} if cfg.is_file() else {}
    size, n = policy.mean.shape[-1], len(c.gates)
    if ov:
        look = int(ov.get("lookahead", 1))
        lid = str(ov.get("lidar.enabled", "false")).lower() == "true"
    else:
        look, lid = 1, size > 28 + n
    lidar = lidar_mod.Scene(c, lidar_mod.settings()) if lid else None
    assert size == 22 + 6 * look + n + (lidar.n if lidar else 0), "policy input size matches no known layout"
    return look, lidar


def fly(controller: str, c, p, mod: Model, seconds=15.0, mass=1.0, motor=1.0, push=0.0, seed=0,
        plan: Plan | None = None, policy: Policy | None = None, mpc_kw=None, ctrl: NMPC | None = None,
        pid: PathPID | None = None) -> dict:
    m = mujoco.MjModel.from_xml_string(hoop.scene_xml(c, p, train=True))
    m.opt.timestep = 0.002
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "drone")
    m.body_mass[body] *= mass
    m.body_inertia[body] *= mass
    d = mujoco.MjData(m)
    x0 = start_state(c, mod)
    d.qpos[:3] = mod.body_origin(x0)[0]
    d.qpos[3:7] = x0[6:10]
    mujoco.mj_forward(m, d)
    judge = Judge(c, p)
    rng = np.random.default_rng(seed)
    hover, wmax = float(p.hover_speed()), float(p.max_rotor_speed)
    c2, c1, c0 = p.thrust_map
    if controller == "pid":
        pid.reset()
    if controller == "mpc":
        if ctrl is None:
            ctrl = NMPC(mod, plan, **(mpc_kw or {}))
        ctrl.reset()
    else:
        ctrl = None
    last = np.zeros(4)
    look, lidar = policy_layout(policy, c) if controller == "ppo" else (1, None)
    next_push, push_left, push_force = rng.uniform(1, 3), 0, np.zeros(3)
    path, thrusts, gate_times, fail = [], [], [], None
    steps = int(round(seconds / 0.01))
    for i in range(steps):
        t = i * 0.01
        pos0 = d.qpos[:3].copy()
        R = quat_to_mat_np(d.qpos[3:7])
        if controller == "mpc":
            u = ctrl(mod.state_from_mujoco(d.qpos, d.qvel), t)
        elif controller == "pid":
            u = pid(d.qpos[:3], d.qvel[:3], d.qpos[3:7], d.qvel[3:6], t)
        else:
            last = policy(ppo_obs(judge, d.qpos[:3], d.qvel[:3], R, d.qvel[3:6], last, lidar, look))
            w = np.clip(hover + hoop.SPEED_SPAN * last, 0, wmax)
            u = np.maximum(0, c2 * w * w + c1 * w + c0)
        d.ctrl[:] = u * motor
        if push > 0:
            if t >= next_push and push_left == 0:
                dirn = rng.normal(size=3)
                dirn /= np.linalg.norm(dirn)
                dur = rng.uniform(0.05, 0.2)
                push_left = int(round(dur / 0.01))
                # half-sine force pulse giving a velocity change of `push` m/s (as the sprout push tests)
                push_force = dirn * m.body_mass[body] * push * np.pi / (2 * dur)
                push_t0, push_dur = t, dur
                next_push = t + dur + rng.uniform(1, 3)
            if push_left > 0:
                d.xfrc_applied[body, :3] = push_force * np.sin(np.pi * (t - push_t0) / push_dur)
                push_left -= 1
            else:
                d.xfrc_applied[body, :3] = 0
        mujoco.mj_step(m, d, 5)
        before = judge.passed
        fail = judge.step(pos0, d.qpos[:3].copy(), quat_to_mat_np(d.qpos[3:7]))
        if judge.passed > before:
            gate_times.append(round(t + 0.01, 2))
        path.append(np.concatenate([d.qpos[:7], [t + 0.01]]))
        thrusts.append(u)
        if fail or judge.finished:
            break
    out = {"controller": controller, "gates": judge.passed, "fail": fail, "finished": judge.finished,
           "finish_s": gate_times[-1] if judge.finished else None, "flight_s": round(len(path) * 0.01, 2),
           "gate_times": gate_times, "mass": mass, "motor": motor, "push": push, "seed": seed,
           "path": np.array(path), "thrusts": np.array(thrusts)}
    if ctrl is not None:
        st = np.array(ctrl.solve_times) * 1e3
        out["solve_ms"] = {"median": round(float(np.median(st)), 2), "p95": round(float(np.percentile(st, 95)), 2),
                           "max": round(float(st.max()), 2)}
    if len(gate_times) >= 33:
        out["t_33_gates"] = gate_times[32]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", default="race_gates")
    ap.add_argument("--controller", nargs="+", default=["mpc", "ppo"], choices=["mpc", "ppo", "pid"])
    ap.add_argument("--path", default=None, help="PID racer's path (path.py); default outputs/mpc/path_<course>.npz")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--mass", type=float, default=1.0)
    ap.add_argument("--motor", type=float, default=1.0)
    ap.add_argument("--push", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--plan", default=str(OUT / "plan.npz"))
    ap.add_argument("--policy", default=str(PPO_POLICY))
    ap.add_argument("--solver", default="ipopt")
    ap.add_argument("--jit", action="store_true", help="compile the MPC problem to C (gcc) for speed")
    ap.add_argument("--horizon", type=int, default=20, help="MPC steps")
    ap.add_argument("--mpc-dt", type=float, default=0.03, help="MPC step [s]")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    c, p = course_mod.load(a.course), params_mod.load()
    mod = Model(p)
    plan = Plan(a.plan) if "mpc" in a.controller else None
    policy = Policy(a.policy) if "ppo" in a.controller else None
    pid = PathPID(p, a.path or OUT / f"path_{a.course}.npz") if "pid" in a.controller else None
    results = []
    for ctl in a.controller:
        for s in range(a.seeds):
            t0 = time.time()
            r = fly(ctl, c, p, mod, a.seconds, a.mass, a.motor, a.push, s, plan, policy,
                    {"solver": a.solver, "jit": a.jit, "N": a.horizon, "dt": a.mpc_dt}, pid=pid)
            summary = {k: v for k, v in r.items() if k not in ("path", "thrusts")}
            print(f"{ctl:3s} seed {s}: {r['gates']} gates, "
                  + (f"FINISHED at {r['finish_s']} s" if r["finished"] else f"{r['fail'] or 'no crash'} after {r['flight_s']} s")
                  + (f", 33 gates at {r['t_33_gates']} s" if "t_33_gates" in r else "")
                  + (f", solve {r['solve_ms']}" if "solve_ms" in r else "") + f"  ({time.time() - t0:.0f} s wall)",
                  flush=True)
            results.append(summary)
            np.savez(OUT / f"race_{ctl}{('_' + a.tag) if a.tag else ''}_s{s}.npz", path=r["path"], thrusts=r["thrusts"],
                     fail=str(r["fail"]), finished=r["finished"])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"race{('_' + a.tag) if a.tag else ''}.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
