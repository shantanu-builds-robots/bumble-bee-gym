"""Time-optimal race plan: the fastest flight through the gates that the drone model can fly.

    python -m bumble_bee_gym.mpc.plan [--laps 4] [--thrust-share 0.9] [--out outputs/mpc/plan.npz]

One optimal control problem over the whole race (Foehn, Romero and Scaramuzza, Science Robotics 2021, with the
gate-to-node assignment fixed in advance): the flight is split into one phase per gate, each with its own number of
nodes and its own (free) time step, and the full rigid-body model (model.py) links consecutive nodes. Minimised: the
total time, plus a small cost on body rates and thrust changes to keep the plan smooth enough to track.

Constraints:
  - starts at rest, level, on the start line, facing gate 0, and after the last gate brakes to a level hover;
  - each phase ends on its gate's plane, moving forward through it, with every one of the drone's collision spheres
    (body and props, at the planned attitude) crossing the plane inside the opening;
  - rotor thrust within thrust_share of the motors' limit (the rest is the tracking controller's margin);
  - body rates and tilt bounded, and the centre of mass above min_height.

Solved once, offline, with IPOPT. The result (node times, states, rotor thrusts, gate times) is what the MPC tracks.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import casadi as ca
import numpy as np

from .. import course as course_mod, hoop, params as params_mod
from .model import NU, NX, Model, quat_to_mat, quat_to_mat_np, yaw_quat
from .path import REACH, Boxes

OUT = Path(__file__).resolve().parents[2] / "outputs" / "mpc"

FINISH_NODES = 30
OBST_POW = 10
GATE_MARGIN = 0.03   # [m] kept between every collision sphere and the gate's frame, on top of the sphere's radius


def start_state(c: course_mod.Course, mod: Model) -> np.ndarray:
    """At rest and level at the race's start, facing gate 0: the spawn point on a point-to-point course, else 30% of
    the way from the last gate to gate 0 (as the PPO race film starts)."""
    from .path import start_point
    base, yaw = start_point(c)
    q = yaw_quat(yaw)
    return np.concatenate([base + quat_to_mat_np(q) @ mod.com, np.zeros(3), q, np.zeros(3)])


def _seg_dist(p, a, b) -> float:
    ab = b - a
    s = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + s * ab)))


def _path_guess(c, mod, path_npz, gate_idx, N, speedup):
    """Per phase: node states (positions and velocities from the path, sped up) and a time step."""
    from .judge import Judge
    z = np.load(path_npz)
    t, P, V = z["t"], z["p"], z["v"]
    j, times = Judge(c, mod.p), [0.0]
    for i in range(1, len(P)):
        before = j.passed
        j.step(P[i - 1], P[i], np.eye(3))
        if j.passed > before:
            times.append(t[i])
    out = []
    for ph in range(len(gate_idx)):
        t0, t1 = times[ph], times[min(ph + 1, len(times) - 1)]
        ts = np.linspace(t0, t1, N[ph] + 1)
        gx = np.zeros((NX, N[ph] + 1))
        gx[:3] = np.stack([np.interp(ts, t, P[:, k]) for k in range(3)])
        gx[3:6] = np.stack([np.interp(ts, t, V[:, k]) for k in range(3)]) * speedup
        out.append((gx, max((t1 - t0) / N[ph] / speedup, 0.006)))
    return out


def solve(c: course_mod.Course, mod: Model, laps: int = 4, thrust_share: float = 0.9, nodes_per_m: float = 4.0,
          max_rate: float = 12.0, max_tilt_deg: float = 70.0, max_yaw_rate: float = 3.0, min_height: float = 0.3, v_guess: float = 8.0,
          max_iter: int = 3000, verbose: bool = True, guess_path: str | None = None, guess_speedup: float = 1.6,
          obstacle_margin: float = 0.2) -> dict:
    loop = bool(c.extra.get("loop", True))
    gate_idx = list(range(len(c.gates))) * laps + [0] if loop else list(range(len(c.gates)))
    gates = [c.gates[i] for i in gate_idx]
    frames = [hoop.hoop_frame(g) for g in gates]
    x0 = start_state(c, mod)
    sph_c, sph_r = hoop.collision_spheres(mod.p)                  # body frame (base_link origin)
    boxes = Boxes(c)

    pts = np.vstack([x0[:3]] + [f[0] for f in frames])
    dist = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    N = [max(8, int(round(d * nodes_per_m))) for d in dist]

    F = mod.rk4(None, steps=1)
    opti = ca.Opti()
    dts = opti.variable(len(gates))
    X = [opti.variable(NX, n + 1) for n in N]
    U = [opti.variable(NU, n) for n in N]
    cost = 0
    opti.subject_to(X[0][:, 0] == x0)
    for j, (g, (centre, rot)) in enumerate(zip(gates, frames)):
        Xj, Uj, h = X[j], U[j], dts[j]
        if j > 0:
            opti.subject_to(Xj[:, 0] == X[j - 1][:, -1])
        for k in range(N[j]):
            opti.subject_to(Xj[:, k + 1] == F(Xj[:, k], Uj[:, k], h))
        opti.subject_to(opti.bounded(0, ca.vec(Uj), thrust_share * mod.f_max))
        opti.subject_to(opti.bounded(-max_rate, ca.vec(Xj[10:12, :]), max_rate))
        opti.subject_to(opti.bounded(-max_yaw_rate, Xj[12, :], max_yaw_rate))
        opti.subject_to(Xj[2, :] >= min_height)
        # Tilt: the body z axis's world z component, 1 - 2 (qx^2 + qy^2), at least cos(max_tilt). The race counts
        # any tilt past horizontal as flipped over, and a time-optimal flight would otherwise use it.
        opti.subject_to(1 - 2 * (Xj[7, :] ** 2 + Xj[8, :] ** 2) >= np.cos(np.radians(max_tilt_deg)))
        opti.subject_to(opti.bounded(0.005, h, 0.12))
        # Obstacles near this phase: every node outside the box inflated by the drone's reach plus a margin, as a
        # smooth box-shaped constraint (a superellipsoid, power OBST_POW, which hugs the inflated box except at its
        # edges, where it rounds in by under 7%).
        a_seg, b_seg = pts[j], pts[j + 1]
        for oc, oh, orot in zip(boxes.c, boxes.h, boxes.r):
            if _seg_dist(oc, a_seg, b_seg) > np.linalg.norm(oh) + 6.0:
                continue
            ax = oh + REACH + obstacle_margin
            local = ca.DM(orot.T) @ (Xj[:3, :] - ca.DM(oc))
            opti.subject_to(ca.sum1((local / ca.DM(ax)) ** OBST_POW) >= 1.0)
        # Through the gate at the phase's last node: the centre of mass on the gate's plane, moving forward, and
        # every collision sphere (hoop.collision_spheres, as the race judges a frame touch) crossing the plane inside
        # the opening by its radius plus GATE_MARGIN. Each sphere's crossing point is extrapolated along the
        # velocity from where it is at that node, which is exact for the straight, fast flight through a gate.
        xe = Xj[:, -1]
        Rg, Rb = ca.DM(rot.T), quat_to_mat(xe[6:10])
        pg, vg = Rg @ (xe[:3] - ca.DM(centre)), Rg @ xe[3:6]
        half = np.array(g.opening) / 2
        opti.subject_to(pg[0] == 0)
        opti.subject_to(vg[0] >= 1.0)
        for sc, sr in zip(sph_c, sph_r):
            s_g = pg + Rg @ Rb @ ca.DM(sc - mod.com)                       # sphere centre, gate frame
            lim = half - sr - GATE_MARGIN
            # Where the sphere is as it enters the frame's depth, at the plane, and as it leaves: a diagonal pass
            # can clear the opening at the plane yet clip a bar's front or back edge.
            for xp in (-(g.depth / 2 + sr), 0.0, g.depth / 2 + sr):
                at = s_g[1:] + (xp - s_g[0]) * vg[1:] / vg[0]
                opti.subject_to(opti.bounded(-lim, at, lim))
        cost += N[j] * h
        cost += 1e-3 * h * ca.sumsqr(Xj[10:13, :])
        cost += 1e-4 * ca.sumsqr(ca.diff(ca.horzcat(Uj), 1, 1))

    # Finish: after the last gate, brake to a hover (anywhere clear of the course), so the plan ends somewhere the
    # tracking controller can stay.
    Xf, Uf, hf = opti.variable(NX, FINISH_NODES + 1), opti.variable(NU, FINISH_NODES), opti.variable()
    opti.subject_to(Xf[:, 0] == X[-1][:, -1])
    for k in range(FINISH_NODES):
        opti.subject_to(Xf[:, k + 1] == F(Xf[:, k], Uf[:, k], hf))
    opti.subject_to(opti.bounded(0, ca.vec(Uf), thrust_share * mod.f_max))
    opti.subject_to(opti.bounded(-max_rate, ca.vec(Xf[10:13, :]), max_rate))
    opti.subject_to(Xf[2, :] >= min_height + 0.5)
    opti.subject_to(1 - 2 * (Xf[7, :] ** 2 + Xf[8, :] ** 2) >= np.cos(np.radians(max_tilt_deg)))
    opti.subject_to(opti.bounded(0.01, hf, 0.1))
    opti.subject_to(Xf[3:6, -1] == 0)
    opti.subject_to(Xf[10:13, -1] == 0)
    opti.subject_to(Xf[7:9, -1] == 0)                                   # level
    cost += 0.1 * FINISH_NODES * hf + 1e-3 * ca.sumsqr(ca.diff(Uf, 1, 1))

    # Initial guess: straight lines between gates at v_guess, level, hovering thrust; or, given a collision-free path
    # (path.py), that path flown guess_speedup times faster.
    yaw = float(np.arctan2(x0[7 + 2], x0[6]) * 2)
    guess = _path_guess(c, mod, guess_path, gate_idx, N, guess_speedup) if guess_path else None
    for j in range(len(gates)):
        if guess is not None:
            gx, gdt = guess[j]
            gx[6:10] = yaw_quat(yaw)[:, None]
            opti.set_initial(X[j], gx)
            opti.set_initial(U[j], mod.hover_thrust)
            opti.set_initial(dts[j], gdt)
            continue
        a, b = pts[j], pts[j + 1]
        d = (b - a) / max(dist[j], 1e-6)
        s = np.linspace(0, 1, N[j] + 1)
        guess = np.zeros((NX, N[j] + 1))
        guess[:3] = a[:, None] + (b - a)[:, None] * s
        guess[3:6] = (d * v_guess)[:, None]
        guess[6:10] = yaw_quat(yaw)[:, None]
        opti.set_initial(X[j], guess)
        opti.set_initial(U[j], mod.hover_thrust)
        opti.set_initial(dts[j], dist[j] / v_guess / N[j])
    end_dir = frames[-1][1][:, 0]
    fin = np.zeros((NX, FINISH_NODES + 1))
    fin[:3] = pts[-1][:, None] + end_dir[:, None] * np.linspace(0, 2.0, FINISH_NODES + 1)
    fin[6:10] = yaw_quat(yaw)[:, None]
    opti.set_initial(Xf, fin)
    opti.set_initial(Uf, mod.hover_thrust)
    opti.set_initial(hf, 0.03)

    opti.minimize(cost)
    opti.solver("ipopt", {"expand": True, "print_time": False},
                {"max_iter": max_iter, "print_level": 5 if verbose else 0, "tol": 1e-6,
                 "linear_solver": "mumps", "mu_strategy": "adaptive"})
    t0 = time.time()
    try:
        sol = opti.solve()
        status = "solved"
    except RuntimeError:
        sol = opti.debug
        status = "not solved: " + opti.stats()["return_status"]
    wall = time.time() - t0

    dt = np.array(sol.value(dts)).ravel()
    xs, us, ts, gate_t, t = [x0[:, None]], [], [0.0], [], 0.0
    for j in range(len(gates)):
        Xj, Uj = np.array(sol.value(X[j])), np.array(sol.value(U[j])).reshape(NU, -1)
        xs.append(Xj[:, 1:])
        us.append(Uj)
        ts += list(t + dt[j] * np.arange(1, N[j] + 1))
        t += dt[j] * N[j]
        gate_t.append(t)
    hf_v = float(sol.value(hf))
    xs.append(np.array(sol.value(Xf))[:, 1:])
    us.append(np.array(sol.value(Uf)).reshape(NU, -1))
    ts += list(t + hf_v * np.arange(1, FINISH_NODES + 1))
    return {"t": np.array(ts), "x": np.hstack(xs).T, "u": np.hstack(us).T, "gate_t": np.array(gate_t),
            "gate_idx": np.array(gate_idx), "status": status, "solve_s": wall,
            "laps": laps, "thrust_share": thrust_share, "f_max": mod.f_max}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", default="race_gates")
    ap.add_argument("--laps", type=int, default=4)
    ap.add_argument("--thrust-share", type=float, default=0.9)
    ap.add_argument("--nodes-per-m", type=float, default=4.0)
    ap.add_argument("--max-tilt", type=float, default=70.0)
    ap.add_argument("--guess-path", default=None, help="collision-free path (path.py) to start the solver from")
    ap.add_argument("--out", default=str(OUT / "plan.npz"))
    a = ap.parse_args()
    c = course_mod.load(a.course)
    mod = Model(params_mod.load())
    r = solve(c, mod, a.laps, a.thrust_share, a.nodes_per_m, max_tilt_deg=a.max_tilt, guess_path=a.guess_path)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **{k: v for k, v in r.items()})
    sp = np.linalg.norm(r["x"][:, 3:6], axis=1)
    print(f"{r['status']} in {r['solve_s']:.0f} s: {len(r['gate_t'])} gates in {r['gate_t'][-1]:.2f} s, "
          f"top speed {sp.max():.1f} m/s, rotor thrust up to {r['u'].max():.2f} of {mod.f_max:.2f} N")
    print("  gate times: " + " ".join(f"{t:.2f}" for t in r["gate_t"]))
    from .controller import Plan
    from .race import judge_plan
    passed, fail, t = judge_plan(c, mod.p, mod, Plan(a.out))
    print(f"  judged as flown exactly: {passed} gates, " + (f"FAILS ({fail}) at {t:.2f} s" if fail else "no failure"))
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
