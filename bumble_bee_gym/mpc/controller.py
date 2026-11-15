"""Nonlinear MPC that tracks the time-optimal race plan.

Every control step (100 Hz) it solves, from the measured state, a finite-horizon optimal control problem on the
model (model.py): N steps of dt seconds, multiple shooting with one RK4 step per interval,

  min  sum_k |x_k - xr_k|^2_Q + |u_k - ur_k|^2_R  +  |x_N - xr_N|^2_Qn
  s.t. x_0 = measured state,  x_k+1 = F(x_k, u_k),  0 <= u_k <= f_max

and applies the first rotor thrusts. xr, ur are the plan sampled at t, t + dt, ... (states interpolated between plan
nodes, thrusts held). The plan already uses only part of the motors' range (plan.py, thrust_share); here the full
range is allowed, so the controller can push harder than the plan to catch up.

The quaternion is compared component by component, after flipping the measured one onto the plan's hemisphere (q
and -q are the same attitude). The solve is warm-started from the previous one, shifted by a step.
"""

from __future__ import annotations

import time

import casadi as ca
import numpy as np

from .model import NU, NX, Model

# Weights per state component: position, velocity, quaternion, body rates; and per rotor thrust [1/N^2].
Q_DIAG = np.array([400, 400, 400, 10, 10, 10, 0, 60, 60, 60, 1, 1, 1], float)
R_DIAG = np.full(NU, 0.5)


class Plan:
    """The time-optimal plan (plan.py's npz), sampled at any time."""

    def __init__(self, path):
        z = np.load(path)
        self.t, self.x, self.u = z["t"], z["x"], z["u"]
        self.gate_t, self.gate_idx = z["gate_t"], z["gate_idx"]
        self.total = float(self.t[-1])

    def __call__(self, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(states (n, 13), thrusts (n, 4)) at times ts; past the end, hovering at the last state."""
        ts = np.asarray(ts, float)
        x = np.stack([np.interp(ts, self.t, self.x[:, i]) for i in range(NX)], 1)
        x[:, 6:10] /= np.linalg.norm(x[:, 6:10], axis=1, keepdims=True)
        k = np.clip(np.searchsorted(self.t, ts, side="right") - 1, 0, len(self.u) - 1)
        u = self.u[k]
        end = ts >= self.total
        if end.any():
            x[end, 3:6], x[end, 10:13] = 0.0, 0.0
            u[end] = np.mean(self.u[-1])
        return x, u


class NMPC:
    def __init__(self, mod: Model, plan: Plan, N: int = 20, dt: float = 0.03, f_max: float | None = None,
                 solver: str = "ipopt", Q=Q_DIAG, R=R_DIAG, terminal: float = 5.0, sqp_iters: int = 2,
                 jit: bool = False):
        self.mod, self.plan, self.N, self.dt = mod, plan, N, dt
        self.f_max = float(f_max if f_max is not None else mod.f_max)
        F = mod.rk4(dt, steps=1)
        Xs, Us = ca.SX.sym("X", NX, N + 1), ca.SX.sym("U", NU, N)
        P = ca.SX.sym("P", NX + NX * (N + 1) + NU * N)     # x0, xref (N+1), uref (N)
        x0 = P[:NX]
        xr = ca.reshape(P[NX:NX + NX * (N + 1)], NX, N + 1)
        ur = ca.reshape(P[NX + NX * (N + 1):], NU, N)
        Qm, Rm = ca.DM(np.diag(Q)), ca.DM(np.diag(R))
        J, g = 0, [Xs[:, 0] - x0]
        for k in range(N):
            e, du = Xs[:, k] - xr[:, k], Us[:, k] - ur[:, k]
            J += e.T @ Qm @ e + du.T @ Rm @ du
            g.append(Xs[:, k + 1] - F(Xs[:, k], Us[:, k]))
        e = Xs[:, N] - xr[:, N]
        J += terminal * e.T @ Qm @ e
        # Interleave per stage (x_k, u_k) so structure-exploiting solvers see the OCP's banded shape.
        w = ca.vertcat(*[ca.vertcat(Xs[:, k], Us[:, k]) for k in range(N)], Xs[:, N])
        nlp = {"x": w, "f": J, "g": ca.vertcat(*g), "p": P}
        self.nw = w.shape[0]
        lbw, ubw = [], []
        for k in range(N):
            lbw += [-np.inf] * NX + [0.0] * NU
            ubw += [np.inf] * NX + [self.f_max] * NU
        lbw += [-np.inf] * NX
        ubw += [np.inf] * NX
        self.lbw, self.ubw = np.array(lbw), np.array(ubw)
        self.ng = NX * (N + 1)
        if solver == "ipopt":
            opts = {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 50, "ipopt.tol": 1e-4,
                    "ipopt.warm_start_init_point": "yes", "ipopt.mu_init": 1e-3, "ipopt.linear_solver": "mumps"}
            if jit:
                # Compile the problem's functions (cost, constraints, their derivatives) to C once, with gcc.
                opts.update({"jit": True, "compiler": "shell", "jit_options": {"flags": ["-O2"], "verbose": False}})
        elif solver == "fatrop":
            # Interior point that exploits the OCP's stage structure (Vanroye et al., IROS 2023), found
            # automatically from the stage-interleaved variables and the gap constraints.
            opts = {"structure_detection": "auto", "expand": True, "print_time": False, "debug": False,
                    "equality": [True] * (NX * (N + 1)),
                    "fatrop": {"print_level": 0, "max_iter": 50, "tol": 1e-4, "mu_init": 1e-2}}
        elif solver.startswith("sqp"):
            # Real-time iteration style: a fixed couple of SQP steps per control step, each a sparse QP, warm-started
            # from the previous solution ("sqp-osqp", "sqp-daqp", "sqp-qrqp"; plain "sqp" is OSQP).
            qp = solver.split("-")[1] if "-" in solver else "osqp"
            qp_opts = {"error_on_fail": False, "print_problem": False} if qp != "osqp" else \
                {"error_on_fail": False, "warm_start_primal": True,
                 "osqp": {"verbose": False, "eps_abs": 1e-5, "eps_rel": 1e-5, "max_iter": 4000, "polish": True}}
            if qp == "qrqp":
                qp_opts.update({"print_iter": False, "print_header": False, "print_info": False})
            opts = {"qpsol": qp, "max_iter": sqp_iters, "print_time": False, "print_header": False,
                    "print_iteration": False, "print_status": False, "error_on_fail": False,
                    "qpsol_options": qp_opts, "convexify_strategy": "regularize", "hessian_approximation": "exact"}
            solver = "sqpmethod"
        else:
            raise ValueError(solver)
        self.solver = ca.nlpsol("nmpc", solver, nlp, opts)
        self.w0 = None
        self.lam_g = None
        self.solve_times: list[float] = []
        self.stage = NX + NU

    def reset(self) -> None:
        """Forget the warm start and timing (for a new flight with the same compiled solver)."""
        self.w0, self.lam_g, self.solve_times = None, None, []

    def _unpack(self, w):
        w = np.asarray(w).ravel()
        st = self.stage
        X = np.stack([w[k * st:k * st + NX] for k in range(self.N)] + [w[self.N * st:]])
        U = np.stack([w[k * st + NX:(k + 1) * st] for k in range(self.N)])
        return X, U

    def _pack(self, X, U):
        return np.concatenate([np.concatenate([X[k], U[k]]) for k in range(self.N)] + [X[self.N]])

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Rotor thrusts [N] for measured state x at race time t."""
        ts = t + self.dt * np.arange(self.N + 1)
        xr, ur = self.plan(ts)
        x = np.array(x, float)
        if np.dot(x[6:10], xr[0, 6:10]) < 0:
            x[6:10] *= -1
        # Keep every reference quaternion on the same hemisphere as the first.
        flip = np.sum(xr[:, 6:10] * xr[0, 6:10], axis=1) < 0
        xr[flip, 6:10] *= -1
        p = np.concatenate([x, xr.ravel(), ur[:-1].ravel()])
        if self.w0 is None:
            Xg = xr.copy()
            Xg[0] = x
            self.w0 = self._pack(Xg, ur[:-1])
        args = {"x0": self.w0, "p": p, "lbx": self.lbw, "ubx": self.ubw, "lbg": 0, "ubg": 0}
        if self.lam_g is not None:
            args["lam_g0"] = self.lam_g
        t0 = time.perf_counter()
        sol = self.solver(**args)
        self.solve_times.append(time.perf_counter() - t0)
        X, U = self._unpack(sol["x"])
        self.lam_g = np.asarray(sol["lam_g"]).ravel()
        # Warm start for the next step: shift by one interval only when a control step (10 ms) has eaten into it;
        # a 30 ms interval is mostly still ahead, so reusing the solution as is is closer than shifting.
        self.w0 = self._pack(X, U)
        self.X, self.U = X, U
        return np.clip(U[0], 0.0, self.f_max)
