"""Ghost race film: recorded flights raced together in one night arena (arena.py), vertical 1080x1920, 50 fps.

    python -m bumble_bee_gym.mpc.ghost --course long_track \
        outputs/mpc/race_mpc_long_s0.npz:NMPC:green outputs/mpc/race_ppo_long_s0.npz:PPO:blue \
        outputs/mpc/race_pid_long_s0.npz:PID:orange --line outputs/mpc/plan_long_track.npz \
        [--out outputs/mpc/ghost_race_long.mp4]

Each racer is <flight.npz>:<name>:<colour>, as written by race.py (path, and whether it finished or how it failed).
The drones never touch each other: each flight was flown alone, and the film only overlays them on one race clock.
``--line`` draws a plan (plan.py) as a thin continuous white line along the course: the ideal line, with no drone on
it.

Runs in the playground image (OpenGL + ffmpeg). One drone is the scene's own; every other racer's drone is added to the
scene from its own MjData (mjv_addGeoms, dynamic geoms only) and recoloured. Behind each drone a streak of its colour,
thin and faded at the tail, thick and bright at the drone. Overlays: a 3-2-1 countdown, a banner when the leader enters
a new section of the course, a leaderboard (gates, FINISHED with the time, or OUT with what was hit), and a results
card at the end. The camera chases the leading racer that is still flying; once it finishes, the next one.
Once at most one racer is still flying, the film plays at FAST times real time (marked on screen); the race clock
stays in real seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import course as course_mod, hoop, params as params_mod
from .arena import arena_xml, gate_colour
from .film import FONT, gate_times
from .judge import quat_to_mat

COLOURS = {"orange": (1.0, 0.5, 0.05), "blue": (0.2, 0.55, 1.0), "white": (0.95, 0.95, 0.95),
           "green": (0.15, 1.0, 0.35), "red": (1.0, 0.2, 0.2), "purple": (0.7, 0.35, 1.0),
           "yellow": (1.0, 0.9, 0.1), "cyan": (0.1, 0.9, 1.0)}
FAIL_TEXT = {"frame": "hit a gate", "obstacle": "hit an obstacle", "missed": "missed a gate", "flipped": "flipped over",
             "ground": "hit the ground", "away": "lost the track"}
FPS, COUNTDOWN, TAIL, STREAK_S = 50, 1.6, 2.0, 0.8
FAST = 2.5           # playback speed once at most one racer is still flying


def plan_line(path_npz, p) -> np.ndarray:
    """The plan's base_link positions every 50 ms."""
    z = np.load(path_npz)
    t = np.arange(0.0, float(z["t"][-1]), 0.05)
    x = np.stack([np.interp(t, z["t"], z["x"][:, i]) for i in range(10)], 1)
    q = x[:, 6:10] / np.linalg.norm(x[:, 6:10], axis=1, keepdims=True)
    com = np.asarray(p.com, float)
    return np.stack([xi[:3] - quat_to_mat(qi) @ com for xi, qi in zip(x, q)])


class Racer:
    def __init__(self, spec, c, p):
        src, self.name, colour = spec.split(":")
        z = np.load(src)
        self.path = z["path"]
        self.rgb = COLOURS[colour]
        self.gates = gate_times(self.path, c, p)
        self.finished = bool(z["finished"]) if "finished" in z.files else len(self.gates) == len(c.gates)
        fail = str(z["fail"]) if "fail" in z.files else "None"
        self.fail = None if fail == "None" or self.finished else fail
        self.end_t = float(self.path[-1, 7])

    def index(self, t: float) -> int:
        return int(np.clip(round(t / 0.01) - 1, 0, len(self.path) - 1))

    def status(self, t: float) -> tuple[int, str]:
        """(gates passed, 'racing' | 'finished' | 'out') at race time t."""
        k = sum(1 for g in self.gates if g <= t + 1e-9)
        if t >= self.end_t:
            return k, "finished" if self.finished else "out"
        return k, "racing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("racers", nargs="+", help="<flight.npz>:<name>:<colour>")
    ap.add_argument("--course", default="long_track")
    ap.add_argument("--line", default=None, help="plan (plan.py npz) drawn as the ideal line")
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--distance", type=float, default=4.0, help="chase camera distance [m]")
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--out", default="outputs/mpc/ghost_race_long.mp4")
    a = ap.parse_args()
    c, p = course_mod.load(a.course), params_mod.load()
    racers = [Racer(s, c, p) for s in a.racers]
    for r in racers:
        print(f"{r.name}: {len(r.gates)} gates, " + (f"finished at {r.gates[-1]:.2f} s" if r.finished
                                                    else f"out ({r.fail}) at {r.end_t:.2f} s"), flush=True)
    line = plan_line(a.line, p) if a.line else None
    race_s = min(max(r.end_t for r in racers) + TAIL, a.max_seconds)

    m = mujoco.MjModel.from_xml_string(arena_xml(c, p))
    m.vis.global_.offwidth, m.vis.global_.offheight = a.width, a.height
    body_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "base_link_visual")
    datas = [mujoco.MjData(m) for _ in racers]
    rend = mujoco.Renderer(m, a.height, a.width, max_geom=8000)
    scn = rend.scene
    opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation = a.distance, -16
    spin = hoop.PropSpinner(m, p)
    fonts = {s: ImageFont.truetype(FONT, s) for s in (38, 46, 64, 96, 220)}
    sections = c.extra.get("sections", [])

    frames, look, az, banner = [], None, None, (-1, -10.0)
    clock, speed = -COUNTDOWN, 1.0
    while clock < race_s:
        t = max(clock, 0.0)
        spin()
        st = []
        for r, d in zip(racers, datas):
            d.qpos[:7] = r.path[r.index(t), :7]
            mujoco.mj_forward(m, d)
            st.append(r.status(t))

        # Camera: the leader among racers still flying (most gates, then furthest along); if none, the last leader.
        flying = [j for j, s in enumerate(st) if s[1] == "racing"] or list(range(len(racers)))
        lead = max(flying, key=lambda j: (st[j][0], -racers[j].gates[-1] if racers[j].gates else 0))
        path, i = racers[lead].path, racers[lead].index(t)
        pos = path[i, :3]
        vel = path[min(i + 4, len(path) - 1), :3] - path[max(i - 4, 0), :3]
        heading = np.degrees(np.arctan2(vel[1], vel[0])) if np.linalg.norm(vel[:2]) > 0.02 else (az or 0.0)
        az = heading if az is None else az + 0.07 * ((heading - az + 180) % 360 - 180)
        look = pos.copy() if look is None else 0.8 * look + 0.2 * pos
        cam.lookat[:], cam.azimuth = look, az

        def view(camera):
            rend.update_scene(datas[0], camera)
            _recolour(scn, 0, scn.ngeom, body_geom, racers[0].rgb, st[0][1] == "out")
            for r, d, s in zip(racers[1:], datas[1:], st[1:]):
                n0 = scn.ngeom
                mujoco.mjv_addGeoms(m, d, opt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
                _recolour(scn, n0, scn.ngeom, body_geom, r.rgb, s[1] == "out")
            if line is not None:
                _line(scn, line)
            for r in racers:
                _streak(scn, r, r.index(t), t)
            return rend.render()

        img = view(cam)
        # Crash replay: a racer going out off camera gets an inset chasing it, from 1 s before until 1.5 s after.
        for j, r in enumerate(racers):
            if j != lead and r.fail and -1.0 <= t - r.end_t <= 1.5 and clock >= 0:
                ri = r.index(t)
                pcam = mujoco.MjvCamera()
                pcam.lookat[:] = r.path[ri, :3]
                v = r.path[min(ri + 4, len(r.path) - 1), :3] - r.path[max(ri - 4, 0), :3]
                pcam.azimuth = np.degrees(np.arctan2(v[1], v[0])) if np.linalg.norm(v[:2]) > 0.02 else az
                pcam.distance, pcam.elevation = 3.0, -20
                inset = view(pcam)[::2, ::2]
                img = _paste_inset(img, inset, r.rgb)
                break

        # Section banner: when the camera's racer reaches a new section's first gate.
        k_lead = st[lead][0]
        sec = max((n for n, s in enumerate(sections) if s["gate"] <= k_lead), default=-1)
        if sec != banner[0] and clock >= 0:
            banner = (sec, clock)
        frames.append(_overlay(img, clock, racers, st, c, sections, banner, fonts, a.width, speed))
        speed = FAST if clock > 0 and sum(s[1] == "racing" for s in st) <= 1 and len(racers) > 1 else 1.0
        clock += speed / FPS

    for _ in range(int(2.5 * FPS)):            # results card over the last frame
        frames.append(_results(frames[-1], racers, fonts, a.width, a.height))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(a.out, frames, fps=FPS)
    print(f"{len(frames)} frames ({len(frames) / FPS:.1f} s) -> {a.out}")


def _recolour(scn, lo, hi, body_geom, rgb, out):
    rgb = (0.35, 0.35, 0.38) if out else rgb
    for g in range(lo, hi):
        geom = scn.geoms[g]
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == body_geom:
            geom.rgba[:] = (*rgb, 1.0)
            geom.emission = 0.0 if out else 0.35


def _add(scn, kind, size, pos, mat, rgba, emission=0.0):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).ravel(),
                        np.asarray(rgba, np.float32))
    g.emission = emission
    scn.ngeom += 1
    return g


def _streak(scn, r: Racer, i: int, t: float):
    """Capsules joining the last STREAK_S of the path: thin and faint at the tail, thick and bright at the drone."""
    n = int(STREAK_S / 0.01)
    lo = max(i - n, 0)
    idx = list(range(lo, i + 1, 2))
    if t >= r.end_t and not r.finished:
        return
    for a_, b_ in zip(idx, idx[1:]):
        frac = (b_ - lo) / max(i - lo, 1)
        p0, p1 = r.path[a_, :3], r.path[b_, :3]
        if np.linalg.norm(p1 - p0) < 1e-4:
            continue
        g = _add(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, [0, 0, 0], p0, np.eye(3),
                 (*r.rgb, 0.08 + 0.75 * frac ** 1.5), emission=0.8)
        if g is not None:
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004 + 0.03 * frac ** 1.2, p0, p1)


def _line(scn, pts: np.ndarray):
    """The ideal line: one thin, continuous, faint white line through the plan's points."""
    for k in range(len(pts) - 1):
        g = _add(scn, mujoco.mjtGeom.mjGEOM_CAPSULE, [0, 0, 0], pts[k], np.eye(3), (1, 1, 1, 0.3), emission=0.5)
        if g is not None:
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, pts[k], pts[k + 1])


def _paste_inset(img, inset, rgb):
    """Inset in the right half, below the leaderboard, with a border in the racer's colour."""
    out = img.copy()
    h, w = inset.shape[:2]
    x0, y0, b = img.shape[1] - w - 40, 330, 6
    col = (np.array(rgb) * 255).astype(np.uint8)
    out[y0 - b:y0 + h + b, x0 - b:x0 + w + b] = col
    out[y0:y0 + h, x0:x0 + w] = inset
    return out


def _centered(dr, y, text, font, fill, width):
    w = dr.textlength(text, font=font)
    dr.text(((width - w) / 2, y), text, font=font, fill=fill)


def _overlay(img, clock, racers, st, c, sections, banner, fonts, width, speed=1.0):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    n_g = len(c.gates)
    # Leaderboard: finished first (by time), then racing (by gates), then out.
    def rank(j):
        k, s = st[j]
        return (0, racers[j].gates[-1]) if s == "finished" else (1, -k) if s == "racing" else (2, -k)
    rows = sorted(range(len(racers)), key=rank)
    h = 96 + 64 * len(racers)
    dr.rounded_rectangle([28, 28, 28 + 640, 28 + h], 24, fill=(5, 8, 20, 170), outline=(90, 120, 200, 120), width=2)
    dr.text((56, 38), f"{max(clock, 0):5.2f} s", font=fonts[46], fill=(255, 255, 255, 255))
    for row, j in enumerate(rows):
        r, (k, s) = racers[j], st[j]
        y = 112 + 64 * row
        col = tuple(int(255 * v) for v in r.rgb)
        dr.rounded_rectangle([56, y + 10, 80, y + 40], 6, fill=(*col, 255))
        dr.text((100, y), r.name, font=fonts[38], fill=(255, 255, 255, 255))
        if s == "finished":
            txt, fill = f"FINISH {r.gates[-1]:.2f} s", (120, 255, 150, 255)
        elif s == "out":
            txt, fill = "OUT", (255, 90, 90, 255)
        else:
            txt, fill = f"{k}/{n_g} gates", (255, 220, 120, 255)
        dr.text((300, y), txt, font=fonts[38], fill=fill)

    if speed > 1:
        txt = f"FAST-FORWARD {speed:g}x"
        w = dr.textlength(txt, font=fonts[46])
        dr.rounded_rectangle([width - 60 - w - 30, 40, width - 40, 110], 20, fill=(5, 8, 20, 170))
        dr.text((width - 60 - w - 15, 48), txt, font=fonts[46], fill=(255, 255, 255, 230))

    # Countdown.
    if clock < 0:
        n = int(np.ceil(-clock / (COUNTDOWN / 3)))
        _centered(dr, 700, str(min(n, 3)), fonts[220], (255, 255, 255, 235), width)
    elif clock < 0.6:
        _centered(dr, 700, "GO!", fonts[220], (120, 255, 150, 255 - int(clock / 0.6 * 200)), width)

    # Section banner: 2.2 s, fading out.
    sec, since = banner
    age = clock - since
    if sec >= 0 and 0 <= age < 2.2 and clock > 0.3 and sec < len(sections):
        alpha = 1.0 if age < 1.6 else (2.2 - age) / 0.6
        g0 = sections[sec]["gate"]
        col = tuple(int(255 * v) for v in gate_colour(c, min(g0, n_g - 1)))
        y0 = 1480
        dr.rounded_rectangle([60, y0, width - 60, y0 + 230], 30, fill=(5, 8, 20, int(190 * alpha)),
                             outline=(*col, int(255 * alpha)), width=5)
        label = "FINAL STRETCH" if sec == len(sections) - 1 else f"SECTION {sec + 1} OF {len(sections) - 1}"
        _centered(dr, y0 + 22, label, fonts[46], (*col, int(255 * alpha)), width)
        _centered(dr, y0 + 80, sections[sec]["name"].upper(), fonts[96], (255, 255, 255, int(255 * alpha)), width)
        # difficulty pips
        pips = len(sections) - 1
        for q in range(pips):
            x = width / 2 - pips * 26 + q * 52
            on = q <= min(sec, pips - 1)
            dr.rounded_rectangle([x, y0 + 196, x + 40, y0 + 212], 6,
                                 fill=(*col, int((255 if on else 60) * alpha)))

    # A racer going out: flash its name and what it hit for 1.5 s.
    for r, (k, s) in zip(racers, st):
        if s == "out" and 0 <= clock - r.end_t < 1.5:
            col = tuple(int(255 * v) for v in r.rgb)
            _centered(dr, 1300, f"{r.name} OUT", fonts[96], (*col, 255), width)
            _centered(dr, 1410, FAIL_TEXT.get(r.fail, r.fail or ""), fonts[64], (255, 255, 255, 230), width)
    return np.asarray(im)


def _results(img, racers, fonts, width, height):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    dr.rectangle([0, 0, width, height], fill=(0, 0, 0, 140))
    _centered(dr, 520, "RESULTS", fonts[96], (255, 255, 255, 255), width)
    order = sorted(racers, key=lambda r: (not r.finished, r.gates[-1] if r.finished else -len(r.gates)))
    for n, r in enumerate(order):
        y = 700 + 150 * n
        col = tuple(int(255 * v) for v in r.rgb)
        dr.rounded_rectangle([110, y, width - 110, y + 120], 24, fill=(10, 14, 30, 220), outline=(*col, 255), width=4)
        dr.text((150, y + 26), f"{n + 1}. {r.name}", font=fonts[64], fill=(*col, 255))
        res = f"{r.gates[-1]:.2f} s" if r.finished else f"OUT ({len(r.gates)} gates)"
        w = dr.textlength(res, font=fonts[64])
        dr.text((width - 150 - w, y + 26), res, font=fonts[64], fill=(255, 255, 255, 255))
    return np.asarray(im)


if __name__ == "__main__":
    main()
