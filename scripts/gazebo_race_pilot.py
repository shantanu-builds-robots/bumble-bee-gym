#!/usr/bin/env python3
"""ROS 2 node: fly a race-gates policy trained in MuJoCo, in Gazebo, against bumble-bee's own drone plugin.

Run inside the bumble-bee container with Gazebo started paused and the drone spawned at the start line; see
scripts/run_gazebo_race.sh. It builds exactly the observation the training environment does
(bumble_bee_gym/learn/gates_env.py) from /drone_pose and /drone_velocity, runs the exported numpy policy, and sends
the four rotor speeds on /motor_speed, as bumble-bee's fly node would. Physics is the Gazebo plugin's, collisions
included: touching a gate's frame here is a real collision.

It runs like a flight computer, meant for Gazebo at full real time: a 100 Hz wall-clock timer acts on the newest
pose it has (subscriptions keep only the latest message), so a slow step never makes it fall behind; it just acts on
a slightly older pose, as real hardware would.

    gazebo_race_pilot.py <policy.npz | mpc:<plan.npz> | hover> <out> [seconds]

``hover`` sends the hover speed on every rotor instead of a policy: a timing check that flies for the given time
without needing a policy that copes.

``mpc:<plan.npz>`` flies the model-based racer instead (bumble_bee_gym/mpc): the NMPC tracking the time-optimal plan,
compiled to C at start-up while Gazebo is still paused. Its rotor thrusts become rotor speeds through the inverse of
the plugin's thrust map. It needs CasADi (run_gazebo_race.sh mounts it).

Writes <out>.csv, one row per control step:
  t        pose stamp the step acted on (sim time from the first pose) [s]
  x..qz    that pose
  target, gates, a1..a4
  age      how stale that pose was when the command went out: wall time since it arrived, plus how much later than
           the fastest-arriving pose it arrived [ms] (sim and wall run together at real time)
  wall     wall time of the command, from the first one [s]
and prints the age distribution and the real-time factor Gazebo actually ran at.
"""

import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from drone_plugin.msg import MotorSpeed
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty

from bumble_bee_gym import course as course_mod
from bumble_bee_gym import hoop, params as params_mod, parity
from bumble_bee_gym.policy_np import Policy

PROPS = ["prop_1", "prop_2", "prop_3", "prop_4"]
CTRL_DT = 0.01


def quat_to_mat(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class Pilot(Node):
    def __init__(self, policy_path: str, out: Path, seconds: float, course: str = "race_gates"):
        super().__init__("gazebo_race_pilot")
        p = params_mod.load()
        self.mpc = None
        if policy_path.startswith("mpc:"):
            from bumble_bee_gym.mpc.controller import NMPC, Plan
            from bumble_bee_gym.mpc.model import Model
            self.model = Model(p)
            self.get_logger().info("compiling the NMPC (about two minutes)...")
            self.mpc = NMPC(self.model, Plan(policy_path[4:]), jit=True)
            self.policy = None
        else:
            self.policy = None if policy_path == "hover" else Policy(policy_path)
        self.thrust_map = p.thrust_map
        self.out, self.seconds = out, seconds
        gates = course_mod.load(course).gates
        self.n = len(gates)
        frames = [hoop.hoop_frame(g) for g in gates]
        self.centres = np.stack([f[0] for f in frames])
        self.rots = np.stack([f[1] for f in frames])
        self.halves = np.array([np.array(g.opening) / 2 for g in gates])
        self.hover, self.wmax = float(p.hover_speed()), float(p.max_rotor_speed)

        latest = QoSProfile(depth=1)
        self.pub = self.create_publisher(MotorSpeed, "/motor_speed", latest)
        self.create_subscription(Twist, "/drone_velocity", self.on_vel, latest)
        self.create_subscription(PoseStamped, "/drone_pose", self.on_pose, latest)
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.vel = self.pose = None
        self.lag_min = np.inf          # smallest (arrival wall time - stamp) seen: the fastest delivery
        self.t0 = self.w0 = None
        self.target, self.gates_passed = 0, 0
        self.last_act = np.zeros(4)
        self.p_prev = None
        self.rows, self.arrivals = [], []
        self.done = False
        self.timer = None

    def start(self):
        self.timer = self.create_timer(CTRL_DT, self.on_tick)

    def on_vel(self, msg: Twist):
        self.vel = (np.array([msg.linear.x, msg.linear.y, msg.linear.z]),
                    np.array([msg.angular.x, msg.angular.y, msg.angular.z]))

    def on_pose(self, msg: PoseStamped):
        now = time.monotonic()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.lag_min = min(self.lag_min, now - stamp)
        self.pose = (stamp, now, msg.pose)
        self.arrivals.append((stamp, now))

    def obs(self, pos, R, v_world, w_world, k):
        rot = self.rots[k]
        nxt = (k + 1) % self.n
        return np.hstack([rot.T @ (pos - self.centres[k]), rot.T @ v_world, (rot.T @ R).ravel(),
                          R.T @ w_world,                       # body rates: MuJoCo's free-joint angular velocity
                          self.last_act, rot.T @ (self.centres[nxt] - self.centres[k]), rot.T @ self.rots[nxt][:, 0],
                          np.eye(self.n)[k]])

    def on_tick(self):
        if self.done or self.vel is None or self.pose is None:
            return
        stamp, arrived, pose = self.pose
        if self.t0 is None:
            self.t0 = stamp
        rel = stamp - self.t0
        P, Q = pose.position, pose.orientation
        pos = np.array([P.x, P.y, P.z])
        R = quat_to_mat(Q.w, Q.x, Q.y, Q.z)

        # Gate passage, as in training: the target's plane crossed inside its opening since the last control step.
        k = self.target
        p1 = self.rots[k].T @ (pos - self.centres[k])
        if self.p_prev is not None and self.p_prev[0] < 0.0 <= p1[0]:
            frac = -self.p_prev[0] / max(p1[0] - self.p_prev[0], 1e-6)
            yz = (self.p_prev + frac * (p1 - self.p_prev))[1:]
            if np.all(np.abs(yz) < self.halves[k]):
                self.gates_passed += 1
                self.target = (k + 1) % self.n
                self.get_logger().info(f"t={rel:5.2f}s  gate {k} passed ({self.gates_passed} total)")
            else:
                self.get_logger().info(f"t={rel:5.2f}s  MISSED gate {k} (offset {yz.round(2)})")
        k = self.target
        self.p_prev = self.rots[k].T @ (pos - self.centres[k])

        if self.mpc is not None:
            w_body = R.T @ self.vel[1]
            com = R @ self.model.com
            x = np.concatenate([pos + com, self.vel[0] + np.cross(self.vel[1], com), [Q.w, Q.x, Q.y, Q.z], w_body])
            speeds = self.speeds_for(self.mpc(x, rel))
            a = (speeds - self.hover) / hoop.SPEED_SPAN       # logged in the policy's units
        else:
            a = np.zeros(4) if self.policy is None else self.policy(self.obs(pos, R, self.vel[0], self.vel[1], k))
            speeds = np.clip(self.hover + hoop.SPEED_SPAN * np.clip(a, -1, 1), 0.0, self.wmax)
        out = MotorSpeed()
        out.name = PROPS
        out.velocity = [float(v) for v in parity.SPIN_SIGN * speeds]
        self.pub.publish(out)
        sent = time.monotonic()
        if self.w0 is None:
            self.w0 = sent
        age = (sent - arrived) + (arrived - stamp - self.lag_min)
        self.last_act = a
        self.rows.append((rel, *pos, Q.w, Q.x, Q.y, Q.z, k, self.gates_passed, *a, age * 1e3, sent - self.w0))

        crashed = pos[2] < 0.1 or R[2, 2] < 0.0
        if rel >= self.seconds or crashed:
            self.done = True
            self.get_logger().info(f"end at t={rel:.2f}s: {self.gates_passed} gates, "
                                   f"{'CRASHED' if crashed else 'no crash'}")
            self.pause.call_async(Empty.Request())
            self.report()

    def speeds_for(self, thrust: np.ndarray) -> np.ndarray:
        """Rotor speeds [rad/s] giving these thrusts [N] under the plugin's map f = c2 w^2 + c1 w + c0 (zero thrust ->
        the speed where the map reaches zero)."""
        c2, c1, c0 = self.thrust_map
        f = np.maximum(np.asarray(thrust, float), 0.0)
        return np.clip((-c1 + np.sqrt(c1 * c1 - 4 * c2 * (c0 - f))) / (2 * c2), 0.0, self.wmax)

    def report(self):
        rows = np.array(self.rows)
        np.savetxt(f"{self.out}.csv", rows, delimiter=",", fmt="%.6f",
                   header="t,x,y,z,qw,qx,qy,qz,target,gates,a1,a2,a3,a4,age,wall", comments="")
        age = rows[:, 14]
        arr = np.array(self.arrivals)
        rtf = (arr[-1, 0] - arr[0, 0]) / max(arr[-1, 1] - arr[0, 1], 1e-9)
        steps = np.diff(rows[:, 0]) * 1e3
        if self.mpc is not None:
            st = np.array(self.mpc.solve_times) * 1e3
            self.get_logger().info(f"NMPC solve ms: median {np.median(st):.2f}  p95 {np.percentile(st, 95):.2f}  "
                                   f"max {st.max():.2f}")
        self.get_logger().info(
            f"timing over {len(rows)} steps: pose age ms  median {np.median(age):.2f}  p90 {np.percentile(age, 90):.2f}"
            f"  p99 {np.percentile(age, 99):.2f}  max {age.max():.2f}  |  sim ms between steps  median "
            f"{np.median(steps):.1f}  min {steps.min():.1f}  max {steps.max():.1f}  |  real-time factor {rtf:.3f}")


def main():
    policy_path, out, seconds = sys.argv[1], Path(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    rclpy.init()
    node = Pilot(policy_path, out, seconds)
    node.unpause.wait_for_service()
    node.unpause.call_async(Empty.Request())
    node.start()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.5)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
