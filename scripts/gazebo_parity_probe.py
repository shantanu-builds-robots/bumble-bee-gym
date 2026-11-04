#!/usr/bin/env python3
"""ROS 2 node: fly the parity profile open-loop in Gazebo and record the drone's pose at every physics step.

Run inside the bumble-bee container with Gazebo started paused (pause:=true) and the drone spawned
mid-air in the empty course; see scripts/run_gazebo_parity.sh. Writes <out>.csv (pose per step) and
<out>.switch.txt (sim time each profile segment took effect, relative to the start).
"""

import sys
from pathlib import Path

import numpy as np
import rclpy
from drone_plugin.msg import MotorSpeed
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_srvs.srv import Empty

from bumble_bee_gym import parity

PROPS = ["prop_1", "prop_2", "prop_3", "prop_4"]


class Probe(Node):
    def __init__(self, out: Path):
        super().__init__("gazebo_parity_probe")
        self.out = out
        self.pub = self.create_publisher(MotorSpeed, "/motor_speed", 10)
        self.create_subscription(PoseStamped, "/drone_pose", self.on_pose, 1000)
        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.t0 = None
        self.segment = 0
        self.switch = [0.0]
        self.rows = []
        self.done = False

    def send(self, speeds):
        msg = MotorSpeed()
        msg.name = PROPS
        msg.velocity = [float(v) for v in parity.SPIN_SIGN * speeds]
        self.pub.publish(msg)

    def on_pose(self, msg: PoseStamped):
        if self.done:
            return
        # The plugin stamps each pose with the sim time of that state and publishes it before applying
        # thrust for the step, so the first message is the untouched spawn state.
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t
        rel = round(t - self.t0, 6)
        p, q = msg.pose.position, msg.pose.orientation
        self.rows.append((rel, p.x, p.y, p.z, q.w, q.x, q.y, q.z))
        # A command sent now is in place for the step that starts at rel + dt: physics runs at a tenth of
        # real time here (10 ms of wall time per step), far longer than ROS needs to deliver it.
        nxt = self.segment + 1
        if nxt < len(parity.PROFILE) and rel + 0.001 >= parity.PROFILE[nxt][0] - 1e-9:
            self.segment = nxt
            self.send(np.array(parity.PROFILE[nxt][1]))
            self.switch.append(rel + 0.001)
        if rel >= parity.DURATION:
            self.done = True
            self.pause.call_async(Empty.Request())
            self.write()

    def write(self):
        with open(self.out.with_suffix(".csv"), "w") as f:
            f.write("t,x,y,z,qw,qx,qy,qz\n")
            for r in self.rows:
                f.write(",".join(f"{v:.9g}" for v in r) + "\n")
        self.out.with_suffix(".switch.txt").write_text("\n".join(f"{s:.6f}" for s in self.switch) + "\n")
        self.get_logger().info(f"wrote {len(self.rows)} samples to {self.out.with_suffix('.csv')}")


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "gazebo_parity")
    rclpy.init()
    node = Probe(out)
    for client in (node.unpause, node.pause):
        if not client.wait_for_service(timeout_sec=60.0):
            node.get_logger().error(f"{client.srv_name} not available")
            sys.exit(1)
    # Hover command first, while physics is still paused, so it is in place at the first step.
    for _ in range(20):
        node.send(np.array(parity.PROFILE[0][1]))
        rclpy.spin_once(node, timeout_sec=0.05)
    node.unpause.call_async(Empty.Request())
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=1.0)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
