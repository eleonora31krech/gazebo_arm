#!/usr/bin/env python3
import math
import time
from typing import Dict, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64



def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ], dtype=float)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ], dtype=float)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=float)


def rpy_to_rot(r: float, p: float, y: float) -> np.ndarray:
    return rot_z(y) @ rot_y(p) @ rot_x(r)


def homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c

    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ], dtype=float)



class PandaIK:
    def __init__(self, base_world=(0.20, 0.0, 0.7)):
        self.base_world = np.array(base_world, dtype=float)

        self.q_min = np.array([
            -2.9671,
            -1.8326,
            -2.9671,
            -3.1416,
            -2.9671,
            -0.0873,
            -2.9671,
        ], dtype=float)

        self.q_max = np.array([
             2.9671,
             1.8326,
             2.9671,
             0.0,
             2.9671,
             3.8223,
             2.9671,
        ], dtype=float)

        self.q_nominal = np.array([
            0.0, -0.35, 0.0, -2.20, 0.0, 2.00, 0.785
        ], dtype=float)

        self.joints = [
            {
                "xyz": np.array([0.0, 0.0, 0.333]),
                "rpy": (0.0, 0.0, 0.0),
                "axis": np.array([0.0, 0.0, 1.0]),
            },
            {
                "xyz": np.array([0.0, 0.0, 0.0]),
                "rpy": (-math.pi / 2.0, 0.0, 0.0),
                "axis": np.array([0.0, 0.0, 1.0]),
            },
            {
                "xyz": np.array([0.0, -0.316, 0.0]),
                "rpy": (math.pi / 2.0, 0.0, 0.0),
                "axis": np.array([0.0, 0.0, 1.0]),
            },
            {
                "xyz": np.array([0.0825, 0.0, 0.0]),
                "rpy": (math.pi / 2.0, 0.0, 0.0),
                "axis": np.array([0.0, 0.0, 1.0]),
            },
            {
                "xyz": np.array([-0.0825, 0.384, 0.0]),
                "rpy": (-math.pi / 2.0, 0.0, 0.0),
                "axis": np.array([0.0, 0.0, 1.0]),
            },
            {
                "xyz": np.array([0.0, 0.0, 0.0]),
                "rpy": (math.pi / 2.0, 0.0, 0.0),
                "axis": np.array([0.0, 0.0, 1.0]),
            },
            {
                "xyz": np.array([0.088, 0.0, 0.0]),
                "rpy": (math.pi / 2.0, 0.0, 0.0),
                "axis": np.array([0.0, 0.0, 1.0]),
            },
        ]

        self.T_joint8 = homogeneous(np.eye(3), np.array([0.0, 0.0, 0.107]))
        self.T_hand = homogeneous(rpy_to_rot(0.0, 0.0, -math.pi / 4.0), np.zeros(3))

        # TCP offset from panda_hand frame
        self.tool_offset = np.array([0.0, 0.0, 0.12], dtype=float)

    def clamp_q(self, q: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(q, self.q_min), self.q_max)

    def fk(self, q: np.ndarray) -> np.ndarray:

        T = np.eye(4, dtype=float)

        for i in range(7):
            joint = self.joints[i]

            T_origin = homogeneous(
                rpy_to_rot(*joint["rpy"]),
                joint["xyz"]
            )

            T_rot = homogeneous(
                rot_axis_angle(joint["axis"], q[i]),
                np.zeros(3)
            )

            T = T @ T_origin @ T_rot

        T = T @ self.T_joint8 @ self.T_hand
        T_tool = homogeneous(np.eye(3), self.tool_offset)
        T = T @ T_tool

        return T

    def ee_position(self, q: np.ndarray) -> np.ndarray:
        return self.fk(q)[:3, 3]

    def numerical_jacobian(self, q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        J = np.zeros((3, 7), dtype=float)
        p0 = self.ee_position(q)

        for i in range(7):
            q2 = q.copy()
            q2[i] += eps
            p1 = self.ee_position(q2)
            J[:, i] = (p1 - p0) / eps

        return J

    def world_to_base(self, target_world: Tuple[float, float, float]) -> np.ndarray:
        return np.array(target_world, dtype=float) - self.base_world

    def solve_position_ik(
        self,
        target_world: Tuple[float, float, float],
        q_seed: np.ndarray = None,
        max_iters: int = 250,
        tol: float = 0.003,
        damping: float = 0.08,
        step_scale: float = 0.7,
        nullspace_gain: float = 0.05,
    ):

        target_base = self.world_to_base(target_world)

        if q_seed is None:
            q = self.q_nominal.copy()
        else:
            q = np.array(q_seed, dtype=float).copy()

        q = self.clamp_q(q)

        for _ in range(max_iters):
            p = self.ee_position(q)
            e = target_base - p
            err = np.linalg.norm(e)

            if err < tol:
                return True, q, err

            J = self.numerical_jacobian(q)

            A = J @ J.T + (damping ** 2) * np.eye(3)
            dq_main = J.T @ np.linalg.solve(A, e)

            J_pinv = J.T @ np.linalg.solve(A, np.eye(3))
            N = np.eye(7) - J_pinv @ J
            dq_null = nullspace_gain * (N @ (self.q_nominal - q))

            dq = step_scale * (dq_main + dq_null)
            dq = np.clip(dq, -0.08, 0.08)

            q = self.clamp_q(q + dq)

        final_err = np.linalg.norm(target_base - self.ee_position(q))
        return False, q, final_err


class PandaPickPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_place_controller')

        self.joint_pubs = {}
        for i in range(1, 8):
            self.joint_pubs[i] = self.create_publisher(
                Float64, f'/panda_joint{i}_cmd', 10
            )

        self.gripper_left_pub = self.create_publisher(
            Float64, '/panda_finger_joint1_cmd', 10
        )
        self.gripper_right_pub = self.create_publisher(
            Float64, '/panda_finger_joint2_cmd', 10
        )

        self.current_pose = {
            1: 0.0,
            2: -0.35,
            3: 0.0,
            4: -2.20,
            5: 0.0,
            6: 2.00,
            7: 0.785,
        }
        self.current_gripper = 0.04

        self.ik = PandaIK(base_world=(0.20, 0.0, 0.7))

        self.get_logger().info('Panda IK pick-place node started')
        time.sleep(3.0)

        self.run_sequence()

    def publish_joint_pose(self, pose: Dict[int, float]):
        for joint_id, value in pose.items():
            msg = Float64()
            msg.data = float(value)
            self.joint_pubs[joint_id].publish(msg)

    def publish_gripper(self, width: float):
        width = float(max(0.0, min(0.04, width)))

        msg_left = Float64()
        msg_left.data = width
        self.gripper_left_pub.publish(msg_left)

        msg_right = Float64()
        msg_right.data = width
        self.gripper_right_pub.publish(msg_right)


    def move_joints_smooth(
        self,
        target_pose: Dict[int, float],
        duration: float = 3.0,
        rate_hz: float = 50.0
    ):
        steps = max(2, int(duration * rate_hz))
        start_pose = self.current_pose.copy()

        self.get_logger().info(
            'Move -> ' +
            ', '.join([f'J{k}={math.degrees(v):.1f}°' for k, v in sorted(target_pose.items())])
        )

        for step in range(steps + 1):
            alpha = step / steps
            s = 3.0 * alpha**2 - 2.0 * alpha**3  # smoothstep

            cmd = {}
            for j in range(1, 8):
                q0 = start_pose[j]
                q1 = target_pose[j]
                cmd[j] = q0 + (q1 - q0) * s

            self.publish_joint_pose(cmd)
            time.sleep(1.0 / rate_hz)

        self.current_pose = target_pose.copy()

    def set_gripper_smooth(
        self,
        target_width: float,
        duration: float = 1.2,
        rate_hz: float = 30.0
    ):
        steps = max(2, int(duration * rate_hz))
        start_width = self.current_gripper
        target_width = float(max(0.0, min(0.04, target_width)))

        self.get_logger().info(f'Gripper -> {target_width:.3f} m')

        for step in range(steps + 1):
            alpha = step / steps
            s = 3.0 * alpha**2 - 2.0 * alpha**3
            width = start_width + (target_width - start_width) * s
            self.publish_gripper(width)
            time.sleep(1.0 / rate_hz)

        self.current_gripper = target_width

    def hold_pose(self, duration: float = 0.5, rate_hz: float = 20.0):
        steps = max(1, int(duration * rate_hz))
        for _ in range(steps):
            self.publish_joint_pose(self.current_pose)
            self.publish_gripper(self.current_gripper)
            time.sleep(1.0 / rate_hz)

    def solve_world_target(self, target_world, q_seed=None) -> Dict[int, float]:
        if q_seed is None:
            q_seed = np.array([self.current_pose[i] for i in range(1, 8)], dtype=float)

        success, q_sol, err = self.ik.solve_position_ik(
            target_world=target_world,
            q_seed=q_seed,
            max_iters=250,
            tol=0.003,
            damping=0.08,
            step_scale=0.7,
            nullspace_gain=0.05,
        )

        self.get_logger().info(
            f'IK target {target_world} -> success={success}, err={err:.4f} m'
        )

        pose = {i + 1: float(q_sol[i]) for i in range(7)}
        return pose


    def run_sequence(self):
        GREEN_X, GREEN_Y, GREEN_Z = 0.65, 0.10, 0.82
        # GREEN_X, GREEN_Y, GREEN_Z = 0.65, 0.10, 0.82
        GREEN_X, GREEN_Y, GREEN_Z = 0.65, -0.2 ,0.82

        HOME = {
            1: 0.0,
            2: -0.35,
            3: 0.0,
            4: -2.20,
            5: 0.0,
            6: 2.00,
            7: 0.785,
        }

        ABOVE_OFFSET = 0.12
        APPROACH_OFFSET = 0.07
        GRASP_OFFSET = 0.00

        self.get_logger().info('--- START PICK SEQUENCE ---')

        self.publish_joint_pose(self.current_pose)
        self.publish_gripper(self.current_gripper)
        self.hold_pose(1.0)

        self.set_gripper_smooth(0.04, duration=1.0)
        self.hold_pose(0.5)

        self.move_joints_smooth(HOME, duration=3.0)
        self.hold_pose(0.8)

        q_home = np.array([HOME[i] for i in range(1, 8)], dtype=float)

        above_pose = self.solve_world_target(
            (GREEN_X, GREEN_Y, GREEN_Z + ABOVE_OFFSET),
            q_seed=q_home
        )
        q_above = np.array([above_pose[i] for i in range(1, 8)], dtype=float)

        approach_pose = self.solve_world_target(
            (GREEN_X, GREEN_Y, GREEN_Z + APPROACH_OFFSET),
            q_seed=q_above
        )
        q_approach = np.array([approach_pose[i] for i in range(1, 8)], dtype=float)

        grasp_pose = self.solve_world_target(
            (GREEN_X, GREEN_Y, GREEN_Z + GRASP_OFFSET),
            q_seed=q_approach
        )

        self.move_joints_smooth(above_pose, duration=4.0)
        self.hold_pose(0.8)

        self.move_joints_smooth(approach_pose, duration=2.0)
        self.hold_pose(0.8)

        self.move_joints_smooth(grasp_pose, duration=1.6)
        self.hold_pose(0.8)

        # self.set_gripper_smooth(0.006, duration=1.2)
        # self.hold_pose(1.0)
        self.set_gripper_smooth(0.014, duration=2.5)
        self.hold_pose(2.0)
        self.move_joints_smooth(approach_pose, duration=1.5)
        self.hold_pose(0.5)

        self.move_joints_smooth(above_pose, duration=2.0)
        self.hold_pose(0.8)

        self.move_joints_smooth(HOME, duration=3.0)
        self.hold_pose(1.0)

        self.get_logger().info('--- DONE ---')


def main(args=None):
    rclpy.init(args=args)
    node = PandaPickPlaceNode()

    try:
        rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()