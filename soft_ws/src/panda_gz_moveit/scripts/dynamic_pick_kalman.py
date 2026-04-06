#!/usr/bin/env python3
"""
Panda pick-and-place controller with:
- full 6-DOF IK (position + orientation)
- Kalman filter for moving cube prediction
- automatic motion time estimation
- automatic grasp intercept prediction
"""

import math
import time
from typing import Dict, List, Optional, Tuple

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
    axis = np.asarray(axis, dtype=float)
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


def rot_to_axis_angle(R: np.ndarray) -> np.ndarray:
    val = (np.trace(R) - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    angle = math.acos(val)

    if abs(angle) < 1e-8:
        return np.zeros(3, dtype=float)

    if abs(angle - math.pi) < 1e-6:
        RpI = R + np.eye(3)
        col = np.argmax(np.linalg.norm(RpI, axis=0))
        axis = RpI[:, col]
        axis = axis / np.linalg.norm(axis)
        return axis * angle

    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=float)
    axis = axis / (2.0 * math.sin(angle))
    return axis * angle


def orientation_error(R_desired: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    R_err = R_desired @ R_current.T
    return rot_to_axis_angle(R_err)


class PandaIK6DOF:
    def __init__(self, base_world=(0.20, 0.0, 0.7)):
        self.base_world = np.array(base_world, dtype=float)

        self.q_min = np.array([
            -2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671
        ], dtype=float)
        self.q_max = np.array([
             2.9671,  1.8326,  2.9671,  0.0,     2.9671,  3.8223,  2.9671
        ], dtype=float)
        self.q_nominal = np.array([
            0.0, -0.35, 0.0, -2.20, 0.0, 2.00, 0.785
        ], dtype=float)

        self.joints = [
            {"xyz": np.array([0.0, 0.0, 0.333]),    "rpy": (0.0, 0.0, 0.0),        "axis": np.array([0.0, 0.0, 1.0])},
            {"xyz": np.array([0.0, 0.0, 0.0]),      "rpy": (-math.pi/2, 0.0, 0.0), "axis": np.array([0.0, 0.0, 1.0])},
            {"xyz": np.array([0.0, -0.316, 0.0]),   "rpy": ( math.pi/2, 0.0, 0.0), "axis": np.array([0.0, 0.0, 1.0])},
            {"xyz": np.array([0.0825, 0.0, 0.0]),   "rpy": ( math.pi/2, 0.0, 0.0), "axis": np.array([0.0, 0.0, 1.0])},
            {"xyz": np.array([-0.0825, 0.384, 0.0]),"rpy": (-math.pi/2, 0.0, 0.0), "axis": np.array([0.0, 0.0, 1.0])},
            {"xyz": np.array([0.0, 0.0, 0.0]),      "rpy": ( math.pi/2, 0.0, 0.0), "axis": np.array([0.0, 0.0, 1.0])},
            {"xyz": np.array([0.088, 0.0, 0.0]),    "rpy": ( math.pi/2, 0.0, 0.0), "axis": np.array([0.0, 0.0, 1.0])},
        ]

        self.T_joint8 = homogeneous(np.eye(3), np.array([0.0, 0.0, 0.107]))
        self.T_hand = homogeneous(rpy_to_rot(0.0, 0.0, -math.pi / 4.0), np.zeros(3))
        self.tool_offset = np.array([0.0, 0.0, 0.12], dtype=float)

    def clamp_q(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.q_min, self.q_max)

    def fk(self, q: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=float)
        for i in range(7):
            joint = self.joints[i]
            T_origin = homogeneous(rpy_to_rot(*joint["rpy"]), joint["xyz"])
            T_rot = homogeneous(rot_axis_angle(joint["axis"], q[i]), np.zeros(3))
            T = T @ T_origin @ T_rot
        T = T @ self.T_joint8 @ self.T_hand
        T = T @ homogeneous(np.eye(3), self.tool_offset)
        return T

    def ee_pose(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = self.fk(q)
        return T[:3, 3].copy(), T[:3, :3].copy()

    def numerical_jacobian_6dof(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        J = np.zeros((6, 7), dtype=float)
        p0, R0 = self.ee_pose(q)
        for i in range(7):
            qp = q.copy()
            qp[i] += eps
            p1, R1 = self.ee_pose(qp)
            J[:3, i] = (p1 - p0) / eps
            dR = R1 @ R0.T
            aa = rot_to_axis_angle(dR)
            J[3:, i] = aa / eps
        return J

    def world_to_base(self, target_world) -> np.ndarray:
        return np.asarray(target_world, dtype=float) - self.base_world

    def make_top_down_orientation(self, yaw: float = 0.0) -> np.ndarray:
        cz = np.array([0.0, 0.0, -1.0], dtype=float)
        cx = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)
        cy = np.cross(cz, cx)
        cy = cy / np.linalg.norm(cy)
        cx = np.cross(cy, cz)
        cx = cx / np.linalg.norm(cx)
        R = np.column_stack([cx, cy, cz])
        return R

    def solve_6dof(
        self,
        target_pos_world: Tuple[float, float, float],
        target_rot_base: np.ndarray,
        q_seed: Optional[np.ndarray] = None,
        max_iters: int = 400,
        pos_tol: float = 0.002,
        orient_tol: float = 0.03,
        damping: float = 0.05,
        step_scale: float = 0.5,
        nullspace_gain: float = 0.02,
        pos_weight: float = 1.0,
        orient_weight: float = 0.3,
    ):
        target_pos_base = self.world_to_base(target_pos_world)

        if q_seed is None:
            q = self.q_nominal.copy()
        else:
            q = np.array(q_seed, dtype=float).copy()
        q = self.clamp_q(q)

        W = np.diag([pos_weight] * 3 + [orient_weight] * 3)

        for _ in range(max_iters):
            p, R = self.ee_pose(q)

            e_pos = target_pos_base - p
            err_pos = np.linalg.norm(e_pos)

            e_orient = orientation_error(target_rot_base, R)
            err_orient = np.linalg.norm(e_orient)

            if err_pos < pos_tol and err_orient < orient_tol:
                return True, q, err_pos, err_orient

            e6 = np.concatenate([e_pos, e_orient])
            e6w = W @ e6

            J = self.numerical_jacobian_6dof(q)
            Jw = W @ J

            A = Jw @ Jw.T + (damping ** 2) * np.eye(6)
            dq_main = Jw.T @ np.linalg.solve(A, e6w)

            J_pinv = Jw.T @ np.linalg.solve(A, np.eye(6))
            N = np.eye(7) - J_pinv @ Jw
            dq_null = nullspace_gain * (N @ (self.q_nominal - q))

            dq = step_scale * (dq_main + dq_null)
            dq = np.clip(dq, -0.1, 0.1)

            q = self.clamp_q(q + dq)

        p_final, R_final = self.ee_pose(q)
        err_pos = np.linalg.norm(target_pos_base - p_final)
        err_orient = np.linalg.norm(orientation_error(target_rot_base, R_final))
        return False, q, err_pos, err_orient


class CubeKalmanFilter:
    def __init__(self):
        self.x = np.zeros((4, 1), dtype=float)
        self.P = np.eye(4, dtype=float) * 0.5
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=float)
        self.R = np.array([
            [0.0004, 0.0],
            [0.0,    0.0004]
        ], dtype=float)
        self.Q_base = np.array([
            [1e-5, 0.0,  0.0,  0.0],
            [0.0,  1e-5, 0.0,  0.0],
            [0.0,  0.0,  5e-4, 0.0],
            [0.0,  0.0,  0.0,  5e-4]
        ], dtype=float)
        self.initialized = False

    def init_state(self, x: float, y: float, vx: float, vy: float):
        self.x = np.array([[x], [y], [vx], [vy]], dtype=float)
        self.initialized = True

    def predict(self, dt: float):
        if not self.initialized:
            return
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q_base

    def update(self, meas_x: float, meas_y: float,
               known_vx: float = None, known_vy: float = None):
        if not self.initialized:
            vx0 = 0.0 if known_vx is None else known_vx
            vy0 = 0.0 if known_vy is None else known_vy
            self.init_state(meas_x, meas_y, vx0, vy0)
            return

        z = np.array([[meas_x], [meas_y]], dtype=float)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        I = np.eye(4, dtype=float)
        self.P = (I - K @ self.H) @ self.P

        if known_vx is not None:
            self.x[2, 0] = 0.8 * self.x[2, 0] + 0.2 * known_vx
        if known_vy is not None:
            self.x[3, 0] = 0.8 * self.x[3, 0] + 0.2 * known_vy

    def get_state(self):
        return self.x.flatten()

    def predict_future_position(self, dt_future: float):
        x, y, vx, vy = self.get_state()
        return x + vx * dt_future, y + vy * dt_future

class CubeTarget:
    def __init__(self, name: str, init_x: float, init_y: float, z: float):
        self.name = name
        self.init_x = init_x
        self.init_y = init_y
        self.z = z  # cube center Z in world


class PandaPickPlace6DOF(Node):
    def __init__(self):
        super().__init__('pick_place_6dof_kalman')

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
            1: 0.0, 2: -0.35, 3: 0.0, 4: -2.20,
            5: 0.0, 6: 2.00, 7: 0.785
        }
        self.current_gripper = 0.04

        self.ik = PandaIK6DOF(base_world=(0.20, 0.0, 0.7))

        self.CUBE_CENTER_Z = 0.82

        self.conveyor_vx = 0.00
        self.conveyor_vy = 0.013

        self.y_lead_offset = 0.0

        self.conveyor_start_delay = 5.0
        self.sequence_start_time = time.time()

        self.kf = CubeKalmanFilter()
        self.last_tracker_time = None

        self.current_cube: Optional[CubeTarget] = None

        self.cubes: List[CubeTarget] = [
            CubeTarget("green_cube", init_x=0.65, init_y=0.1,  z=self.CUBE_CENTER_Z),
            # CubeTarget("red_cube",   init_x=0.65, init_y=-0.5, z=self.CUBE_CENTER_Z),
            # CubeTarget("blue_cube",  init_x=0.65, init_y=-0.9, z=self.CUBE_CENTER_Z),
        ]

        self.get_logger().info('6-DOF pick-place with Kalman started')
        time.sleep(3.0)
        self.run_sequence()

    def init_cube_tracker(self, cube: CubeTarget):
        self.current_cube = cube
        self.kf = CubeKalmanFilter()
        self.last_tracker_time = None
        self.kf.init_state(cube.init_x, cube.init_y, 0.0, 0.0)
        self.get_logger().info(
            f'Tracking cube "{cube.name}" at ({cube.init_x}, {cube.init_y}, {cube.z})'
        )

    def publish_joint_pose(self, pose: Dict[int, float]):
        for jid, val in pose.items():
            msg = Float64()
            msg.data = float(val)
            self.joint_pubs[jid].publish(msg)

    def publish_gripper(self, width: float):
        width = float(np.clip(width, 0.0, 0.04))
        for pub in [self.gripper_left_pub, self.gripper_right_pub]:
            msg = Float64()
            msg.data = width
            pub.publish(msg)


    def move_joints_smooth(self, target: Dict[int, float],
                           duration: float = 3.0, hz: float = 50.0):
        steps = max(2, int(duration * hz))
        start = self.current_pose.copy()

        self.get_logger().info(
            'Move -> ' + ', '.join(
                f'J{k}={math.degrees(v):.1f}°'
                for k, v in sorted(target.items())
            )
        )

        for step in range(steps + 1):
            a = step / steps
            s = 3.0 * a * a - 2.0 * a * a * a
            cmd = {j: start[j] + (target[j] - start[j]) * s for j in range(1, 8)}
            self.publish_joint_pose(cmd)
            time.sleep(1.0 / hz)

        self.current_pose = target.copy()

    def set_gripper_smooth(self, target_w: float,
                           duration: float = 1.2, hz: float = 30.0):
        steps = max(2, int(duration * hz))
        start_w = self.current_gripper
        target_w = float(np.clip(target_w, 0.0, 0.04))

        self.get_logger().info(f'Gripper -> {target_w:.3f} m')

        for step in range(steps + 1):
            a = step / steps
            s = 3.0 * a * a - 2.0 * a * a * a
            self.publish_gripper(start_w + (target_w - start_w) * s)
            time.sleep(1.0 / hz)

        self.current_gripper = target_w

    def hold_pose(self, duration: float = 0.5, hz: float = 20.0):
        for _ in range(max(1, int(duration * hz))):
            self.publish_joint_pose(self.current_pose)
            self.publish_gripper(self.current_gripper)
            time.sleep(1.0 / hz)

    def move_tracking(
        self,
        z_world: float,
        R_target: np.ndarray,
        duration: float = 2.0,
        hz: float = 20.0,
        ik_every_n: int = 5,
        tight_ik: bool = False,
    ):
        steps = max(2, int(duration * hz))
        start_q = np.array(
            [self.current_pose[i] for i in range(1, 8)], dtype=float
        )

        cube = self.current_cube

        self.update_tracker()
        cy = self.kf.get_state()[1]
        vy = self.kf.get_state()[3]
        pred_y = cy + vy * duration + self.y_lead_offset
        target_pos = (cube.init_x, pred_y, z_world)

        target_pose, _ = self.solve_6dof_target(
            target_pos, R_target, q_seed=start_q, tight=tight_ik
        )
        target_q = np.array(
            [target_pose[i] for i in range(1, 8)], dtype=float
        )

        self.get_logger().info(
            f'Tracking move z={z_world:.3f}, pred_y={pred_y:.3f}, dur={duration:.1f}s'
        )

        for step in range(steps + 1):
            a = step / steps
            s = 3.0 * a * a - 2.0 * a * a * a

            if step > 0 and step % ik_every_n == 0 and step < steps - 1:
                self.update_tracker()
                cy = self.kf.get_state()[1]
                vy = self.kf.get_state()[3]
                t_left = (steps - step) / hz
                new_pred_y = cy + vy * t_left + self.y_lead_offset

                new_pos = (cube.init_x, new_pred_y, z_world)
                seed_q = start_q + (target_q - start_q) * s
                new_pose, ok = self.solve_6dof_target(
                    new_pos, R_target, q_seed=seed_q, tight=tight_ik
                )
                if ok:
                    target_q = np.array(
                        [new_pose[i] for i in range(1, 8)], dtype=float
                    )
                    target_pose = new_pose

            cmd_q = start_q + (target_q - start_q) * s
            cmd = {i + 1: float(cmd_q[i]) for i in range(7)}
            self.publish_joint_pose(cmd)
            time.sleep(1.0 / hz)

        self.current_pose = target_pose.copy()

        fq = np.array([self.current_pose[i] for i in range(1, 8)], dtype=float)
        T = self.ik.fk(fq)
        pw = T[:3, 3] + self.ik.base_world
        self.get_logger().info(
            f'Tracking done -> ({pw[0]:.3f}, {pw[1]:.3f}, {pw[2]:.3f})'
        )


    def estimate_motion_time(
        self,
        start_pose: Dict[int, float],
        target_pose: Dict[int, float],
        joint_speed_limits: Optional[Dict[int, float]] = None,
        safety_factor: float = 1.5,
        min_time: float = 1.0
    ) -> float:
        if joint_speed_limits is None:
            joint_speed_limits = {
                1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8,
                5: 1.0, 6: 1.0, 7: 1.0,
            }
        times = []
        for j in range(1, 8):
            dq = abs(target_pose[j] - start_pose[j])
            vmax = max(joint_speed_limits[j], 1e-6)
            times.append(dq / vmax)

        return max(min_time, max(times) * safety_factor)

    def estimate_gripper_time(
        self,
        current_width: float,
        target_width: float,
        finger_speed: float = 0.02,
        safety_factor: float = 1.2,
        min_time: float = 0.5
    ) -> float:
        dw = abs(target_width - current_width)
        return max(min_time, (dw / max(finger_speed, 1e-6)) * safety_factor)

    def update_tracker(self):
        now = time.time()

        if self.last_tracker_time is None:
            self.last_tracker_time = now

        dt = now - self.last_tracker_time
        self.last_tracker_time = now

        self.kf.predict(dt)

        cube = self.current_cube
        t_total = now - self.sequence_start_time
        t_moving = max(0.0, t_total - self.conveyor_start_delay)

        meas_x = cube.init_x + self.conveyor_vx * t_moving
        meas_y = cube.init_y + self.conveyor_vy * t_moving

        if t_total < self.conveyor_start_delay:
            cur_vx, cur_vy = 0.0, 0.0
        else:
            cur_vx, cur_vy = self.conveyor_vx, self.conveyor_vy

        self.kf.update(meas_x, meas_y, known_vx=cur_vx, known_vy=cur_vy)

        x, y, vx, vy = self.kf.get_state()
        self.get_logger().info(
            f'KF [{cube.name}]: x={x:.3f}, y={y:.3f}, vx={vx:.3f}, vy={vy:.3f} '
            f'(t={t_total:.1f}s, moving={t_moving:.1f}s)'
        )

    def predict_cube_at(self, dt_future: float) -> Tuple[float, float]:
        return self.kf.predict_future_position(dt_future)


    def solve_6dof_target(
        self,
        pos_world: Tuple[float, float, float],
        R_target: np.ndarray,
        q_seed: Optional[np.ndarray] = None,
        tight: bool = False,
    ) -> Tuple[Dict[int, float], bool]:
        if q_seed is None:
            q_seed = np.array(
                [self.current_pose[i] for i in range(1, 8)], dtype=float
            )

        pos_tol = 0.001 if tight else 0.002
        max_iters = 600 if tight else 400

        ok, q_sol, e_pos, e_orient = self.ik.solve_6dof(
            target_pos_world=pos_world,
            target_rot_base=R_target,
            q_seed=q_seed,
            max_iters=max_iters,
            pos_tol=pos_tol,
            orient_tol=0.03,
            damping=0.05,
            step_scale=0.5,
            pos_weight=1.0,
            orient_weight=0.3,
        )

        self.get_logger().info(
            f'IK target={pos_world} -> ok={ok}, '
            f'e_pos={e_pos:.4f}m, e_orient={math.degrees(e_orient):.2f}°'
        )

        pose = {i + 1: float(q_sol[i]) for i in range(7)}
        return pose, ok

    def verify_fk(self, pose: Dict[int, float], label: str = ""):
        q = np.array([pose[i] for i in range(1, 8)], dtype=float)
        T = self.ik.fk(q)
        p_world = T[:3, 3] + self.ik.base_world
        z_axis = T[:3, 2]
        self.get_logger().info(
            f'FK [{label}] world=({p_world[0]:.3f}, {p_world[1]:.3f}, '
            f'{p_world[2]:.3f}), TCP_z=({z_axis[0]:.2f}, {z_axis[1]:.2f}, '
            f'{z_axis[2]:.2f})'
        )

    def plan_intercept(
        self,
        from_pose: Dict[int, float],
        z_offset: float,
        R_target: np.ndarray,
        q_seed: Optional[np.ndarray] = None,
        iterations: int = 4
    ) -> Tuple[Dict[int, float], float, bool, Tuple[float, float, float]]:
        if q_seed is None:
            q_seed = np.array(
                [from_pose[i] for i in range(1, 8)], dtype=float
            )

        cube = self.current_cube
        self.update_tracker()

        duration = 2.0
        pose = from_pose.copy()
        ok = False
        target_world = (0.0, 0.0, 0.0)

        for it in range(iterations):
            px, py = self.predict_cube_at(duration)
            py += self.y_lead_offset
            target_world = (px, py, cube.z + z_offset)

            self.get_logger().info(
                f'  intercept iter {it}: dt={duration:.2f}s -> '
                f'pred=({px:.3f}, {py:.3f}, {target_world[2]:.3f})'
            )

            pose, ok = self.solve_6dof_target(
                target_world, R_target, q_seed=q_seed
            )

            duration = self.estimate_motion_time(from_pose, pose)

        return pose, duration, ok, target_world


    def close_gripper_while_tracking(
        self,
        target_width: float,
        z_world: float,
        R_target: np.ndarray,
        duration: float = 0.5,
        hz: float = 30.0,
    ):
        steps = max(2, int(duration * hz))
        start_w = self.current_gripper
        target_w = float(np.clip(target_width, 0.0, 0.04))
        cube = self.current_cube

        self.get_logger().info(
            f'Closing gripper {start_w:.3f}->{target_w:.3f} while tracking'
        )

        q_current = np.array(
            [self.current_pose[i] for i in range(1, 8)], dtype=float
        )

        for step in range(steps + 1):
            a = step / steps
            s = 3.0 * a * a - 2.0 * a * a * a

            w = start_w + (target_w - start_w) * s
            self.publish_gripper(w)

            if step % 5 == 0 and step < steps:
                self.update_tracker()
                cy = self.kf.get_state()[1]
                vy = self.kf.get_state()[3]
                t_left = (steps - step) / hz
                pred_y = cy + vy * t_left

                new_pos = (cube.init_x, pred_y, z_world)
                new_pose, ok = self.solve_6dof_target(
                    new_pos, R_target, q_seed=q_current, tight=True
                )
                if ok:
                    q_current = np.array(
                        [new_pose[i] for i in range(1, 8)], dtype=float
                    )

            cmd = {i + 1: float(q_current[i]) for i in range(7)}
            self.publish_joint_pose(cmd)
            time.sleep(1.0 / hz)

        self.current_gripper = target_w
        self.current_pose = {i + 1: float(q_current[i]) for i in range(7)}

        self.get_logger().info('Gripper closed while tracking done')


    def pick_cube(self, cube: CubeTarget, R_grasp: np.ndarray,
                  ABOVE: float, APPROACH: float, GRASP: float):
        self.init_cube_tracker(cube)

        HOME = {
            1: 0.0, 2: -0.35, 3: 0.0, 4: -2.20,
            5: 0.0, 6: 2.00, 7: 0.785
        }
        q_home = np.array([HOME[i] for i in range(1, 8)], dtype=float)

        self.get_logger().info(f'=== PICKING {cube.name} ===')

        self.set_gripper_smooth(0.04, duration=0.5)
        self.hold_pose(0.2)

        home_dur = self.estimate_motion_time(self.current_pose, HOME)
        self.move_joints_smooth(HOME, duration=max(home_dur, 2.0))
        self.hold_pose(0.3)

        self.get_logger().info(f'--- Planning ABOVE for {cube.name} ---')
        above_pose, above_dur, ok1, above_tgt = self.plan_intercept(
            from_pose=HOME,
            z_offset=ABOVE,
            R_target=R_grasp,
            q_seed=q_home,
            iterations=4
        )
        self.verify_fk(above_pose, f'{cube.name}_above')
        self.move_joints_smooth(above_pose, duration=above_dur)
        self.hold_pose(0.2)

        self.get_logger().info(f'--- APPROACH {cube.name} ---')
        self.move_tracking(
            z_world=cube.z + APPROACH,
            R_target=R_grasp,
            duration=1.2,
            hz=20.0,
            ik_every_n=4,
        )

        self.get_logger().info(f'--- GRASP {cube.name} ---')
        self.move_tracking(
            z_world=cube.z + GRASP,
            R_target=R_grasp,
            duration=1.2,
            hz=20.0,
            ik_every_n=4,
            tight_ik=True,
        )

        grasp_close_target = 0.010
        self.get_logger().info(f'--- Closing gripper on {cube.name} (tracking) ---')
        self.close_gripper_while_tracking(
            target_width=grasp_close_target,
            z_world=cube.z + GRASP,
            R_target=R_grasp,
            duration=0.4,
            hz=30.0,
        )
        self.hold_pose(0.3)

        self.get_logger().info(f'--- Lifting {cube.name} ---')
        q_grasp = np.array(
            [self.current_pose[i] for i in range(1, 8)], dtype=float
        )
        T_grasp = self.ik.fk(q_grasp)
        grasp_world = T_grasp[:3, 3] + self.ik.base_world

        lift_approach_pos = (
            grasp_world[0], grasp_world[1], cube.z + APPROACH
        )
        lift_approach_pose, _ = self.solve_6dof_target(
            lift_approach_pos, R_grasp, q_seed=q_grasp
        )
        lift_dur = self.estimate_motion_time(
            self.current_pose, lift_approach_pose
        )
        self.move_joints_smooth(lift_approach_pose, duration=lift_dur)
        self.hold_pose(0.3)

        lift_above_pos = (
            grasp_world[0], grasp_world[1], cube.z + ABOVE
        )
        q_la = np.array(
            [self.current_pose[i] for i in range(1, 8)], dtype=float
        )
        lift_above_pose, _ = self.solve_6dof_target(
            lift_above_pos, R_grasp, q_seed=q_la
        )
        lift_above_dur = self.estimate_motion_time(
            self.current_pose, lift_above_pose
        )
        self.move_joints_smooth(lift_above_pose, duration=lift_above_dur)
        self.hold_pose(0.5)

        return lift_above_pose


    def place_cube(self, R_place: np.ndarray,
                   place_x: float, place_y: float, place_z: float,
                   ABOVE: float, APPROACH: float):
        self.get_logger().info(
            f'--- Placing at ({place_x}, {place_y}, {place_z}) ---'
        )

        q_current = np.array(
            [self.current_pose[i] for i in range(1, 8)], dtype=float
        )

        place_above, ok4 = self.solve_6dof_target(
            (place_x, place_y, place_z + ABOVE), R_place, q_seed=q_current
        )
        place_above_dur = self.estimate_motion_time(
            self.current_pose, place_above
        )
        q_pa = np.array(
            [place_above[i] for i in range(1, 8)], dtype=float
        )

        place_down, ok5 = self.solve_6dof_target(
            (place_x, place_y, place_z + APPROACH), R_place, q_seed=q_pa
        )
        place_down_dur = self.estimate_motion_time(place_above, place_down)

        self.move_joints_smooth(place_above, duration=place_above_dur)
        self.hold_pose(0.5)

        self.move_joints_smooth(place_down, duration=place_down_dur)
        self.hold_pose(0.5)

        release_dur = self.estimate_gripper_time(self.current_gripper, 0.04)
        self.set_gripper_smooth(0.04, duration=release_dur)
        self.hold_pose(1.0)


        retreat_dur = self.estimate_motion_time(
            self.current_pose, place_above
        )
        self.move_joints_smooth(place_above, duration=retreat_dur)
        self.hold_pose(0.5)


    def run_sequence(self):
        CUBE_YAW_DEG = 90.0
        PLACE_X, PLACE_Y, PLACE_Z = 0.29, -0.35, 0.82

        HOME = {
            1: 0.0, 2: -0.35, 3: 0.0, 4: -2.20,
            5: 0.0, 6: 2.00, 7: 0.785
        }


        ABOVE   =  0.15
        APPROACH =  0.06
        GRASP    = -0.005

        CUBE_YAW = math.radians(CUBE_YAW_DEG)
        R_grasp = self.ik.make_top_down_orientation(yaw=CUBE_YAW)

        self.get_logger().info('=== START MULTI-CUBE PICK-PLACE ===')

        self.publish_joint_pose(self.current_pose)
        self.publish_gripper(self.current_gripper)
        self.hold_pose(1.0)

        self.set_gripper_smooth(0.04, duration=1.0)
        self.hold_pose(0.5)

        self.move_joints_smooth(HOME, duration=3.0)
        self.hold_pose(0.8)

        for i, cube in enumerate(self.cubes):
            self.get_logger().info(
                f'========== CUBE {i+1}/{len(self.cubes)}: {cube.name} =========='
            )

            self.pick_cube(cube, R_grasp, ABOVE, APPROACH, GRASP)

            offset_y = i * 0.06
            self.place_cube(
                R_grasp,
                PLACE_X, PLACE_Y + offset_y, PLACE_Z,
                ABOVE, APPROACH
            )

        home_dur = self.estimate_motion_time(self.current_pose, HOME)
        self.move_joints_smooth(HOME, duration=home_dur)
        self.hold_pose(1.0)

        self.get_logger().info('=== ALL CUBES DONE ===')


def main(args=None):
    rclpy.init(args=args)
    node = PandaPickPlace6DOF()
    try:
        rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()