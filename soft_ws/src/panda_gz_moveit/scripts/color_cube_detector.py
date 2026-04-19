#!/usr/bin/env python3
"""
Color Cube Detector with Point Cloud + per-cube topics.

Publishes:
  /cube_detection/visualization          — annotated camera image
  /cube_detection/pose                   — latest detection (any cube)
  /cube_detection/{class_name}/pose      — per-cube PoseStamped (e.g. green_cube, red_cube, blue_cube)
"""

import os
import math
import struct

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloCubeDetector(Node):
    def __init__(self):
        super().__init__('yolo_cube_detector')
        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.latest_pc = None

        model_path = '/workspace/soft_ws/src/panda_gz_moveit/scripts/best.pt'
        self.model = YOLO(model_path)
        self.get_logger().info(f'YOLOv8 model loaded from {model_path}')

        self.rgb_sub = self.create_subscription(
            Image, '/camera/image', self.rgb_callback, 10)
        self.pc_sub = self.create_subscription(
            PointCloud2, '/camera/points', self.pc_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.info_callback, 10)

        self.vis_pub = self.create_publisher(Image, '/cube_detection/visualization', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/cube_detection/pose', 10)

        self._cube_pubs: dict[str, rclpy.publisher.Publisher] = {}

        self.camera_pos_world = np.array([0.63, -0.8, 1.50])

    def _get_cube_pub(self, class_name: str):
        topic_name = class_name.strip().replace(' ', '_').lower()
        if topic_name not in self._cube_pubs:
            topic = f'/cube_detection/{topic_name}/pose'
            self._cube_pubs[topic_name] = self.create_publisher(PoseStamped, topic, 10)
            self.get_logger().info(f'Created per-cube publisher: {topic}')
        return self._cube_pubs[topic_name]


    def info_callback(self, msg):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(
                f'Camera intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} '
                f'cx={self.cx:.1f} cy={self.cy:.1f}')

    def pc_callback(self, msg):
        self.latest_pc = msg

    # ── Detection ──

    def get_orientation(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        angle = rect[2]
        if rect[1][0] < rect[1][1]:
            angle += 90
        return angle

    def rgb_callback(self, msg):
        if self.latest_pc is None:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image, conf=0.5, verbose=False)
        vis = cv_image.copy()

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = self.model.names[cls]

                pad = 5
                crop = cv_image[
                    max(0, y1 - pad):min(cv_image.shape[0], y2 + pad),
                    max(0, x1 - pad):min(cv_image.shape[1], x2 + pad)]
                angle = self.get_orientation(crop) if crop.size > 0 else 0.0

                cx_px = (x1 + x2) // 2
                cy_px = (y1 + y2) // 2
                point_3d = self.get_3d_point(cx_px, cy_px)

                if point_3d:
                    world_pt = self.camera_to_world(point_3d)

                    color = {
                        'green': (0, 255, 0),
                        'red': (0, 0, 255),
                        'blue': (255, 0, 0),
                    }
                    draw_color = (0, 255, 255)
                    for key, val in color.items():
                        if key in class_name.lower():
                            draw_color = val
                            break

                    cv2.rectangle(vis, (x1, y1), (x2, y2), draw_color, 2)
                    label = f"{class_name} {conf:.2f} Ang:{angle:.1f}"
                    cv2.putText(vis, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
                    coord_label = (f"W:({world_pt[0]:.2f},"
                                   f"{world_pt[1]:.2f},{world_pt[2]:.2f})")
                    cv2.putText(vis, coord_label, (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)

                    self.get_logger().info(
                        f'[{class_name}] world=({world_pt[0]:.3f},'
                        f'{world_pt[1]:.3f},{world_pt[2]:.3f}) '
                        f'angle={angle:.1f}°',
                        throttle_duration_sec=1.0)

                    # Publish to global and per-cube topics
                    pose_msg = self._make_pose_msg(
                        world_pt, angle, msg.header.stamp)
                    self.pose_pub.publish(pose_msg)
                    self._get_cube_pub(class_name).publish(pose_msg)

        vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        self.vis_pub.publish(vis_msg)

    # ── 3D point from point cloud ──

    def get_3d_point(self, u, v, search_radius=3):
        if self.latest_pc is None:
            return None
        pc = self.latest_pc
        width, height = pc.width, pc.height
        if u < 0 or u >= width or v < 0 or v >= height:
            return None
        point_step = pc.point_step
        row_step = pc.row_step
        data = pc.data

        offsets = {f.name: f.offset for f in pc.fields}
        x_off = offsets.get('x', 0)
        y_off = offsets.get('y', 4)
        z_off = offsets.get('z', 8)

        def read_point(pu, pv):
            if pu < 0 or pu >= width or pv < 0 or pv >= height:
                return None
            pos = pv * row_step + pu * point_step
            try:
                px = struct.unpack_from('f', data, pos + x_off)[0]
                py = struct.unpack_from('f', data, pos + y_off)[0]
                pz = struct.unpack_from('f', data, pos + z_off)[0]
            except struct.error:
                return None
            if math.isnan(px) or math.isnan(py) or math.isnan(pz):
                return None
            if math.isinf(px) or math.isinf(py) or math.isinf(pz):
                return None
            if pz <= 0.01 or pz > 5.0:
                return None
            return [px, py, pz]

        pt = read_point(u, v)
        if pt is not None:
            return pt

        valid = []
        for dv in range(-search_radius, search_radius + 1):
            for du in range(-search_radius, search_radius + 1):
                pt = read_point(u + du, v + dv)
                if pt is not None:
                    valid.append(pt)
        if valid:
            return np.median(valid, axis=0).tolist()
        return None

    # ── Helpers ──

    def _make_pose_msg(self, world_pt, angle_deg, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(world_pt[0])
        msg.pose.position.y = float(world_pt[1])
        msg.pose.position.z = float(world_pt[2])
        q = self.euler_to_quaternion(0.0, 0.0, math.radians(angle_deg))
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        return msg

    def euler_to_quaternion(self, roll, pitch, yaw):
        cr, sr = math.cos(roll/2), math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
        return [
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
            cr*cp*cy + sr*sp*sy,
        ]

    def camera_to_world(self, point_cam):
        self.get_logger().info(
            f'RAW cam: x={point_cam[0]:.3f} y={point_cam[1]:.3f} z={point_cam[2]:.3f}',
            throttle_duration_sec=2.0)
        R = np.array([
            [ 0, 0, 1],   # world x = cam z
            [ 0, 1, 0],   # world y = cam y
            [-1, 0, 0],   # world z = -cam x
        ])
        return (self.camera_pos_world + R @ np.array(point_cam)).tolist()


def main(args=None):
    rclpy.init(args=args)
    node = YoloCubeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()