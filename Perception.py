#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge
import tf
import tf.transformations as tft

class ParkingSpotIPMNode:
    def __init__(self):
        rospy.init_node("parking_spot_ipm_node")

        # 参数
        self.image_topic = rospy.get_param("~image_topic", "/camera/rgb/image_raw")
        self.marker_topic = rospy.get_param("~marker_topic", "/visualization_marker")
        self.pose_topic = rospy.get_param("~pose_topic", "/parking_spot/pose")
        self.spot_corners_path = rospy.get_param("~spot_corners_path", "spot_corners.npy")
        self.spot_size = np.array([0.24, 0.20], dtype=np.float32)
        self.inner_mean_thresh = rospy.get_param("~inner_mean_thresh", 150)
        self.ps_center_pub = rospy.Publisher("/parking_spot_center", Pose2D, queue_size=2)

        # 相机外参
        self.CAMERA_TO_BASE_X = 0.35
        self.CAMERA_TO_BASE_Y = -0.18
        self.CAMERA_TO_BASE_Z = 0.1
        self.CAMERA_PITCH_DEG = -20.0
        self.CAMERA_ROLL_DEG = 0.0
        self.CAMERA_YAW_DEG = 0.0

        # 标定四角和单应性矩阵
        self.image_pts = np.load(self.spot_corners_path).astype(np.float32)
        self.world_pts = np.array([
            [0, 0],
            [self.spot_size[0], 0],
            [self.spot_size[0], self.spot_size[1]],
            [0, self.spot_size[1]]
        ], dtype=np.float32)
        self.H, _ = cv2.findHomography(self.image_pts, self.world_pts)
        self.bridge = CvBridge()

        # ROS通信
        self.marker_pub = rospy.Publisher(self.marker_topic, Marker, queue_size=2)
        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=2)
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        self.tf_listener = tf.TransformListener()
        rospy.loginfo("节点初始化完成，等待检测停车位...")

        # 检测历史
        self.history = []    # 存储最近N次车位中心点
        self.N = 30        # 连续多少帧判定一次
        self.publish_ready = False

    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        corners = self.detect_white_rectangle_with_inner_dark(img)
        if corners is not None:
            ordered = self.order_points(corners)
            center_px = ordered.mean(axis=0).reshape(1, 2)
            center_xy_cam = self.ipm_transform(center_px)[0]
            x_base, y_base, z_base = self.camera_to_base(*center_xy_cam, 0.0)

            ordered_ipm = self.ipm_transform(ordered)
            x0, y0, _ = self.camera_to_base(*ordered_ipm[0], 0.0)
            x1, y1, _ = self.camera_to_base(*ordered_ipm[1], 0.0)
            yaw_base = math.atan2(y1 - y0, x1 - x0)    # 朝向角

            self.publish_marker_pose(x_base, y_base, yaw_base)

            for p in ordered.astype(int):
                cv2.circle(img, tuple(p), 4, (0,0,255), -1)
            cv2.polylines(img, [ordered.astype(int)], True, (0,255,0), 2)
            # 若要显示opencv窗口可以取消下面注释
            # cv2.imshow("spot_detection", img)
            # if cv2.waitKey(1) == 27:
            #     cv2.destroyAllWindows()

    def detect_white_rectangle_with_inner_dark(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0
        for cnt in contours:
            epsilon = 0.03 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if area < 500:
                    continue
                w, h = cv2.minAreaRect(approx)[1]
                if w == 0 or h == 0:
                    continue
                ratio = max(w, h) / min(w, h)
                if 0.5 < ratio < 3.2:
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [approx], -1, 255, -1)
                    mask_inner = cv2.erode(mask, np.ones((11,11), np.uint8))
                    mean_val = cv2.mean(gray, mask=mask_inner)[0]
                    if mean_val > self.inner_mean_thresh: # 只检测内部较亮的矩形
                        score = area / (abs(ratio-1.2)+1e-3)
                        if score > best_score:
                            best_score = score
                            best = approx
        if best is not None:
            return best.reshape(4,2).astype(np.float32)
        return None

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]         # 左上
        rect[2] = pts[np.argmax(s)]         # 右下
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]      # 右上
        rect[3] = pts[np.argmax(diff)]      # 左下
        return rect

    def ipm_transform(self, pts_uv):
        pts_uv = np.array(pts_uv, dtype=np.float32).reshape(-1,1,2)
        pts_xy = cv2.perspectiveTransform(pts_uv, self.H)
        return pts_xy.reshape(-1,2)

    def camera_to_base(self, x_cam, y_cam, z_cam=0.0):
        pitch = math.radians(self.CAMERA_PITCH_DEG)
        roll = math.radians(self.CAMERA_ROLL_DEG)
        yaw = math.radians(self.CAMERA_YAW_DEG)
        R = tft.euler_matrix(roll, pitch, yaw)
        T = tft.translation_matrix([self.CAMERA_TO_BASE_X, self.CAMERA_TO_BASE_Y, self.CAMERA_TO_BASE_Z])
        M = np.dot(T, R)
        pt_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        pt_base = np.dot(M, pt_cam)
        return pt_base[0], pt_base[1], pt_base[2]

    def publish_marker_pose(self, x, y, yaw_rad=0.0, ordered_ipm=None):
        now = rospy.Time.now()
        frame_id = "base_footprint"
        x_map, y_map, yaw_map = x, y, yaw_rad
        quat = tft.quaternion_from_euler(0, 0, yaw_rad)
        quat_map = None

        print(f"parking spot center under /base_link is:x = {x:.2f},y = {y:.2f},yaw={math.degrees(yaw_rad):.1f}degree" )

        try:
            self.tf_listener.waitForTransform("map", "base_footprint", now, rospy.Duration(0.5))
            import geometry_msgs.msg
            ps = geometry_msgs.msg.PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = "base_footprint"
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = quat

            ps_map = self.tf_listener.transformPose("map", ps)
            x_map = ps_map.pose.position.x
            y_map = ps_map.pose.position.y
            quat_map = ps_map.pose.orientation
            yaw_map = tft.euler_from_quaternion([
                quat_map.x, quat_map.y, quat_map.z, quat_map.w
            ])[2]
            frame_id = "map"
        except (tf.Exception, tf.LookupException, tf.ConnectivityException) as e:
            rospy.logwarn("TF transform to map failed, using base_footprint. %s", str(e))

        # ============ 历史判断并只发一次 Pose2D ============
        self.history.append((x_map, y_map, yaw_map))
        if len(self.history) > self.N:
            self.history.pop(0)

        if len(self.history) == self.N:
            center_x = np.mean([h[0] for h in self.history])
            center_y = np.mean([h[1] for h in self.history])
            center_theta = np.mean([h[2] for h in self.history])
            dists = [np.hypot(h[0] - center_x, h[1] - center_y) for h in self.history]
            if max(dists) < 0.2:
                if not self.publish_ready:
                    pose2d = Pose2D()
                    pose2d.x = center_x
                    pose2d.y = center_y
                    pose2d.theta = center_theta
                    self.ps_center_pub.publish(pose2d)
                    rospy.loginfo(f"已发布稳定车位中心: x={center_x:.3f}, y={center_y:.3f}, theta={math.degrees(center_theta):.1f}°")
                    self.publish_ready = True
            else:
                self.publish_ready = False

        # ============ rviz红色小箭头（每帧都发，rviz里永久保留） ============
        mk = Marker()
        mk.header.stamp = now
        mk.header.frame_id = frame_id
        mk.ns = "arrow"
        mk.id = 2
        mk.type = Marker.ARROW
        mk.action = Marker.ADD
        mk.scale.x = 0.10  # 箭头长度
        mk.scale.y = 0.04
        mk.scale.z = 0.04
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = 1, 0, 0, 1
        mk.pose.position.x = x_map
        mk.pose.position.y = y_map
        mk.pose.position.z = 0.04
        if quat_map is not None:
            mk.pose.orientation = quat_map
        else:
            mk.pose.orientation.x, mk.pose.orientation.y, mk.pose.orientation.z, mk.pose.orientation.w = quat
        self.marker_pub.publish(mk)

        # 下面PoseStamped继续发布（可选，调试/rviz用）
        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = frame_id
        pose.pose.position.x = x_map
        pose.pose.position.y = y_map
        pose.pose.position.z = 0.03
        if quat_map is not None:
            pose.pose.orientation = quat_map
        else:
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = quat
        self.pose_pub.publish(pose)

if __name__ == '__main__':
    try:
        ParkingSpotIPMNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
