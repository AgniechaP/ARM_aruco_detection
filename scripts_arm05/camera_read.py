#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import cv2.aruco
import numpy as np 
from geometry_msgs.msg import PoseArray, Pose
from geometry_msgs.msg import PoseArray, Pose, TransformStamped
import tf2_ros
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation as R


class CameraReader(Node):
    _DICTS = {
        "4x4_100" : cv2.aruco.DICT_4X4_100,
        "4x4_1000" : cv2.aruco.DICT_4X4_1000,
        "4x4_250" : cv2.aruco.DICT_4X4_250,
        "4x4_50" : cv2.aruco.DICT_4X4_50,
        "5x5_100" : cv2.aruco.DICT_5X5_100,
        "5x5_1000" : cv2.aruco.DICT_5X5_1000,
        "5x5_250" : cv2.aruco.DICT_5X5_250,
        "5x5_50" : cv2.aruco.DICT_5X5_50,
        "6x6_100" : cv2.aruco.DICT_6X6_100,
        "6x6_1000" : cv2.aruco.DICT_6X6_1000,
        "6x6_250" : cv2.aruco.DICT_6X6_250,
        "6x6_50" : cv2.aruco.DICT_6X6_50,
        "7x7_100" : cv2.aruco.DICT_7X7_100,
        "7x7_1000" : cv2.aruco.DICT_7X7_1000,
        "7x7_250": cv2.aruco.DICT_7X7_250,
        "7x7_50": cv2.aruco.DICT_7X7_50,
        "apriltag_16h5" : cv2.aruco.DICT_APRILTAG_16H5,
        "apriltag_25h9" : cv2.aruco.DICT_APRILTAG_25H9,
        "apriltag_36h10" : cv2.aruco.DICT_APRILTAG_36H10,
        "apriltag_36h11" : cv2.aruco.DICT_APRILTAG_36H11,
        "aruco_original" : cv2.aruco.DICT_ARUCO_ORIGINAL
    }

    def __init__(self, tag_set="aruco_original", target_width=0.9):
        super().__init__('camera_reader')
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()

        dict = CameraReader._DICTS.get(tag_set.lower(), None)
        if dict is None:
            self.get_logger().error(f'ARUCO tag set {tag_set} not found')
        else:
            self._aruco_dict = cv2.aruco.Dictionary_get(dict)
            self._aruco_params = cv2.aruco.DetectorParameters_create()
            self._target_width = target_width
            self._image = None
            self.get_logger().info(f"using dictionary {tag_set}")
    
    def camera_info_callback(self, msg):
        # Camera matrix (K) and distortion coefficients (D) from camera_info topic 
        # (distortion model is plumb_bob)
        self._camera_matrix = np.reshape(msg.k, (3, 3))
        self._dist_coeff = np.reshape(msg.d,(1,5))

    def callback(self, msg):
        try:
            self._image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(str(e))
            return

        grey = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(grey, self._aruco_dict)
        frame = cv2.aruco.drawDetectedMarkers(self._image, corners, ids)

        if ids is None:
            self.get_logger().info(f"No targets found")
            return
        if self._camera_matrix is None:
            self.get_logger().info(f"Not received cam info msg")
            return
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
            corners, self._target_width, self._camera_matrix, self._dist_coeff)
        result = self._image.copy()


        try:
            transform_stamped = self.tf_buffer.lookup_transform('map', 'camera_link', rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Failed to lookup transform: {str(e)}")
            return
        
        for i, id_aruco in enumerate(ids):
            aruco_pose = Pose()
            aruco_pose.position.x = float(tvec[i][0][0])
            aruco_pose.position.y = float(tvec[i][0][1])
            aruco_pose.position.z = float(tvec[i][0][2])

            rot_matrix = np.eye(4)
            rot_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvec[i][0]))[0]
            r = R.from_matrix(rot_matrix[0:3, 0:3])
            quat = r.as_quat() 

            aruco_pose.orientation.x = quat[0]
            aruco_pose.orientation.y = quat[1]
            aruco_pose.orientation.z = quat[2]
            aruco_pose.orientation.w = quat[3]

            # Transformation to the map frame
            transformed_pose = tf2_geometry_msgs.do_transform_pose(aruco_pose, transform_stamped)

            self.get_logger().info(f"ArUco ID {id_aruco} Target at {transformed_pose.position} rotation {transformed_pose.orientation}")

        for id_marker, r,t in zip(ids, rvec, tvec):
            # self.get_logger().info(f"ArUco ID {id_marker} Target at {t} rotation {r}")
            result = cv2.aruco.drawAxis(
                result, self._camera_matrix, self._dist_coeff, r, t, self._target_width)
            result_resized = cv2.resize(result, (360, 640))

        cv2.imshow("Camera output resized", result_resized)
        cv2.waitKey(3)

def main(args=None):
    rclpy.init(args=args)
    camera_reader = CameraReader()
    try:
        rclpy.spin(camera_reader)
    except KeyboardInterrupt:
        pass

    camera_reader.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
