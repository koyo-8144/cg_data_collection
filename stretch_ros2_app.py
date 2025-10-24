import numpy as np
import pyrealsense2 as rs
import torch
from scipy.spatial.transform import Rotation as R 
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from sensor_msgs_py import point_cloud2
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from cv_bridge import CvBridge

# ===== Convert from Quaternion to Rotation Matrix ===== #
def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion into a rotation matrix.
    
    Parameters:
    - q: A quaternion in the format [x, y, z, w].
    
    Returns:
    - A 3x3 rotation matrix.
    """
    w, x, y, z = q[3], q[0], q[1], q[2]
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])


# ===== StretchApp Class ===== #
class StretchApp(Node):
    def __init__(self):
        super().__init__('stretch_app_node')

        # Initialize ROS2 components
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        # TF broadcaster already exists; add buffer + listener for lookups
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Common BEST_EFFORT profile for camera topics (matches all publishers)
        camera_qos = QoSProfile(
            depth=1,  # Matches all camera publishers
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        self.create_subscription(
            Image,
            '/camera/color/image_raw', # head camera
            # '/gripper_camera/image_raw',
            self._head_color_image_callback,
            qos_profile=camera_qos
        )
        
        self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw', # head camera
            # '/gripper_camera/depth/image_rect_raw',
            self._head_depth_image_callback,
            qos_profile=camera_qos
        )
        
        self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info', # head camera
            # '/gripper_camera/camera_info',
            self._head_camera_info_callback,
            qos_profile=camera_qos
        )
        
        self.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            10
        )

        self.color_image = None
        self.depth_image = None
        self.width = None
        self.height = None
        self.intrinsic_matrix = None
        self.T_odom_to_base = np.eye(4)
    
    # --- ROS2 Callbacks ---
    def _head_color_image_callback(self, msg: Image):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        print("Received color image")
            

    def _head_depth_image_callback(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg)
        print("Received depth image")
        # depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _head_camera_info_callback(self, msg: CameraInfo):
        self.width = msg.width
        self.height = msg.height
        self.intrinsic_matrix = self.get_intrinsic_mat_from_intrinsics(msg)
        print("Received camera info")

    def _odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        translation = pose.position
        quat = pose.orientation
        rot_matrix = quaternion_to_rotation_matrix([quat.x, quat.y, quat.z, quat.w])
        T_odom_to_base = np.eye(4)
        T_odom_to_base[:3, :3] = rot_matrix
        T_odom_to_base[0, 3] = translation.x
        T_odom_to_base[1, 3] = translation.y
        T_odom_to_base[2, 3] = translation.z
        self.T_odom_to_base = T_odom_to_base

    # --- Utilities ---

    def get_intrinsic_mat_from_intrinsics(self, intrinsics):
        if isinstance(intrinsics, CameraInfo):
            K = intrinsics.k
        else:
            K = intrinsics.coeffs
        intrinsic_matrix = np.array(K).reshape(3, 3)
        return intrinsic_matrix
    
    def lookup_transform(self, target_frame, source_frame, time=rclpy.time.Time()):
        try:
            trans = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                time,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return trans
        except Exception as e:
            self.get_logger().error(f"Transform lookup failed: {e}")
            return None

    def get_extrinsics_from_tf(self, target_frame, source_frame):
        trans = self.lookup_transform(target_frame, source_frame)
        if trans:
            return {
                'translation': np.array([trans.transform.translation.x,
                                          trans.transform.translation.y,
                                          trans.transform.translation.z]),
                'quaternion': np.array([trans.transform.rotation.x,
                                      trans.transform.rotation.y,
                                      trans.transform.rotation.z,
                                      trans.transform.rotation.w])
            }
        return None


    # --- Main Function to Get Frame Data ---
    def get_frame_data(self):
        # Get color and depth frames
        color_frame = self.color_image
        depth_frame = self.depth_image
        print(f"Color frame: {color_frame is not None}, Depth frame: {depth_frame is not None}")
        # print("color_frame:", color_frame)
        # print("depth_frame:", depth_frame)

        if color_frame is None or depth_frame is None:
            return None, None, None, None
        
        # Convert images to numpy arrays
        rgb = color_frame
        depth = depth_frame
        
        # Get intrinsic matrix 
        intrinsics_matrix = self.intrinsic_matrix

        # Reformat the intrinsic matrix 
        intrinsics_full = np.eye(4) 
        intrinsics_full[:3, :3] = intrinsics_matrix 

        # Get base_link to camera transform
        base_to_camera = self.get_extrinsics_from_tf("camera_color_optical_frame", "base_link")
        if base_to_camera is None:
            return None, None, None, None
        translation_base_to_camera = base_to_camera['translation']
        quat_base_to_camera = base_to_camera['quaternion']
        rotation_matrix_base_to_camera = quaternion_to_rotation_matrix(quat_base_to_camera)
        T_base_to_camera = np.eye(4)
        T_base_to_camera[:3, :3] = rotation_matrix_base_to_camera
        T_base_to_camera[:3, 3] = translation_base_to_camera
        print("T_base_to_camera:", T_base_to_camera)

        # Get odom to base_link transform from odom topic
        T_odom_to_base = self.T_odom_to_base
        print("T_odom_to_base:", T_odom_to_base)

        # Transform from odom to camera coordinates
        T_odom_to_camera = T_odom_to_base @ T_base_to_camera
        print("T_odom_to_camera:", T_odom_to_camera)

        # transformation_matrix = self.correct_pose(
        #     transformation_matrix).numpy()

        return rgb, depth, intrinsics_full, T_odom_to_camera

