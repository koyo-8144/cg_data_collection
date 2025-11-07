import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
import tf


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
class FrankaApp:
    def __init__(self):
        rospy.init_node("franka_controller_node")

        # Initialize ROS2 components
        self.bridge = CvBridge()
        self.tf_broadcaster = tf.TransformBroadcaster()
        # TF broadcaster already exists; add buffer + listener for lookups
        self.tf_listener = tf.TransformListener()

        # Common BEST_EFFORT profile for camera topics (matches all publishers)
        # camera_qos = QoSProfile(
        #     depth=1,  # Matches all camera publishers
        #     reliability=QoSReliabilityPolicy.BEST_EFFORT,
        #     durability=QoSDurabilityPolicy.VOLATILE,
        #     history=QoSHistoryPolicy.KEEP_LAST
        # )

        rospy.Subscriber(
            '/mujoco_server/cameras/eef_camera/rgb/image_raw',  # head camera
            Image,
            self._head_color_image_callback,
        )

        rospy.Subscriber(
            '/mujoco_server/cameras/eef_camera/depth/image_raw',  # head camera
            Image,
            self._head_depth_image_callback,
        )

        rospy.Subscriber(
            '/mujoco_server/cameras/eef_camera/rgb/camera_info',  # head camera
            CameraInfo,
            # '/gripper_camera/camera_info',
            self._head_camera_info_callback,
        )

        rospy.Timer(rospy.Duration(1), lambda _: self.get_frame_data())

        # rospy.Subscriber(
        #     Odometry,
        #     '/odom',
        #     self._odom_callback,
        #     10
        # )

        self.color_image = None
        self.depth_image = None
        self.width = None
        self.height = None
        self.intrinsic_matrix = None
        self.T_odom_to_base = np.eye(4)

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

    # --- Utilities ---

    def get_intrinsic_mat_from_intrinsics(self, intrinsics):
        if isinstance(intrinsics, CameraInfo):
            K = intrinsics.K
        else:
            K = intrinsics.coeffs
        intrinsic_matrix = np.array(K).reshape(3, 3)
        return intrinsic_matrix

    def lookup_transform(self, target_frame, source_frame):
        while not rospy.is_shutdown():
            try:
                tvec, quat = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
                return np.asarray(tvec), quat
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.sleep(0.05)
                continue

    def get_extrinsics_from_tf(self, target_frame, source_frame):
        tvec, quat = self.lookup_transform(target_frame, source_frame)
        tvec = tvec.flatten()
        return {
            'translation': np.array(tvec),
            'quaternion': np.array(quat)
        }

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
        base_to_camera = self.get_extrinsics_from_tf("eef_camera_optical_frame", "panda_link0")

        if base_to_camera is None:
            return None, None, None, None

        translation_base_to_camera = base_to_camera['translation']
        quat_base_to_camera = base_to_camera['quaternion']
        rotation_matrix_base_to_camera = quaternion_to_rotation_matrix(quat_base_to_camera)
        T_base_to_camera = np.eye(4)
        T_base_to_camera[:3, :3] = rotation_matrix_base_to_camera
        T_base_to_camera[:3, 3] = translation_base_to_camera
        print("T_base_to_camera:", T_base_to_camera)

        return rgb, depth, intrinsics_full, T_base_to_camera


def main():
    app = FrankaApp()
    rospy.spin()


if __name__ == '__main__':
    main()
