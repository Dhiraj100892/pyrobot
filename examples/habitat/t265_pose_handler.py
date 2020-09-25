import rospy
from tf import TransformListener, TransformBroadcaster
from nav_msgs.msg import Odometry
import time
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion
from pyrobot import Robot
from IPython import embed


class TrackingCamPose:
    def __init__(self, *args):
        # TODO: might need to remove it when running with Pyrobot
        #rospy.init_node("test")

        self.global_frame = "t265_odom_frame"
        self.init_frame = "t265_odom_init_frame"
        self.source_frame = "t265_link"
        self.tf_listener_ = TransformListener()
        self.init_pose = None
        self.robot_tracking_camera_frame = "tracking_cam"
        self.robot_base_frame = "base_link"

        #
        time.sleep(2.0)
        self.set_init_frame()

        # fake callback for broadcasting init frame
        rospy.Subscriber("/t265/odom/sample", Odometry, self.callback)

    def callback(self, data):
        if self.init_pose is not None:
            br = TransformBroadcaster()
            br.sendTransform(self.init_pose[0], self.init_pose[1], rospy.Time.now(),
                             self.init_frame, self.global_frame)

    def set_init_frame(self):
        while self.init_pose is None:
            try:
                self.init_pose = self.tf_listener_.lookupTransform(self.global_frame, self.source_frame,
                                                                   self.tf_listener_.getLatestCommonTime(
                                                                       self.global_frame, self.source_frame))
            except:
                self.init_pose = None
                time.sleep(1)

    def get_base_state(self):
        # get the pose of camera in init_tracking frame
        cam_pose_in_cam_frame = self.tf_listener_.lookupTransform(self.source_frame,
                                                                  self.init_frame,
                                                                  self.tf_listener_.getLatestCommonTime(
                                                                      self.source_frame,
                                                                      self.init_frame))
        print("cam pose in init frame ={}".format(cam_pose_in_cam_frame))

        # get the pose of base in robot_tracking_camera_frame(fake source frame)
        base_pose_in_cam_frame = self.tf_listener_.lookupTransform(self.robot_tracking_camera_frame,
                                                                   self.robot_base_frame,
                                                                   self.tf_listener_.getLatestCommonTime(
                                                                       self.robot_tracking_camera_frame,
                                                                       self.robot_base_frame))
        print("base pose in cam frame ={}".format(base_pose_in_cam_frame))

        # get the pose of base frame in init frame
        pose = PoseStamped()
        pose.header.frame_id = self.source_frame
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = base_pose_in_cam_frame[0]
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = \
            base_pose_in_cam_frame[1]
        base_pose_in_init_frame = self.tf_listener_.transformPose(self.init_frame, pose)
        (roll, pitch, yaw) = euler_from_quaternion([base_pose_in_init_frame.pose.orientation.x,
                                                    base_pose_in_init_frame.pose.orientation.y,
                                                    base_pose_in_init_frame.pose.orientation.z,
                                                    base_pose_in_init_frame.pose.orientation.w])

        return base_pose_in_init_frame.pose.position.x, base_pose_in_init_frame.pose.position.y, yaw

robot = Robot('locobot')
p = TrackingCamPose()
embed()