#!/usr/bin/env python
'''
Subscriber class
'''
import numpy as np
import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

ROSTOPIC_COLOR_IMAGE = '/kinect2/qhd/image_color_rect'
ROSTOPIC_DEPTH_IMAGE = '/kinect2/qhd/image_depth_rect'
ROSTOPIC_CAMERA_INFO_STREAM = '/kinect2/qhd/camera_info'
ROSTOPIC_POINTCLOUD = '/kinect2/qhd/points'


class SubscriberCallbacks(object):
    def __init__(self, node_name=None):
        if node_name is not None:
            rospy.init_node(node_name, anonymous=True)
        self.bridge = CvBridge()
        self.image_color = None
        self.image_depth = None
        self.camera_P = None
        self.ptcloud = None
        rospy.Subscriber(ROSTOPIC_COLOR_IMAGE, Image, self._image_color_callback, queue_size=1)
        rospy.Subscriber(ROSTOPIC_DEPTH_IMAGE, Image, self._image_depth_callback, queue_size=1)
        rospy.Subscriber(ROSTOPIC_CAMERA_INFO_STREAM, CameraInfo, self._camera_info_callback, queue_size=1)
        rospy.Subscriber(ROSTOPIC_POINTCLOUD, PointCloud2, self._ptcloud_callback, queue_size=1)

    def _image_color_callback(self, msg):
        self.image_color = self.bridge.imgmsg_to_cv2(msg)

    def _image_depth_callback(self, msg):
        self.image_depth = self.bridge.imgmsg_to_cv2(msg)

    def _camera_info_callback(self, msg):
        self.camera_P = np.array(msg.P).reshape((3,4))

    def _ptcloud_callback(self, msg):
        self.ptcloud = msg