#!/usr/bin/env python
'''
Create camera object.
'''
import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
    import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import rospy
from camera_callback import SubscriberCallbacks
from copy import deepcopy as copy
import numpy as np
from IPython import embed
from geometry_msgs.msg import PointStamped
import tf
from IPython import embed

MAX_DEPTH = 5.0
BASE_FRAME = '/base'
KINECT_FRAME = '/kinect2_rgb_optical_frame'


class Camera(SubscriberCallbacks):

    def __init__(self):
        SubscriberCallbacks.__init__(self)
        self.lr = tf.listener.TransformListener()

    def get_3D(self, pt, z_norm=None, depth_arr=None):
        temp_p = self._get_3D_camera(pt, z_norm, depth_arr)
        base_pt = self._convert_frames(temp_p)
        return base_pt

    def _get_image(self):
        return copy(self.image_color)

    def _get_depth(self):
        return copy(self.image_depth)

    def _process_depth(self, cur_depth=None):
        if cur_depth is None:
            cur_depth = self._get_depth()
        cur_depth = cur_depth / 1000.  # conversion from mm to m
        # cur_depth[cur_depth>MAX_DEPTH] = 0.
        return cur_depth

    def _get_3D_camera(self, pt, norm_z=None, depth_arr=None):
        assert len(pt) == 2
        cur_depth = self._process_depth(cur_depth=depth_arr)
        z = self._get_z_mean(cur_depth, [pt[0], pt[1]], 5) - 0.5
        # z = 1.28823
        print("\n z_camera = {} \n".format(z))
        # z = self._get_z_min(cur_depth, [pt[0], pt[1]], 5)
        if z == 0.:
            raise RuntimeError
        if norm_z is not None:
            z = z / norm_z
        u = pt[0]
        v = pt[1]
        P = copy(self.camera_P)
        P_n = np.zeros((3, 3))
        P_n[:, :2] = P[:, :2]
        P_n[:, 2] = P[:, 3] + P[:, 2] * z
        P_n_inv = np.linalg.inv(P_n)
        temp_p = np.dot(P_n_inv, np.array([u, v, 1]))
        temp_p = temp_p / temp_p[-1]
        temp_p[-1] = z
        return temp_p

    def _get_2D_camera(self, pt):
        """
        convrert 3D point in camera frame to image cordinate
        pt = [X,Y,Z]
        """
        img_pt = np.dot(self.camera_P, np.array(pt + [1]))
        img_pt /= img_pt[-1]
        return [int(img_pt[0]), int(img_pt[1])]

    def _convert_frames(self, pt):
        """
        convert point to base robot frame
        Args:
            pt:

        Returns:

        """
        assert len(pt) == 3
        ps = PointStamped()
        ps.header.frame_id = KINECT_FRAME
        ps.point.x, ps.point.y, ps.point.z = pt
        base_ps = self.lr.transformPoint(BASE_FRAME, ps)
        base_pt = np.array([base_ps.point.x, base_ps.point.y, base_ps.point.z])
        return base_pt

    def _convert_base_frame(self, pt):
        """
        transform point to kinect frame
        Args:
            pt:

        Returns:

        """
        assert len(pt) == 3
        ps = PointStamped()
        ps.header.frame_id = BASE_FRAME
        ps.point.x, ps.point.y, ps.point.z = pt
        base_ps = self.lr.transformPoint(KINECT_FRAME, ps)
        base_pt = np.array([base_ps.point.x, base_ps.point.y, base_ps.point.z])
        return base_pt

    def _get_z_mean(self, depth, pt, bb=5):
        sum_z = 0.
        nps = 0
        for i in range(bb * 2):
            for j in range(bb * 2):
                new_pt = [pt[0] - bb + i, pt[1] - bb + j]
                try:
                    new_z = depth[int(new_pt[0]), int(new_pt[1])]
                    if new_z > 0.:
                        sum_z += new_z
                        nps += 1
                except:
                    pass
        if nps == 0.:
            return 0.
        else:
            return sum_z / nps

    def _get_z_min(self, depth, pt, bb=5):
        min_z = 1e10
        nps = 0
        for i in range(bb * 2):
            for j in range(bb * 2):
                new_pt = [pt[0] - bb + i, pt[1] - bb + j]
                try:
                    new_z = depth[int(new_pt[0]), int(new_pt[1])]
                    if new_z == 0:
                        continue
                    if new_z < min_z:
                        min_z = new_z
                        nps += 1
                except:
                    pass
        if nps == 0.:
            return 0.
        else:
            return min_z

    def _get_pcd(self):
        """
        Returns the point cloud in camera's coordinate frame

        :param depth_im: depth image (shape: :math:`[H, W]`)
        :param rgb_im: rgb image (shape: :math:`[H, W, 3]`)

        :type depth_im: numpy.ndarray
        :type rgb_im: numpy.ndarray

        :returns: pts_in_cam: point coordinates in
                              camera frame (shape: :math:`[3, N]`)
        :rtype (numpy.ndarray, numpy.ndarray)
        """
        depth = copy(self.image_depth) / 1000.
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        valid = (depth > 1.4) & (depth < 1.9)
        z = np.where(valid, depth, np.nan)
        x = np.where(valid, z * (c - 484.2632619012746) / 529.2861620676242, 0)
        y = np.where(valid, z * (r - 255.18966778927916) / 529.2825667938571, 0)
        return np.dstack((x, y, z))


if __name__ == "__main__":
    embed()
