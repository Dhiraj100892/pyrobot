import torch
import time
import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers
from copy import deepcopy as copy
import numpy as np
import torch
from IPython import embed
import io

import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
    import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

def quat2rot(quat, format='wxyz'):
    if format == 'wxyz':
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    elif format == 'xyzw':
        x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    else:
        raise ValueError(("'format' should either be '{}' or '{}'".format('wxyz', 'xyzw')))

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def quat2mat(quat):
    """
    stolen from https://github.com/ClementPinard/SfmLearner-Pytorch/blob/8b8398433709b8aa77e021caea64ab48207c8ef5/inverse_warp.py#L112
    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourth is then computed to have a norm of 1 -- size = [B,3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    rotMat = quat2rot(norm_quat, format='wxyz')
    return rotMat, norm_quat


def get_random_rot_trans(num=1, scale=1):
    """Generate random rotation matrix and translation vector
    Args:
        num: num of data points
        scale: range for translation vector [-scale, scale]
    """
    temp = torch.rand(num,3)
    rot, _ = quat2mat(temp)
    trans = scale*(torch.rand(num,3)-0.5)
    if num == 1:
        return rot[0], trans
    else:
        return rot, trans


class ArMarker(object):
    def __init__(self, ar_id, wait_time=5):
        self.ar_tag_id = ar_id
        self.wait_time = wait_time
        rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self._callback, queue_size=1)
        self.record = False
        self.marker_dict = None
    
    def _callback(self, data):
        for marker in data.markers:
            if marker.id == self.ar_tag_id and self.record:
                position_marker = [marker.pose.pose.position.x,
                                   marker.pose.pose.position.y,
                                   marker.pose.pose.position.z]
                orientation_marker = [marker.pose.pose.orientation.x,
                                      marker.pose.pose.orientation.y,
                                      marker.pose.pose.orientation.z,
                                      marker.pose.pose.orientation.w]
                marker_dict = {
                    'position': copy(np.array(position_marker)),
                    'orientation': copy(orientation_marker),
                }
                self.marker_dict = marker_dict
    
    def get_pose(self):
        """

        :return:
        """
        self.record = True
        start_time = time.time()
        pose = None
        while time.time() - start_time <= self.wait_time:
            if self.marker_dict is not None:
                pose = {'position': copy(self.marker_dict['position']),
                        'orientation': copy(self.marker_dict['orientation'])}

        self.record = False
        self.marker_dict = None
        if pose is None:    
            print("AR tag with id {} is not able to found in last {} sec ".format(self.ar_tag_id, self.wait_time))
            return None
        return pose


def compute_loss(t_B_G, r_B_G, t_C_A, r_C_A, t_G_C, q_G_C, rot_loss_w=0.0):
    """

    :param t_B_G:
    :param r_B_G:
    :param t_C_A:
    :param r_C_A:
    :param t_G_C:
    :param q_G_C:
    :param rot_loss_w:
    :return:
    """
    pred_r_G_C, _ = quat2mat(q_G_C)
    pred_r_G_C = pred_r_G_C[0]

    # for B-->G-->C-->A
    pred_t_B_A = torch.zeros_like(t_B_G)
    # for B-->G-->C-->A
    pred_r_B_A = torch.zeros_like(r_B_G)

    for i in range(t_B_G.shape[0]):
        r_B_C = torch.mm(r_B_G[i], pred_r_G_C)
        pred_t_B_A[i] = t_B_G[i] + torch.mm(r_B_G[i], t_G_C.t()).t() + torch.mm(r_B_C, t_C_A[i].reshape(-1,1)).t()
        pred_r_B_A[i] = torch.mm(torch.mm(r_B_G[i], pred_r_G_C), r_C_A[i])

    # NOTE: rot loss has to be relooked into, its not proper
    std = torch.cat((pred_t_B_A.std(dim=0), rot_loss_w * pred_r_B_A.reshape(-1,9).std(dim=0)))
    indv_loss = torch.norm(pred_t_B_A - pred_t_B_A.mean(), p=2, dim=1) + \
                rot_loss_w * torch.norm(pred_r_B_A.reshape(-1,9) - pred_r_B_A.reshape(-1,9).mean(), p=2, dim=1)
    return indv_loss, std.sum()


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img
        
        

    


