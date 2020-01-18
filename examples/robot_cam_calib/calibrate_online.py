# for doing optimization online
# run optimization in parallel with the data collection

# In this case the camera is attached to gripper of
# for complete calibration, in this we will consider base(B), camera(C), gripper(G) and AR Tag(A) frames
# transform from B<-->G and C<-->A is known
# need to figure out transform between G<-->C
# P_X_Y --> represent origin of Y frame in X frame of reference

import torch
from torch import optim
from IPython import embed
import numpy as np
from util import quat2mat, quat2rot, compute_loss, get_img_from_fig
import argparse
import os
import pickle
import threading
from util import ArMarker
from termcolor import colored
from pyrobot import Robot
from copy import deepcopy
mport tkinter
import matplotlib
matplotlib.use('TkAgg')
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
    import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

class GetCameraExtrensics(object):

    def __init__(self, args):
        """

        :param args:
        """
        self.data_dir = args.data_dir
        self.rot_loss_w = args.rot_loss_w
        self.vis = args.vis
        self.num_iter = args.num_iter
        self.gripper_frame = args.gripper_frame
        self.camera_frame = args.camera_frame
        self.ar_marker = ArMarker(args.ar_id)
        self.robot = Robot(args.robot_name, use_base=False, use_camera=False)
        self.num_points = 0

        # start with some random guess
        self.quat_G_C = torch.rand(1, 3).double().requires_grad_(True)
        self.trans_G_C = torch.rand(1, 3).double().requires_grad_(True)
        self.optimizer = optim.Adam([self.quat_G_C, self.trans_G_C], lr=0.1)

        # threading things
        self._optimization_thread = threading.Thread(target=self.optimize, args=())
        self._optimization_thread.daemon = True  # Daemonize thread
        self._lock = threading.Lock()
        self._num_iter = 0
        self.trans_B_G = None
        self.rot_B_G = None
        self.trans_C_A = None
        self.rot_C_A = None

    def calibrate(self):
        """

        :return:
        """
        self._optimization_thread.start()
        while True:
            # get the ar marker position
            marker_pose = self.ar_marker.get_pose()

            if marker_pose is None:
                continue

            # project marker pose to base frame and vizualize it  from using G<-->C found calibration
            trans_B_G, rot_B_G, _ = self.robot.arm.pose_ee
            marker_t, marker_r = marker_pose['position'], quat2rot(marker_pose['orientation'], format='xyzw')
            with self._lock:
                pred_r_G_C_tensor, pred_q_G_C_tensor = quat2mat(self.quat_G_C)
                pred_trans_G_C = self.trans_G_C.numpy()
                pred_r_G_C = pred_r_G_C_tensor.numpy()

            # convert all points in base frame
            if self.trans_B_G is not None:
                prev_t_B_A = np.zeros_like(self.trans_B_G)
                with self._lock:
                    for i in range(self.trans_B_G.shape[0]):
                        r_B_C = np.matmul(self.rot_B_G[i], pred_r_G_C)
                        prev_t_B_A[i] = self.trans_B_G[i] + np.matmul(self.rot_B_G[i], pred_trans_G_C.t()).t() + \
                                        np.matmul(r_B_C, self.trans_C_A[i].reshape(-1, 1)).t()

            r_B_C = np.matmul(rot_B_G, pred_r_G_C)
            cur_t_B_A = trans_B_G + np.matmul(rot_B_G, pred_trans_G_C.t()).t() + \
                        np.matmul(r_B_C, marker_t.reshape(-1, 1)).t()

            # TODO: display it in the real world
            if self.trans_B_G is not None and self.trans_B_G.shape[0] >=2:
                img = self.get_plot_img(prev_t_B_A, cur_t_B_A)
                cv2.imshow('test', img)
                key = cv2.waitKey(33)

                if key == 'r':
                    inp = input(colored("Looks like you want to record data[Y/N] and press enter\n", "green"))
                    if inp == 'Y':
                        print(colored("data added\n", 'red'))
                        if self.trans_B_G is None:
                            self.trans_B_G = deepcopy(trans_B_G.reshape(1, -1))
                            self.rot_B_G = deepcopy(rot_B_G.reshape(1, 3, 3))
                            self.trans_C_A = deepcopy(marker_t.reshape(1, -1))
                            self.rot_C_A = deepcopy(marker_r.reshape(1, 3, 3))
                        else:
                            with self._lock:
                                self.trans_B_G = np.concatenate((self.trans_B_G, trans_B_G.reshape(-1)), axis=0)
                                self.rot_B_G = np.concatenate((self.rot_B_G, rot_B_G.reshape(3, 3)), axis=0)
                                self.trans_C_A = np.concatenate((self.trans_C_A, marker_t.reshape(-1)), axis=0)
                                self.rot_C_A = np.concatenate((self.rot_C_A, marker_r.reshape(3,3)), axis=0)

                elif key == 'k':
                    inp = input(colored("Looks like you want to kill the process [Y/N] and press enter\n", "green"))
                    if inp == 'Y':
                        print(colored("killing process", 'red'))
                        pred_q_G_C = pred_q_G_C_tensor.numpy()
                        cmd = "rosrun tf static_transform_publisher " + str(float(pred_trans_G_C[0])) + ' ' + \
                              str(float(pred_trans_G_C[1])) + ' ' + str(float(pred_trans_G_C[2])) + ' ' + str(pred_q_G_C[1]) + ' '  \
                              + str(pred_q_G_C[2]) + ' ' + str(pred_q_G_C[3]) + ' ' + str(pred_q_G_C[0]) + ' ' \
                              + self.gripper_frame + ' ' + self.camera_frame + ' 10'
                        print("\nRun Command\n")
                        print(colored(cmd, 'green'))
                        break

    def get_plot_img(self, prev_point, cur_point):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(prev_point[:,0], prev_point[:,1], prev_point[:,2], marker='o')
        ax.scatter3D(cur_point[0], cur_point[1], cur_point[2], marker='^')
        scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", marker='o')
        ax.legend([scatter1_proxy], ['Base to Ar from Gripper & Camera'], numpoints=1)
        img = get_img_from_fig(fig)
        return img

    # optimize the parameters
    def optimize(self):
        """

        Returns:

        """

        ###################
        # optimize the G<-->C in 3D space
        while True:
            with self._lock:
                if self.trans_B_G is None or self.trans_B_G.shape[0] < 2:
                    continue
                _, loss = compute_loss(torch.from_numpy(self.trans_B_G), torch.from_numpy(self.rot_B_G),
                                       torch.from_numpy(self.trans_C_A), torch.from_numpy(self.rot_C_A),
                                       self.trans_G_C, self.quat_G_C,
                                       0)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.num_iter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process args for calibration")
    parser.add_argument('--rot_loss_w', help='weight on rotational loss for optimizing the camera extrensic parameters',
                        type=float, default=0.0)
    parser.add_argument('--vis', action='store_true', default=False,
                        help='for visualizing data points after calibration')
    parser.add_argument('--num_iter', help='number of iteration of optimization', type=int, default=1000)
    parser.add_argument('--data_dir', help='Directory to load data points', type=str, default="robot_ar_data")
    parser.add_argument('--gripper_frame', help='robot gripper frame name', type=str, default="/wrist")
    parser.add_argument('--camera_frame', help='camera frame name', type=str, default="/kinect2_rgb_optical_frame")
    parser.add_argument('--ar_id', help='id for the ar marker', type=int)
    parser.add_argument('--robot_name', help='name of the robot', type=str, default="sawyer")

    args = parser.parse_args()
    get_camera_extrensics = GetCameraExtrensics(args)
    get_camera_extrensics.calibrate()