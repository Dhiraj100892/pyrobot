# In this case the camera is attached to gripper of
# for complete calibration, in this we will consider base(B), camera(C), gripper(G) and AR Tag(A) frames
# transform from B<-->G and C<-->A is known
# need to figure out transform between G<-->C
# P_X_Y --> represent origin of Y frame in X frame of reference

import torch
from torch import optim
from IPython import embed
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib
from util import quat2mat, quat2rot, compute_loss
import argparse
import os
import pickle


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

    def calibrate(self):
        """

        :return:
        """
        # generate data
        self.load_data(self.data_dir)
        # optimize the parmas
        self.optimize(self.num_iter, self.rot_loss_w, self.vis)

    # data set generator
    def load_data(self, data_dir):
        """

        :param data_dir:
        :return:
        """
        # load hand data
        with open(os.path.join(data_dir, 'arm.p'), 'rb') as f:
            arm_data = pickle.load(f)['data']

        # load marker data
        with open(os.path.join(data_dir, 'marker.p'), 'rb') as f:
            marker_data = pickle.load(f)['data']

        self.num_points = min(len(arm_data),len(marker_data))
        self.trans_B_G = torch.from_numpy(np.array([arm_data[i]['position'] for i in range(self.num_points)])
                                          .reshape(-1, 3))
        self.rot_B_G = torch.from_numpy(np.array([arm_data[i]['orientation'] for i in range(self.num_points)]))
        self.trans_C_A = torch.from_numpy(np.array([marker_data[i]['position'] for i in range(self.num_points)]).
                                          reshape(-1, 3))
        quat_C_A = torch.from_numpy(np.array([marker_data[i]['orientation'] for i in range(self.num_points)]))
        self.rot_C_A = quat2rot(quat_C_A, format='xyzw')

    # optimize the parameters
    def optimize(self, num_iter, rot_loss_w, vis):
        """

        :param num_iter:
        :param rot_loss_w:
        :param vis:
        :return:
        """
        # start with some random guess
        quat_G_C = torch.rand(1,3).double().requires_grad_(True)
        trans_G_C = torch.rand(1,3).double().requires_grad_(True)
        optimizer = optim.Adam([quat_G_C, trans_G_C], lr=0.1)
        best_loss, best_quat_G_C, best_trans_G_C = None, None, None

        ###################
        # optimize the G<-->C
        for it in range(num_iter):
            _, loss = compute_loss(self.trans_B_G, self.rot_B_G, self.trans_C_A, self.rot_C_A, trans_G_C, quat_G_C,
                                   rot_loss_w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if best_loss is None or best_loss > loss.item():
                best_loss = loss.item()
                best_quat_G_C = quat_G_C.detach().numpy()
                best_trans_G_C = trans_G_C[0].detach().numpy()

            print("iter = {:04d} loss = {}".format(it, loss.item()))

        best_rot_G_C, best_quat_G_C = quat2mat(torch.from_numpy(best_quat_G_C))
        best_rot_G_C, best_quat_G_C = best_rot_G_C[0].detach().numpy(), best_quat_G_C[0].detach().numpy()

        print("\n for B<-->C ")
        cmd = "rosrun tf static_transform_publisher " + str(float(best_trans_G_C[0])) + ' ' + \
              str(float(best_trans_G_C[1])) + ' ' + str(float(best_trans_G_C[2])) + ' ' + str(best_quat_G_C[1]) + ' '  \
              + str(best_quat_G_C[2]) + ' ' + str(best_quat_G_C[3]) + ' ' + str(best_quat_G_C[0]) + ' ' \
              + self.gripper_frame + ' ' + self.camera_frame + ' 10'
        print("Run Command")
        print(cmd)

        # plot the points for visualization
        if vis:
            rot_B_G_C = np.array([np.matmul(self.rot_B_G[i].numpy(), best_rot_G_C) for i in range(self.num_points)])
            trans_B_G_C_A = self.trans_B_G.numpy().reshape(-1,3) +\
                            np.array([np.matmul(self.rot_B_G[i].numpy(),best_trans_G_C.reshape(-1,3).T).T
                                      for i in range(self.num_points)]).reshape(-1,3) + \
                            np.array([np.matmul(rot_B_G_C[i], self.trans_C_A[i].t().numpy())
                                      for i in range(self.num_points)]).reshape(-1,3)
            ax = plt.axes(projection='3d')
            ax.scatter3D(trans_B_G_C_A[:,0], trans_B_G_C_A[:,1], trans_B_G_C_A[:,2])
            scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", marker = 'o')
            ax.legend([scatter1_proxy], ['Base to Ar from Gripper & Camera'], numpoints = 1)
            plt.show()


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

    args = parser.parse_args()
    get_camera_extrensics = GetCameraExtrensics(args)
    get_camera_extrensics.calibrate()