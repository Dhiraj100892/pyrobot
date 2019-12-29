# In this case the camera is attached to gripper of
# for comeplte calibratio, in this we will consider base(B), camera(C), gripper(G) and AR Tag(A) frames
# trasform from B<-->G and C<-->A is known
# need to figure out transform between G<-->C and B<-->A
# P_X_Y --> represent origin of Y frame in X frame of reference

import torch
from torch import optim
from IPython import embed
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib
from util import quat2mat, get_random_rot_trans, compute_loss
import argparse


class GetCameraExtrensics(object):

    def __init__(self, args):
        self.num_points = args.num_data
        self.noise_max = args.noise_max
        self.rot_loss_w =  args.rot_loss_w
        self.region_g, self.region_a, self.region_c = args.region_g, args.region_a, args.region_c
        self.vis = args.vis
        self.num_iter = args.num_iter

    def test(self):
        # generate data
        self.generate_data(self.num_points, self.noise_max, self.region_g, self.region_g, self.region_c)
        # optimize the parmas
        self.optimize(self.num_iter, self.rot_loss_w, self.vis)

    # data set generator
    def generate_data(self, num_points, noise_max, region_g, region_a, region_c):
        # we need to find out these values
        self.gt_rot_G_C, self.gt_trans_G_C = get_random_rot_trans(scale=region_a)
        self.gt_rot_B_A, self.gt_trans_B_A = get_random_rot_trans(scale=region_c)

        # generate data for training 
        self.rot_B_G, self.trans_B_G = get_random_rot_trans(num=num_points, scale =region_g)
        self.rot_C_A, self.trans_C_A = torch.zeros_like(self.rot_B_G), torch.zeros_like(self.trans_B_G)
        for i in range(num_points):
            rot_B_C = torch.mm(self.rot_B_G[i], self.gt_rot_G_C)
            self.rot_C_A[i] = torch.mm(rot_B_C.t(), self.gt_rot_B_A)
            trans_B_C = self.trans_B_G[i] + torch.mm(self.rot_B_G[i], self.gt_trans_G_C.t()).t()
            trans_C_A_in_B_frame = self.gt_trans_B_A - trans_B_C
            self.trans_C_A[i] = torch.mm(rot_B_C.t(), trans_C_A_in_B_frame.t()).t()

    # optimize the parameters
    def optimize(self, num_iter, rot_loss_w, vis):
        # start with some random guess
        quat_G_C = torch.rand(1,3).requires_grad_(True)
        trans_G_C = torch.rand(1,3).requires_grad_(True)
        optimizer = optim.Adam([quat_G_C, trans_G_C], lr=0.1)
        best_quat_G_C, best_trans_G_C, best_loss = None, None, None

        ###################
        # optimize the G<-->C
        for it in range(num_iter):
            _, loss = compute_loss(self.trans_B_G, self.rot_B_G, self.trans_C_A, self.rot_C_A, trans_G_C, quat_G_C
                                   , rot_loss_w)
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

        print("\n for G<-->C ")
        print(" org_rot = {} \n pred_rot = {}".format(self.gt_rot_G_C.numpy(), best_rot_G_C))
        print(" org_trans = {} \n pred_trans = {}".format(self.gt_trans_G_C.numpy(), best_trans_G_C))

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
    parser = argparse.ArgumentParser(description="Process args for calibration testing")
    parser.add_argument('--num_data', help='number of data points to generate for training', type=int, default=100)
    parser.add_argument('--noise_max', help='maximum noise to be added to location data points in meter', type=float,
                        default=0.01)
    parser.add_argument('--rot_loss_w', help='weight on rotational loss for optimizing the camera extrensic parameters',
                        type=float, default=0.0)
    parser.add_argument('--region_g', help='diameter of sphere in which location of gripper will be sampled wrt '
                                           'robot base', type=float, default=1.0)
    parser.add_argument('--region_a', help='diameter of sphere in which location of AR marker will be sampled wrt'
                                           ' gripper', type=float, default=0.1)
    parser.add_argument('--region_c', help='diameter of sphere in which location of camera will be sampled wrt'
                                           ' robot base', type=float, default=5.0)
    parser.add_argument('--vis', action='store_true', default=False, help='for visualizing data points after'
                                                                          ' calibration')
    parser.add_argument('--num_iter', help='number of iteration of optimization', type=int, default=1000)

    args = parser.parse_args()
    get_camera_extrensics = GetCameraExtrensics(args)
    get_camera_extrensics.test()