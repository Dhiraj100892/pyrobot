# need to convert it to api
from pyrobot import Robot

import os
import open3d
from IPython import embed
import numpy as np
import sys
import matplotlib.pyplot as plt
from pyrobot.utils.util import try_cv2_import
import argparse
from scipy import ndimage
from copy import deepcopy as copy
import time

cv2 = try_cv2_import()
# for slam modules

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(BASE_AGENT_ROOT)
from slam.map_builder import MapBuilder as mb
from skimage.morphology import binary_closing, disk, binary_dilation, binary_erosion
from slam.fmm_planner import FMMPlanner
import slam.depth_utils as du


class Slam(object):
    def __init__(self, robot, map_size, resolution, robot_rad, agent_min_z, agent_max_z, save_folder='.tmp'):
        self.robot = robot
        self.robot_rad = robot_rad
        self.map_builder = mb(map_size_cm=map_size, resolution=resolution, agent_min_z=agent_min_z,
                              agent_max_z=agent_max_z)

        # initialize variable
        self.init_state = self.robot.base.get_state('odom')
        self.prev_bot_state = (0, 0, 0)
        self.col_map = np.zeros((self.map_builder.map.shape[0], self.map_builder.map.shape[1]))
        self.robot_loc_list_map = np.array([self.real2map(
            self.get_rel_state(self.robot.base.get_state('odom'), self.init_state)[:2])])
        self.map_builder.update_map(self.robot.camera.get_current_pcd(in_cam=False)[0],
                                    self.get_rel_state(self.robot.base.get_state('odom'), self.init_state))

        # for visualization purpose #
        self.save_folder = save_folder
        # to visualize robot heading
        triangle_scale = 0.5
        self.triangle_vertex = np.array([[0.0, 0.0],
                                         [-2.0, 1.0],
                                         [-2.0, -1.0]])
        self.triangle_vertex *= triangle_scale
        self.save_folder = os.path.join(save_folder, str(int(time.time())))
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
        self.start_vis = False
        self.vis_count = 0

        # rotate intial to get the whole map
        for _ in range(7):
            self.robot.base.go_to_relative((0,0,np.pi/4))
            self.map_builder.update_map(self.robot.camera.get_current_pcd(in_cam=False)[0],
                                        self.get_rel_state(self.robot.base.get_state('odom'), self.init_state))

    def set_goal(self, goal):
        """
        goal is 3 len tuple with position in real world in robot start frame
        :param goal:
        :return:
        """
        self.goal_loc = goal
        self.goal_loc_map = self.real2map(self.goal_loc[:2])

    def take_step(self, step_size):
        """
        step size in meter
        :param step_size:
        :return:
        """
        # explode the map by robot shape
        obstacle = self.map_builder.map[:, :, 1] >= 1.0
        selem = disk(self.robot_rad / self.map_builder.resolution)
        traversable = binary_dilation(obstacle, selem) != True

        # add robot collision map to traversable area
        unknown_region = self.map_builder.map.sum(axis=-1) < 1
        col_map_unknown = np.logical_and(self.col_map > 0.1, unknown_region)
        traversable = np.logical_and(traversable, np.logical_not(col_map_unknown))

        # call the planner
        self.planner = FMMPlanner(traversable,
                                  step_size=int(step_size / self.map_builder.resolution))

        # set the goal
        self.planner.set_goal(self.goal_loc_map)

        # get the short term goal
        self.robot_map_loc = self.real2map(
            self.get_rel_state(self.robot.base.get_state('odom'), self.init_state))
        self.stg = self.planner.get_short_term_goal((self.robot_map_loc[1], self.robot_map_loc[0]))

        # convert goal from map space to robot space
        stg_real = self.map2real([self.stg[1], self.stg[0]])
        print("stg = {}".format(self.stg))
        print("stg real = {}".format(stg_real))

        # convert stg real from init frame to global frame#
        stg_real_g = [0.0, 0.0]
        # 1) orient it to global frame
        stg_real_g[0], stg_real_g[1], _ = self.get_rel_state(
            (stg_real[0], stg_real[1], 0), (0.0, 0.0, -self.init_state[2]))

        # 2) add the offset
        stg_real_g[0] += self.init_state[0]
        stg_real_g[1] += self.init_state[1]

        self.robot_state = self.get_rel_state(self.robot.base.get_state('odom'), self.init_state)
        print("bot_state before executing action = {}".format(self.robot_state))

        # go to the location the robot
        exec = self.robot.base.go_to_absolute((stg_real_g[0],
                                             stg_real_g[1],
                                             np.arctan2(stg_real[1] - self.prev_bot_state[1],
                                                        stg_real[0] - self.prev_bot_state[0])
                                             + self.init_state[2]))

        print("bot_state after executing action = {}".format(
            self.get_rel_state(self.robot.base.get_state('odom'), self.init_state)))

        # if robot collides
        if not exec:
            # add obstacle in front of  cur location
            self.col_map += self.get_collision_map(
                self.get_rel_state(self.robot.base.get_state('odom'), self.init_state),
                obstacle_size=(100, 100))

        # update map
        self.map_builder.update_map(self.robot.camera.get_current_pcd(in_cam=False)[0],
                                    self.get_rel_state(self.robot.base.get_state('odom'), self.init_state))

        # update robot location list
        self.robot_loc_list_map = np.concatenate((self.robot_loc_list_map,
                                                np.array([
                                                    self.real2map(
                                                        self.get_rel_state(self.robot.base.get_state('odom'),
                                                                           self.init_state)[:2]
                                                    )])))
        self.prev_bot_state = self.robot_state

        # return whether its reachable or its already reached or not retched
        return self.stg[2]

    def real2map(self, loc):
        # converts real location to map location
        loc = np.array([loc[0], loc[1], 0])
        loc *= 100  # convert location to cm
        map_loc = du.transform_pose(loc, (self.map_builder.map_size_cm / 2.0,
                                          self.map_builder.map_size_cm / 2.0, np.pi / 2.0))
        map_loc /= self.map_builder.resolution
        map_loc = map_loc.reshape((3))
        return map_loc[:2]

    def map2real(self, loc):
        # converts map location to real location
        loc = np.array([loc[0], loc[1], 0])
        real_loc = du.transform_pose(loc,
                                     (-self.map_builder.map.shape[0] / 2.0,
                                      self.map_builder.map.shape[1] / 2.0, -np.pi / 2.0))
        real_loc *= self.map_builder.resolution  # to take into account map resolution
        real_loc /= 100  # to convert from cm to meter
        real_loc = real_loc.reshape((3))
        return real_loc[:2]

    def get_collision_map(self, state, obstacle_size=(20, 20)):

        # get the collision map for robot collison based on sensor reading
        col_map = np.zeros((self.map_builder.map.shape[0], self.map_builder.map.shape[1]))
        map_state = self.real2map([state[0], state[1]])
        map_state = [int(x) for x in map_state]
        center_map_state = self.real2map([0, 0])
        center_map_state = [int(x) for x in center_map_state]
        col_map[center_map_state[1] + 1: center_map_state[1] + 1 + obstacle_size[1],
        center_map_state[0] - int(obstacle_size[0] / 2): center_map_state[0] + int(obstacle_size[0] / 2)] = True

        # rotate col_map based on the state
        col_map = ndimage.rotate(col_map, -np.rad2deg(state[2]), reshape=False)

        # take crop around the center
        pad_len = 2 * max(obstacle_size)
        cropped_map = copy(col_map[center_map_state[1] - pad_len: center_map_state[1] + pad_len,
                           center_map_state[0] - pad_len: center_map_state[0] + pad_len])

        # make the crop value zero
        col_map = np.zeros((self.map_builder.map.shape[0], self.map_builder.map.shape[1]))

        # pad the col_map
        col_map = np.pad(col_map, pad_len)

        # paste the crop robot location shifted by pad len
        col_map[map_state[1] - pad_len + pad_len: map_state[1] + pad_len + pad_len,
        map_state[0] - pad_len + pad_len: map_state[0] + pad_len + pad_len] = cropped_map
        return col_map[pad_len:-pad_len, pad_len:-pad_len]

    def get_rel_state(self, cur_state, init_state):
        # get relative in global frame
        rel_X = cur_state[0] - init_state[0]
        rel_Y = cur_state[1] - init_state[1]
        # transfer from global frame to init frame
        R = np.array([[np.cos(init_state[2]), np.sin(init_state[2])],
                      [-np.sin(init_state[2]), np.cos(init_state[2])]])
        rel_x, rel_y = np.matmul(R, np.array([rel_X, rel_Y]).reshape(-1, 1))
        return rel_x[0], rel_y[0], cur_state[2] - init_state[2]

    def vis(self):
        if not self.start_vis:
            plt.figure(figsize=(20, 20))
            self.start_vis = True
        plt.clf()
        num_plots = 3

        # visualize RGB image
        plt.subplot(1, num_plots, 1)
        plt.imshow(self.robot.camera.get_rgb())

        # visualize Depth image
        plt.subplot(1, num_plots, 2)
        plt.imshow(self.robot.camera.get_depth())

        # visualize distance to goal & map, robot current location, goal, short term goal, robot path #
        plt.subplot(1, num_plots, 3)
        # distance to goal & map
        #plt.imshow(self.planner.fmm_dist, origin='lower')
        plt.imshow(self.map_builder.map[:, :, 1] >= 1.0, origin='lower')
        # goal
        plt.plot(self.goal_loc_map[0], self.goal_loc_map[1], 'y*')
        # short term goal
        plt.plot(self.stg[1], self.stg[0], 'b*')
        plt.plot(self.robot_loc_list_map[:, 0], self.robot_loc_list_map[:, 1], 'r--')

        # draw heading of robot
        robot_state = self.get_rel_state(self.robot.base.get_state('odom'), self.init_state)
        R = np.array([[np.cos(robot_state[2]), np.sin(robot_state[2])],
                      [-np.sin(robot_state[2]), np.cos(robot_state[2])]])
        global_tri_vertex = np.matmul(R.T, self.triangle_vertex.T).T
        map_global_tra_vertex = np.array(
            [self.real2map((x[0] + robot_state[0], x[1] + robot_state[1]))
             for x in global_tri_vertex])
        t1 = plt.Polygon(map_global_tra_vertex, color='red')
        plt.gca().add_patch(t1)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.draw()
        plt.savefig(os.path.join(self.save_folder, '{:04d}.jpg'.format(self.vis_count)))
        plt.pause(0.1)
        self.vis_count += 1


def main(args):

    if args.robot == 'habitat':
        # Please change this to match your habitat_sim repo's path
        path_to_habitat_scene = os.path.dirname(os.path.realpath(__file__))
        relative_path = "scenes/skokloster-castle.glb"

        common_config = dict(scene_path=os.path.join(path_to_habitat_scene, relative_path))
        robot = Robot("habitat", common_config=common_config)
    elif args.robot == 'locobot':
        robot = Robot("locobot")
        robot.camera.reset()

    slam = Slam(robot, args.map_size, args.resolution, args.robot_rad, args.agent_min_z, args.agent_max_z)
    slam.set_goal(args.goal)
    while True:
        slam.take_step(step_size=args.step_size)
        slam.vis()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for testing simple SLAM algorithm")
    parser.add_argument(
        "--robot", help="Name of the robot [locobot, habitat]", type=str, default='habitat'
    )
    parser.add_argument("--goal", help="goal the robot should reach", nargs='+', type=float)
    parser.add_argument("--map_size", help="lenght and with of map in cm", type=float, default=4000)
    parser.add_argument("--resolution", help="per pixel resolution of map in cm", type=float, default=5)
    parser.add_argument("--step_size", help="step size in cm", type=float, default=25)
    parser.add_argument("--robot_rad", help="robot radius in cm", type=float, default=25)
    parser.add_argument("--agent_min_z", help="agent min height in cm", type=float, default=5)
    parser.add_argument("--agent_max_z", help="robot max height in cm", type=float, default=70)

    args = parser.parse_args()
    main(args)