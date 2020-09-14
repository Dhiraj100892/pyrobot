from pyrobot import Robot

import os
import open3d
from IPython import embed
import numpy as np
import sys
import matplotlib.pyplot as plt
from pyrobot.utils.util import try_cv2_import

cv2 = try_cv2_import()
BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(BASE_AGENT_ROOT)
from slam.map_builder import MapBuilder as mb
from skimage.morphology import binary_closing, disk, binary_dilation, binary_erosion
from slam.fmm_planner import FMMPlanner
import slam.depth_utils as du
from scipy import ndimage
from copy import deepcopy as copy
import time


def real2map(loc, map):
    # converts real location to map location
    loc = np.array([loc[0], loc[1], 0])
    loc *= 100  # convert location to cm
    map_loc = du.transform_pose(loc, (map.map_size_cm / 2.0, map.map_size_cm / 2.0, np.pi / 2.0))
    map_loc /= map.resolution
    map_loc = map_loc.reshape((3))
    return map_loc[:2]


def map2real(loc, map):
    # converts map location to real location
    loc = np.array([loc[0], loc[1], 0])
    real_loc = du.transform_pose(loc,
                                 (-map.map.shape[0] / 2.0, map.map.shape[1] / 2.0, -np.pi / 2.0))
    real_loc *= map.resolution  # to take into account map resolution
    real_loc /= 100  # to convert from cm to meter
    real_loc = real_loc.reshape((3))
    return real_loc[:2]


def get_collision_map(state, map, obstacle_size=(20, 20)):
    '''

    '''
    # get the collision map for robot collison based on sensor reading
    col_map = np.zeros((map.map.shape[0], map.map.shape[1]))
    map_state = real2map([state[0], state[1]], map)
    map_state = [int(x) for x in map_state]
    center_map_state = real2map([0, 0], map)
    center_map_state = [int(x) for x in center_map_state]
    col_map[center_map_state[1] + 1: center_map_state[1] + 1 + obstacle_size[1],
    center_map_state[0] - int(obstacle_size[0] / 2): center_map_state[0] + int(obstacle_size[0] / 2)] = True
    # rotate colmap based on the state
    col_map = ndimage.rotate(col_map, -np.rad2deg(state[2]), reshape=False)

    # take crop around the center
    pad_len = 2 * max(obstacle_size)
    cropped_map = copy(col_map[center_map_state[1] - pad_len: center_map_state[1] + pad_len,
                       center_map_state[0] - pad_len: center_map_state[0] + pad_len])

    # make the crop value zero
    col_map = np.zeros((map.map.shape[0], map.map.shape[1]))

    # pad the col_map
    col_map = np.pad(col_map, pad_len)

    # paste the crop robot location shifted by pad len
    col_map[map_state[1] - pad_len + pad_len: map_state[1] + pad_len + pad_len,
    map_state[0] - pad_len + pad_len: map_state[0] + pad_len + pad_len] = cropped_map

    return col_map[pad_len:-pad_len, pad_len:-pad_len]


# Please change this to match your habitat_sim repo's path
path_to_habitat_scene = os.path.dirname(os.path.realpath(__file__))
relative_path = "scenes/skokloster-castle.glb"

common_config = dict(scene_path=os.path.join(path_to_habitat_scene, relative_path))
bot = Robot("habitat", common_config=common_config)

map_builder = mb(bot)
col_map = get_collision_map((11.487311108775993, -0.3395407211859647, 2.356193675930288),
                            map_builder, obstacle_size=(100, 100))

# fetch the point
pts, colors = bot.camera.get_current_pcd(in_cam=False)
map_builder.update_map()

goal_loc_real = (10, 10)
goal_loc_map = real2map(goal_loc_real, map_builder)
bot_loc_list_map = np.array([real2map(bot.base.get_state()[:2], map_builder)])

reached = False
count = 0
prev_bot_state = (0, 0, 0)
col_map = np.zeros((map_builder.map.shape[0], map_builder.map.shape[1]))
triangle_scale = 0.5
triangle_vertex = np.array([[0.0, 0.0],
                            [-2.0, 1.0],
                            [-2.0, -1.0]])
triangle_vertex *= triangle_scale

# plt.ion()
plt.figure(figsize=(20, 20))
save_folder = os.path.join('.tmp', str(int(time.time())))
if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

while not reached:
    # explode the map
    obstacle = map_builder.map[:, :, 1] >= 1.0
    # explode by robot shape
    robot_radius, resolution = 25, 5
    selem = disk(robot_radius / resolution)
    traversible = binary_dilation(obstacle, selem) != True

    unknown_region = map_builder.map.sum(axis=-1) < 1
    col_map_unknown = np.logical_and(col_map > 0.1, unknown_region)
    # traversible = np.logical_or(traversible, col_map_unknown)
    traversible = np.logical_and(traversible, np.logical_not(col_map_unknown))

    # call the planner
    delts_rot = 10  # in degree
    planner = FMMPlanner(traversible, 360 / delts_rot, step_size=5)

    # set the goal
    planner.set_goal(goal_loc_map)

    '''
    # get short term goal
    bot_map_loc = real2map(bot.base.get_state(), map_builder)
    stg = planner.get_short_term_goal((bot_map_loc[1], bot_map_loc[0]))

    # execute the goal on robot
    stg_real = map2real([stg[1], stg[0]], map_builder)
    print("stg = {}".format(stg))
    print("stg real = {}".format(stg_real))

    # first align the robot
    bot_state= bot.base.get_state()
    print("bot_state = {}".format(bot_state))
    exec = bot.base.go_to_absolute((bot_state[0], bot_state[1], np.arctan2(stg_real[1] - prev_bot_state[1],
                                                                           stg_real[0] - prev_bot_state[0])))
    if not exec:
        # add obstacle in front of  cur location
        col_map += get_collision_map(bot.base.get_state(), map_builder, obstacle_size=(100,100))

        # go to prev location
        bot.base.go_to_absolute(prev_bot_state)


        # take 360 view
        # rotate around
        for i in np.arange(np.pi/4, 2*np.pi, np.pi/4):
            bot.base.go_to_relative((0.0,0.0,i))
            map_builder.update_map()
        continue
    print("bot_state_after = {}".format(bot.base.get_state()))

    # take observation and update map
    map_builder.update_map()
    '''

    # then replan it again
    bot_map_loc = real2map(bot.base.get_state(), map_builder)
    stg = planner.get_short_term_goal_2((bot_map_loc[1], bot_map_loc[0]))

    # execute the goal on robot
    stg_real = map2real([stg[1], stg[0]], map_builder)
    print("stg = {}".format(stg))
    print("stg real = {}".format(stg_real))

    # go to the location the robot
    # prev_bot_state = bot_state
    bot_state = bot.base.get_state()
    print("bot_state = {}".format(bot.base.get_state()))
    exec = bot.base.go_to_absolute((stg_real[0], stg_real[1], np.arctan2(stg_real[1] - prev_bot_state[1],
                                                                         stg_real[0] - prev_bot_state[0])))
    if not exec:
        # add obstacle in front of  cur location
        col_map += get_collision_map(bot.base.get_state(), map_builder, obstacle_size=(100, 100))
        continue
    print("bot_state_after = {}".format(bot.base.get_state()))

    # TODO: Need to handle if robot get stuck somewhere

    # visualize
    if count % 1 == 0:
        plt.clf()

        num_plots = 3
        plt.subplot(1, num_plots, 1)
        plt.imshow(bot.camera.get_rgb())

        plt.subplot(1, num_plots, 2)
        plt.imshow(bot.camera.get_depth())

        plt.subplot(1, num_plots, 3)
        plt.imshow(planner.fmm_dist, origin='lower')
        plt.plot(goal_loc_map[0], goal_loc_map[1], 'y*')
        plt.plot(bot_map_loc[0], bot_map_loc[1], 'm.')
        plt.plot(bot_loc_list_map[:, 0], bot_loc_list_map[:, 1], 'r--')

        # plt.plot(stg[1], stg[0], 'b*')

        # draw heading of robot
        R = np.array([[np.cos(bot_state[2]), np.sin(bot_state[2])],
                      [-np.sin(bot_state[2]), np.cos(bot_state[2])]])
        global_tri_vertex = np.matmul(R.T, triangle_vertex.T).T
        map_global_tra_vertex = np.array(
            [real2map((x[0] + bot_state[0], x[1] + bot_state[1]), map_builder) for x in global_tri_vertex])
        # plt.plot(map_global_tra_vertex[:,0], map_global_tra_vertex[:,1], )
        t1 = plt.Polygon(map_global_tra_vertex, color='red')
        plt.gca().add_patch(t1)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.draw()
        plt.savefig(os.path.join(save_folder, '{:04d}.jpg'.format(count)))
        plt.pause(0.1)

    # update map

    map_builder.update_map()
    bot_loc_list_map = np.concatenate((bot_loc_list_map,
                                       np.array([real2map(bot.base.get_state()[:2], map_builder)])))
    prev_bot_state = bot_state
    # reached = stg[2]
    count += 1