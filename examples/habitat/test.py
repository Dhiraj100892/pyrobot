from pyrobot import Robot
import time
import os
import open3d
import numpy as np
import sys
from IPython import embed

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(BASE_AGENT_ROOT)
import slam.depth_utils as du
import slam.rotation_utils as ru

# Please change this to match your habitat_sim repo's path
path_to_habitat_scene = os.path.dirname(os.path.realpath(__file__))
relative_path = "scenes/skokloster-castle.glb"

common_config = dict(scene_path=os.path.join(path_to_habitat_scene, relative_path))
bot = Robot("habitat", common_config=common_config)
pts, colors = bot.camera.get_current_pcd(in_cam=False)

def vis_point_cloud(pts, colors):
    temp_pts, temp_colors = bot.camera.get_current_pcd(in_cam=False)
    temp_pts = du.transform_pose(temp_pts, bot.base.get_state())
    pts = np.concatenate((pts, temp_pts))
    colors = np.concatenate((colors, temp_colors))

    # convert points to open3d
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(pts)
    pcd.colors = open3d.Vector3dVector(colors / 255.0)

    # for visualizing the origin
    coord = open3d.create_mesh_coordinate_frame(1, [0, 0, 0])

    # visualize point cloud
    open3d.visualization.draw_geometries([pcd, coord])

while True:
    bot.base.go_to_absolute((0.0,0.0,0.0))
    print(bot.base.get_full_state())
    pts, colors = bot.camera.get_current_pcd(in_cam=False)
    bot.base.go_to_absolute(((np.random.rand()-0.5)*10,
                             (np.random.rand() - 0.5) * 10,
                             (np.random.rand() - 0.5) * 2* np.pi))
    temp_pts, temp_colors = bot.camera.get_current_pcd(in_cam=False)
    temp_pts = du.transform_pose(temp_pts, bot.base.get_state())

    pts = np.concatenate((pts, temp_pts))
    colors = np.concatenate((colors, temp_colors))

    # convert points to open3d
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(pts)
    pcd.colors = open3d.Vector3dVector(colors / 255.0)

    # for visualizing the origin
    coord = open3d.create_mesh_coordinate_frame(1, [0, 0, 0])

    # visualize point cloud
    open3d.visualization.draw_geometries([pcd, coord])
    #embed()


# fetch the point
for d, r in zip(np.arange(0.1, 0.8, 0.1), np.arange(np.pi/4, 2*np.pi, np.pi/4)):
    #bot.base.go_to_absolute((3.0*d,0,r))

    bot.base.go_to_absolute(((np.random.rand() - 0.5) * 10,
                             (np.random.rand() - 0.5) * 10,
                             (np.random.rand() - 0.5) * 2 * np.pi))

    temp_pts, temp_colors = bot.camera.get_current_pcd(in_cam=False)
    temp_pts = du.transform_pose(temp_pts, bot.base.get_state())
    pts = np.concatenate((pts,temp_pts))
    colors = np.concatenate((colors, temp_colors))

# convert points to open3d point cloud object
pcd = open3d.PointCloud()
pcd.points = open3d.Vector3dVector(pts)
pcd.colors = open3d.Vector3dVector(colors / 255.0)

# for visualizing the origin
coord = open3d.create_mesh_coordinate_frame(1, [0, 0, 0])

# visualize point cloud
open3d.visualization.draw_geometries([pcd, coord])

# also test the speed
num_calls = 1000
time_list = []
for _ in range(num_calls):
    start_time = time.time()
    _, _ = bot.camera.get_current_pcd(in_cam=False)
    stop_time = time.time()
    time_list.append(stop_time-start_time)

print("Point in base frame speed = {:.1f} hz".format(float(num_calls) /sum(time_list) ))