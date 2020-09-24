from pyrobot import Robot
import time
import os
import open3d
import numpy as np
from scipy.spatial.transform import Rotation


def transform_pose(XYZ, current_pose):
    """
    Transforms the point cloud into geocentric frame to account for
    camera position
    Input:
        XYZ                     : ...x3
        current_pose            : camera position (x, y, theta (radians))
    Output:
        XYZ : ...x3
    """
    R = Rotation.from_euler("Z", current_pose[2]).as_matrix()
    XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape((-1,3))
    XYZ[:, 0] = XYZ[:, 0] + current_pose[0]
    XYZ[:, 1] = XYZ[:, 1] + current_pose[1]
    return XYZ

# Please change this to match your habitat_sim repo's path
path_to_habitat_scene = os.path.dirname(os.path.realpath(__file__))
relative_path = "scenes/skokloster-castle.glb"

common_config = dict(scene_path=os.path.join(path_to_habitat_scene, relative_path))
bot = Robot("habitat", common_config=common_config)

# fetch the point
pts_in_global, colors = bot.camera.get_current_pcd(in_cam=False)

# move the robot and transform captured pcd in global frame
for t, r in zip(np.arange(0.1, 0.8, 0.1), np.arange(np.pi/4, 2*np.pi, np.pi/4)):
    bot.base.go_to_absolute((t, t, r))
    pts_in_robot, temp_colors = bot.camera.get_current_pcd(in_cam=False)
    pts_in_global = np.concatenate((pts_in_global, transform_pose(pts_in_robot, bot.base.get_state())))
    colors = np.concatenate((colors, temp_colors))

# convert points to open3d point cloud object
pcd = open3d.PointCloud()
pcd.points = open3d.Vector3dVector(pts_in_global)
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