import numpy as np

import slam.depth_utils as du
from IPython import embed


class MapBuilder(object):
    def __init__(self, robot, params=None):
        self.robot = robot
        self.params = params

        self.vision_range = 100

        self.map_size_cm = 4000
        self.resolution = 5

        self.du_scale = 1
        self.visualize = 0
        self.obs_threshold = 1

        self.agent_height = 0.88

        agent_min_z = 0.05
        agent_max_z = 0.7
        self.z_bins = [agent_min_z, agent_max_z]

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

    def update_map(self):
        '''
        with np.errstate(invalid="ignore"):
            depth[depth > self.vision_range * self.resolution] = np.NaN

        point_cloud = du.get_point_cloud_from_z(depth, self.camera_matrix, \
                                                scale=self.du_scale)

        agent_view = du.transform_camera_view(point_cloud,
                                              self.agent_height,
                                              self.agent_view_angle)

        shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
        agent_view_centered = du.transform_pose(agent_view, shift_loc)

        agent_view_flat = du.bin_points(
            pcd_in_base_frame_cm,
            self.vision_range,
            self.z_bins,
            self.resolution)

        agent_view_cropped = agent_view_flat[:, :, 1]
        #cv2.imshow("AV", agent_view_cropped)
        #cv2.waitKey(1)

        agent_view_cropped = agent_view_cropped / self.obs_threshold
        agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        agent_view_cropped[agent_view_cropped < 0.5] = 0.0

        agent_view_explored = agent_view_flat.sum(2)
        agent_view_explored[agent_view_explored > 0] = 1.0
        '''
        pcd, _ = self.robot.camera.get_current_pcd(in_cam=False)
        pcd_in_base_frame_cm = pcd * 100
        #geocentric_pc = du.transform_pose(pcd_in_base_frame_cm, self.robot.base.get_state())

        # for mapping we want global center to be at origin
        geocentric_pc_for_map = du.transform_pose(pcd_in_base_frame_cm,
                                                  (self.map_size_cm/2.0, self.map_size_cm/2.0, np.pi/2.0))
        geocentric_flat = du.bin_points(
            geocentric_pc_for_map,
            self.map.shape[0],
            self.z_bins,
            self.resolution)

        self.map = self.map + geocentric_flat

        map_gt = self.map[:, :, 1] / self.obs_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        explored_gt = self.map.sum(2)
        explored_gt[explored_gt > 1] = 1.0
        return map_gt

    def get_st_pose(self, current_loc):
        loc = [- (current_loc[0] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) / \
               (self.map_size_cm // (self.resolution * 2)),
               - (current_loc[1] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) / \
               (self.map_size_cm // (self.resolution * 2)),
               90 - np.rad2deg(current_loc[2])]
        return loc

    def reset_map(self, map_size):
        self.map_size_cm = map_size

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

    def get_map(self):
        return self.map
