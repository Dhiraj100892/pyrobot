import numpy as np

import slam.depth_utils as du
from IPython import embed


class MapBuilder(object):
    def __init__(self, map_size_cm=4000, resolution=5, obs_thr=1,
                 agent_min_z=0.05, agent_max_z=0.7):
        self.map_size_cm = map_size_cm
        self.resolution = resolution
        self.obs_threshold = obs_thr
        self.z_bins = [agent_min_z, agent_max_z]

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

    def update_map(self, pcd, pose):
        # transfer points from base frame to global frame
        pcd = du.transform_pose(pcd, pose)

        # convert point from m to cm
        pcd = pcd * 100

        # for mapping we want global center to be at origin
        geocentric_pc_for_map = du.transform_pose(pcd,
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

    def reset_map(self, map_size):
        self.map_size_cm = map_size

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

    def get_map(self):
        return self.map
