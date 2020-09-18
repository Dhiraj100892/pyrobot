from pyrobot.utils.util import try_cv2_import

cv2 = try_cv2_import()

import numpy as np
import skfmm
from numpy import ma

num_rots = 36


def get_mask(sx, sy, step_size):
    size = step_size * 2 + 1
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
                    step_size ** 2:
                mask[i, j] = 1
    return mask


def get_dist(sx, sy, step_size):
    size = step_size * 2 + 1
    mask = np.zeros((size, size)) + 1e-10
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
                    step_size ** 2:
                mask[i, j] = max(5, (((i + 0.5) - (size // 2 + sx)) ** 2 +
                                     ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
    return mask


class FMMPlanner():
    def __init__(self, traversable, step_size=5):
        self.step_size = step_size
        self.traversable = traversable

    def set_goal(self, goal):
        '''
        here basically calculating distance from goal, try to visualize dd to get more intution
        '''
        traversable_ma = ma.masked_values(self.traversable * 1, 0)
        goal_x, goal_y = int(goal[0]), int(goal[1])
        traversable_ma[goal_y, goal_x] = 0
        dd = skfmm.distance(traversable_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        return dd_mask

    def get_short_term_goal_2(self, state):
        state = [int(x) for x in state]
        # pad the map with
        # to handle corners pad the dist with step size and values equal to max
        dist = np.pad(self.fmm_dist, self.step_size,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)
        # take subset fo distance around the start, as its padded start should be corner instead of center
        subset = dist[state[0]:state[0] + 2 * self.step_size + 1,
                 state[1]:state[1] + 2 * self.step_size + 1]

        # find the index which has minimum distance
        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        # convert index from subset frame
        return (stg_x + state[0] - self.step_size) + 0.5, \
               (stg_y + state[1] - self.step_size) + 0.5

    def get_short_term_goal(self, state):
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        # for in grid around step_size x step_size == 1. other zeros, because state can be float not int
        mask = get_mask(dx, dy, self.step_size)

        # has Manhatten distance into dist into it
        dist_mask = get_dist(dx, dy, self.step_size)

        state = [int(x) for x in state]

        # to handle corners pad the dist with step size and values equal to max
        dist = np.pad(self.fmm_dist, self.step_size,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)

        # take the subset of dist map centered around start, it considers start as top left corner because we pad it in previsous step
        subset = dist[state[0]:state[0] + 2 * self.step_size + 1,
                 state[1]:state[1] + 2 * self.step_size + 1]

        assert subset.shape[0] == 2 * self.step_size + 1 and \
               subset.shape[1] == 2 * self.step_size + 1, \
            "Planning error: unexpected subset shape {}".format(subset.shape)

        subset *= mask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2
        # what this step is for
        subset -= subset[self.step_size, self.step_size]
        ratio1 = subset / dist_mask
        subset[ratio1 < -1.5] = 1

        trav = np.pad(self.traversable, self.step_size,
                      'constant', constant_values=0)

        subset_trav = trav[state[0]:state[0] + 2 * self.step_size + 1,
                      state[1]:state[1] + 2 * self.step_size + 1]
        traversable_ma = ma.masked_values(subset_trav * 1, 0)
        goal_x, goal_y = self.step_size, self.step_size
        traversable_ma[goal_y, goal_x] = 0
        dd = skfmm.distance(traversable_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        dd = ma.filled(dd, np.max(dd) + 1)
        subset_fmm_dist = dd

        subset_fmm_dist[subset_fmm_dist < 4] = 4.
        subset = subset / subset_fmm_dist
        subset[subset < -1.5] = 1
        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        if subset[stg_x, stg_y] > -0.0001:
            replan = True
        else:
            replan = False

        # add 0.5 so that you go to the center of the cell
        return (stg_x + state[0] - self.step_size) + 0.5, \
               (stg_y + state[1] - self.step_size) + 0.5, replan
