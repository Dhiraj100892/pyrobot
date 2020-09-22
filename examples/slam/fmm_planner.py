import numpy as np
import skfmm
from numpy import ma

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

    def get_short_term_goal(self, state):
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