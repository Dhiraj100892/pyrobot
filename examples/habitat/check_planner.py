import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
import os
import sys

BASE_AGENT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(BASE_AGENT_ROOT)
from slam.fmm_planner import FMMPlanner

traversible = np.ones((800, 800)).astype(np.bool)
traversible[300:500, 300:500] = False

planner = FMMPlanner(traversible, 36)
goal_loc = (280,400)
planner.set_goal(goal_loc)

state = (400.5, 280.5)
bot_loc_list = [state]

plt.ion()
reached = False
while not reached:
    stg = planner.get_short_term_goal((state[1], state[0]))
    print("stg = {}".format(stg))

    # visualize
    plt.imshow(planner.fmm_dist, origin='lower')
    plt.plot(goal_loc[0], goal_loc[1], 'y*')
    plt.plot(state[0], state[1], 'm.')
    plt.draw()
    plt.pause(0.001)
    state = (stg[1], stg[0])
    bot_loc_list.append(state)
    reached = stg[2]