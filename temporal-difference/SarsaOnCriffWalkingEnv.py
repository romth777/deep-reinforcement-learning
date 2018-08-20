# -*- coding: utf-8 -*-
import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt

env = gym.make('CliffWalking-v0')
print(env.action_space)
print(env.observation_space)
# define the optimal state-value function
V_opt = np.zeros((4,12))
V_opt[0:13][0] = -np.arange(3, 15)[::-1]
V_opt[0:13][1] = -np.arange(3, 15)[::-1] + 1
V_opt[0:13][2] = -np.arange(3, 15)[::-1] + 2
V_opt[3][0] = -13

plot_values(V_opt)