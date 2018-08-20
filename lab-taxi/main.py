from agent import Agent
from monitor import interact
import gym
import numpy as np
from hyperopt import hp, tpe, Trials, fmin

env = gym.make('Taxi-v2')
agent = Agent(alpha=0.85, eps=None, gamma=0.999)
avg_rewards, best_avg_reward = interact(env, agent)
