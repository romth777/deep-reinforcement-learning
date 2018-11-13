# -*- coding: utf-8 -*-
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
import logging
from datetime import datetime

from unityagents import UnityEnvironment
import numpy as np
import torch
import tensorboardX as tbx


from ddpg_agent import Agent, ReplayBuffer, OUNoise

env1_path = Path("./Reacher_Windows_x86_64_v1/Reacher.app")
env2_path = Path("./Reacher_Windows_x86_64_v2/Reacher.app")
env_path = str(env2_path.resolve())
env = UnityEnvironment(file_name=env_path)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
state_size = env_info.vector_observations.shape[1]
action_size = env_info.previous_vector_actions.shape[1]
NUM_AGENTS = 20
SEED = 71

def ddpg(n_episodes=120, max_t=1000, ave_length=100, noise_type="uniform"):
    scores_deque = deque(maxlen=ave_length)
    cp = CheckPoint()

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(NUM_AGENTS)
        for t in range(max_t):
            actions = agent.act(states, noise_type)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            scores += rewards

            if np.any(dones):
                break
        score = np.mean(scores)
        scores_deque.append(score)
        ave_score = np.mean(scores_deque)
        print('\rEpisode {}\t[Score] Current: {:.2f}\tAverage: {:.2f}'.format(i_episode, score, ave_score))
        if cp.is_maximum(ave_score):
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_v2.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_v2.pth')
        writer.add_scalar('data/score', score, i_episode)
        writer.add_scalar('data/ave_score', ave_score, i_episode)

    return scores_deque


class CheckPoint:
    def __init__(self):
        self.current = None
        self.target = None

    def is_maximum(self, data):
        if self.target is None:
            self.target = data
        else:
            self.current = data
            if self.current > self.target:
                return True
        return False


# for type in ["normal", "cauchy", "exponential", "gamma", "t", "uniform", "binomial", "beta", "chisquare"]:
for type in ["else"]:
    print(type)
    agent = Agent(state_size=state_size, action_size=action_size, num_agents=NUM_AGENTS, random_seed=SEED)
    writer = tbx.SummaryWriter()
    scores = ddpg(noise_type=type)
    writer.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()