# -*- coding: utf-8 -*-
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
import numpy as np
import torch
import tensorboardX as tbx

from ddpg_agent import Agent, ReplayBuffer

env1_path = Path("./Tennis_Windows_x86_64/Tennis.exe")
env_path = str(env1_path.resolve())
env = UnityEnvironment(file_name=env_path)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

SEED = 71


def ddpg(n_episodes=4000, ave_length=100, noise_type="uniform"):
    scores_deque = deque(maxlen=ave_length)
    cp = CheckPoint()

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = np.reshape(env_info.vector_observations, (1, 48))
        scores = np.zeros(num_agents)
        agent0.reset()
        agent1.reset()
        while True:
            action0 = agent0.act(states, noise_type)
            action1 = agent1.act(states, noise_type)
            actions = np.concatenate((action0, action1), axis=0)
            actions = np.reshape(actions, (1, 4))
            env_info = env.step(actions)[brain_name]
            next_states = np.reshape(env_info.vector_observations, (1, 48))
            rewards = env_info.rewards
            done = env_info.local_done
            agent0.step(states, actions, rewards[0], next_states, done, 0)
            agent1.step(states, actions, rewards[1], next_states, done, 1)
            states = next_states
            scores += rewards

            if np.any(done):
                break
        score = np.mean(scores)
        scores_deque.append(score)
        ave_score = np.mean(scores_deque)
        print('\rEpisode {}\t[Score] Current: {:.2f}\tAverage: {:.2f}'.format(i_episode, score, ave_score))
        if cp.is_maximum(ave_score):
            torch.save(agent0.actor_local.state_dict(), 'checkpoint_actor0_tennis.pth')
            torch.save(agent0.critic_local.state_dict(), 'checkpoint_critic0_tennis.pth')
            torch.save(agent1.actor_local.state_dict(), 'checkpoint_actor1_tennis.pth')
            torch.save(agent1.critic_local.state_dict(), 'checkpoint_critic1_tennis.pth')
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


agent0 = Agent(state_size=state_size, action_size=action_size, num_agents=1, random_seed=SEED)
agent1 = Agent(state_size=state_size, action_size=action_size, num_agents=1, random_seed=SEED)
writer = tbx.SummaryWriter()
scores = ddpg()
writer.close()
env.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

