# -*- coding: utf-8 -*-
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
import numpy as np
import torch

from ddpg_agent import Agent, ReplayBuffer, OUNoise

env1_path = Path("./Reacher_Windows_x86_64_v1/Reacher.app")
env2_path = Path("./Reacher_Windows_x86_64_v2/Reacher.app")
env_path = str(env1_path.resolve())
env = UnityEnvironment(file_name=env_path)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
state_size = env_info.vector_observations.shape[1]
action_size = env_info.previous_vector_actions.shape[1]
NUM_AGENTS = 20
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
SEED = 72
actor_path = Path("./checkpoint_actor_v1.pth")
critic_path = Path("./checkpoint_critic_v1.pth")

agents = [Agent(state_size=state_size, action_size=action_size, random_seed=2)]
agents[0].load_weights(actor_path, critic_path)


def ddpg(n_episodes=100, max_t=300, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        state = env.reset(train_mode=False)[brain_name].vector_observations
        for agent in agents:
            agent.reset()
        score = 0
        for t in range(max_t):
            for i, agent in enumerate(agents):
                action = agent.act(state)
                obj = env.step(action)
                next_state = obj["ReacherBrain"].vector_observations
                reward = obj["ReacherBrain"].rewards[0]
                done = obj["ReacherBrain"].local_done[0]
                state = next_state
                score += reward
                if done:
                    break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores


scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()