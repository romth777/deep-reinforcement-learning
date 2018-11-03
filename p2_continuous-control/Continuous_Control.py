# -*- coding: utf-8 -*-

from unityagents import UnityEnvironment
import numpy as np

from pathlib import Path
env1_path = Path("./Reacher_Windows_x86_64_v1/Reacher.app")
env2_path = Path("./Reacher_Windows_x86_64_v2/Reacher.app")
env_path = str(env1_path.resolve())
env = UnityEnvironment(file_name=env_path)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]



import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import Agent

env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
state_size = env_info.vector_observations.size
action_size = env_info.previous_vector_actions.size
agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)


def ddpg(n_episodes=1000, max_t=300, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        state = env.reset(train_mode=True)[brain_name].vector_observations
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)[0]
            obj = env.step(action)
            next_state = obj["ReacherBrain"].vector_observations
            reward = obj["ReacherBrain"].rewards[0]
            done = obj["ReacherBrain"].local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
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