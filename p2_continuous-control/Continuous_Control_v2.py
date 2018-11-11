# -*- coding: utf-8 -*-
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
from multiprocessing import Manager

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
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 256
SEED = 72
agent = Agent(state_size=state_size, action_size=action_size, random_seed=SEED)
noises = [OUNoise(action_size, SEED + i) for i in range(NUM_AGENTS)]
[noise.reset() for noise in noises]
writer = tbx.SummaryWriter()


def ddpg(n_episodes=2000, max_t=1000, print_every=10):
    scores_deque = deque(maxlen=print_every)
    scores = []
    timesteps = 0
    tbx_counter = 0

    for i_episode in range(1, n_episodes + 1):
        states = env.reset(train_mode=True)[brain_name].vector_observations
        score = 0
        for t in range(max_t):
            actions = []
            for i in range(NUM_AGENTS):
                action = agent.act(states[i].reshape(1, -1)) + noises[i].sample()
                action = np.clip(action, -1, 1)
                actions.append(action.reshape(-1))
            actions = np.array(actions)
            obj = env.step(actions)
            next_states = obj["ReacherBrain"].vector_observations
            rewards = obj["ReacherBrain"].rewards
            dones = obj["ReacherBrain"].local_done
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.add(state, action, reward, next_state, done)
            timesteps += 1
            if timesteps > 20:
                agent.learn_many_times(train_counter=10)
                timesteps = 0
            states = next_states
            score += np.mean(rewards)

            if dones.count(True) > 0:
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_v2.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_v2.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

        writer.add_scalar('data/score', score, i_episode)

    return scores


scores = ddpg()
writer.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()