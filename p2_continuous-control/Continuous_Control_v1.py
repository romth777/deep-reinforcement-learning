# -*- coding: utf-8 -*-
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
import numpy as np
import torch
import tensorboardX as tbx

from ddpg_agent import Agent, ReplayBuffer, OUNoise

env1_path = Path("./Reacher_Windows_x86_64_v1/Reacher.app")
env2_path = Path("./Reacher_Windows_x86_64_v2/Reacher.app")
env_path = str(env1_path.resolve())
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
agents = Agent(state_size=state_size, action_size=action_size, random_seed=SEED)
noises = [OUNoise(action_size, SEED + i) for i in range(NUM_AGENTS)]
[noise.reset() for noise in noises]
writer = tbx.SummaryWriter()


def ddpg(n_episodes=2000, max_t=1000, print_every=10):
    scores_deque = deque(maxlen=print_every)
    scores = []
    timesteps = 0
    tbx_counter = 0

    for i_episode in range(1, n_episodes + 1):
        state = env.reset(train_mode=True)[brain_name].vector_observations
        agents.reset()
        score = 0
        for t in range(max_t):
            for i, noise in enumerate(noises):
                action = agents.act(state) + noise.sample()
                action = np.clip(action, -1, 1)
                obj = env.step(action)
                next_state = obj["ReacherBrain"].vector_observations
                reward = obj["ReacherBrain"].rewards[0]
                done = obj["ReacherBrain"].local_done[0]
                agents.add(state, action, reward, next_state, done)
                state = next_state

                writer.add_scalar('data/reward', reward, global_step=tbx_counter)
                writer.add_scalar('data/action1', action[0][0], global_step=tbx_counter)
                writer.add_scalar('data/action2', action[0][1], global_step=tbx_counter)
                writer.add_scalar('data/action3', action[0][2], global_step=tbx_counter)
                writer.add_scalar('data/action4', action[0][3], global_step=tbx_counter)
                tbx_counter += 1

                timesteps += 1
                if timesteps > 20:
                    agents.learn_many_times(train_counter=10)
                    timesteps = 0
                score += reward
                if done:
                    break

        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        torch.save(agents.actor_local.state_dict(), 'checkpoint_actor_v1.pth')
        torch.save(agents.critic_local.state_dict(), 'checkpoint_critic_v1.pth')

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