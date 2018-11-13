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
SEED = 72
agent = Agent(state_size=state_size, action_size=action_size, random_seed=SEED)
writer = tbx.SummaryWriter()


def ddpg(n_episodes=2000, max_t=1000):
    scores_deque = deque(maxlen=100)
    scores = []
    timesteps = 0
    tbx_counter = 0
    max_score = -np.Inf

    for i_episode in range(1, n_episodes + 1):
        state = env.reset(train_mode=True)[brain_name].vector_observations
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            obj = env.step(action)
            next_state = obj["ReacherBrain"].vector_observations
            reward = obj["ReacherBrain"].rewards[0]
            done = obj["ReacherBrain"].local_done[0]
            agent.step(state, action, reward, next_state, done)

            writer.add_scalar('data/reward', reward, global_step=tbx_counter)
            writer.add_scalar('data/action1', action[0][0], global_step=tbx_counter)
            writer.add_scalar('data/action2', action[0][1], global_step=tbx_counter)
            writer.add_scalar('data/action3', action[0][2], global_step=tbx_counter)
            writer.add_scalar('data/action4', action[0][3], global_step=tbx_counter)
            tbx_counter += 1

            state = next_state
            score += reward
            if done:
                break

        scores_deque.append(score)
        scores.append(score)
        ave_score = np.mean(scores_deque)
        print('\rEpisode {}\t[Score] Current: {:.2f}\tAverage: {:.2f}'.format(i_episode, score, ave_score))
        writer.add_scalar('data/score', score, i_episode)
        writer.add_scalar('data/ave_score', ave_score, i_episode)

        if ave_score > 30:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_v1.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_v1.pth')
            break
    return scores


scores = ddpg()
writer.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()