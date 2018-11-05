# -*- coding: utf-8 -*-
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
from multiprocessing import Manager

from unityagents import UnityEnvironment
import numpy as np
import torch

from ddpg_agent import Agents, Agent, ReplayBuffer

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
BATCH_SIZE = 128
SEED = 72
agents = [Agent(state_size=state_size, action_size=action_size, random_seed=2) for i in range(NUM_AGENTS)]

def ddpg(n_episodes=1000, max_t=300, print_every=100):
    memories = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, SEED)
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        states = env.reset(train_mode=True)[brain_name].vector_observations
        for agent in agents:
            agent.reset()
        score = 0
        for t in range(max_t):
            actions = []
            for i, agent in enumerate(agents):
                if i > states.shape[0] - 1:
                    i = states.shape[0] - 1
                action = agent.act(states[i].reshape(1, -1))
                actions.append(action.reshape(-1))
            actions = np.array(actions)
            obj = env.step(actions)
            next_states = obj["ReacherBrain"].vector_observations
            rewards = obj["ReacherBrain"].rewards
            dones = obj["ReacherBrain"].local_done
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                memories = agent.step_with_shared_memory(state, action, reward, next_state, done, memories)
            states = next_states
            score += sum(rewards) / len(rewards)
            if dones.count(True) > 0:
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_v2.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_v2.pth')
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