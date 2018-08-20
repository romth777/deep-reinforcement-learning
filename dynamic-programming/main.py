import sys
import numpy as np
from collections import defaultdict, deque
from frozenlake import FrozenLakeEnv

env = FrozenLakeEnv()

# print the state space and action space
print(env.observation_space)
print(env.action_space)

# print the total number of states and actions
print(env.nS)
print(env.nA)


def update_V(Vs, Vs_next, reward, alpha, gamma):
    return Vs + (alpha *(reward + (gamma * Vs_next) - Vs))


def epsilon_greedy_probs(env, Q_s, i_episode, eps=None):
    epsilon = 1.0 / i_episode
    if eps is not None:
        epsilon = eps
    policy_s = np.ones(env.nA) * epsilon / env.nA
    policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / env.nA)
    return policy_s


def policy_evaluation(env, policy, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)

    ## TODO: complete the function
    num_episodes = 20000
    plot_every = 100
    tmp_scores = deque(maxlen=plot_every)
    scores = deque(maxlen=num_episodes)

    alpha = 0.8
    for i_episode in range(1, num_episodes):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        score = 0
        state = env.reset()
        while True:
            policy_s = policy[state]
            action = np.random.choice(np.arange(env.nA), p=policy_s)
            next_state, reward, done, info = env.step(action)
            score += reward
            V[state] = update_V(V[state], np.max(V[next_state]), reward, alpha, gamma)
            state = next_state
            if done:
                tmp_scores.append(score)
                break
        if (i_episode % plot_every == 0):
            scores.append(np.mean(tmp_scores))

    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))
    return V

random_policy = np.ones([env.nS, env.nA]) / env.nA

from plot_utils import plot_values

# evaluate the policy
V = policy_evaluation(env, random_policy)

plot_values(V)