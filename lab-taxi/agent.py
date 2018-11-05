import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=0.8, gamma=0.99, eps=None):
        """ Initialize agent.
        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.policy_s = np.ones(6)
        self.i_episode = 1
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state):
        """ Given the state, select an action.
        Params
        ======
        - state: the current state of the environment
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.policy_s = self.epsilon_greedy_probs(self.Q[state])
        return np.random.choice(np.arange(self.nA), p=self.policy_s)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.sarsa(state, action, reward, next_state, done)

    def update_Q(self, Qsa, Qsa_next, reward):
        return Qsa + (self.alpha * (reward + (self.gamma * Qsa_next) - Qsa))

    def epsilon_greedy_probs(self, Q_s):
        epsilon = 1.0 / self.i_episode
        if self.eps is not None:
            epsilon = self.eps
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s

    def set_i_episode(self, i_episode):
        self.i_episode = i_episode

    # alpha=0.10498944841841751, gamma=0.999, eps=None / Episode 20000/20000 || Best average reward 9.516
    def sarsa(self, state, action, reward, next_state, done):
        if not done:
            policy_s = self.epsilon_greedy_probs(self.Q[next_state])
            next_action = np.random.choice(np.arange(self.nA), p=policy_s)
            self.Q[state][action] = self.update_Q(self.Q[state][action], self.Q[next_state][next_action], reward)
            state = next_state
            action = next_action
        if done:
            self.Q[state][action] = self.update_Q(self.Q[state][action], 0, reward)

    # alpha=0.8487094653853746, gamma=0.999, eps=None / Episode 20000/20000 || Best average reward 9.392
    def expected_sarsa(self, state, action, reward, next_state, done):
        self.Q[state][action] = self.update_Q(self.Q[state][action], np.dot(self.Q[next_state], self.policy_s), reward)

    # alpha=0.4368600991691859, gamma=0.999, eps=None / Episode 20000/20000 || Best average reward 9.687
    def sarsa_max(self, state, action, reward, next_state, done):
        self.Q[state][action] = self.update_Q(self.Q[state][action], np.max(self.Q[next_state]), reward)
