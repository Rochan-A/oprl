import numpy as np


class Policy(object):
    def action_prob(self, state: int, action: int) -> float:
        """
        input:
            state, action
        return:
            \pi(a|s)
        """
        raise NotImplementedError()

    def action(self, state: int) -> int:
        """
        input:
            state
        return:
            action
        """
        raise NotImplementedError()


class RandomPolicy(Policy):
    def __init__(self, nA, rng, p=None):
        self.p = p if p is not None else np.array([1 / nA] * nA)
        self.rng = rng

    def action_prob(self, state, action=None):
        return self.p[action]

    def action(self, state):
        return self.rng.random.choice(len(self.p), p=self.p)


class GreedyPolicy(object):
    def __init__(self, Q):
        """Greedy Policy"""
        self.Q = Q
        self.probs = np.zeros(self.Q.shape)
        for state in range(self.Q.shape[0]):
            self.probs[state, np.argmax(self.Q[state, :])] = 1

    def update(self, Q):
        self.Q = Q
        self.probs = np.zeros(self.Q.shape)
        for state in range(self.Q.shape[0]):
            self.probs[state, np.argmax(self.Q[state, :])] = 1

    def action_prob(self, state, action):
        return self.probs[state, action]

    def action(self, state):
        return np.argmax(self.Q[state, :])


class EGPolicy(object):
    def __init__(self, Q, epsilon):
        """Epsilon Greedy Policy"""
        self.Q = Q
        self.epsilon = epsilon
        self.probs = np.zeros(self.Q.shape)
        for state in range(self.Q.shape[0]):
            self.probs[state, np.argmax(self.Q[state, :])] = 1

    def update(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
        self.probs = np.zeros(self.Q.shape)
        for state in range(self.Q.shape[0]):
            self.probs[state, np.argmax(self.Q[state, :])] = 1

    def action_prob(self, state, action=None):
        return self.probs[state, action]

    def action(self, state):
        if np.random.random(1)[0] < self.epsilon:
            return np.random.randint(0, self.Q.shape[-1])
        else:
            if np.sum(np.where(self.Q[state, :] == self.Q[state, :].max(), 1, 0)) == 1:
                return np.argmax(self.Q[state, :])
            else:
                return np.random.choice(np.flatnonzero(self.Q[state, :] == self.Q[state, :].max()))


class EGPolicyExtra(object):
    def __init__(self, Q, epsilon):
        """Epsilon Greedy Policy"""
        self.Q = Q
        self.epsilon = epsilon
        self.probs = np.zeros(self.Q.shape)
        for state in range(self.Q.shape[0]):
            if np.sum(self.Q[state, :]) != 0:
                self.probs[state, :] = self.Q[state, :]/np.sum(self.Q[state, :])
            else:
                self.probs[state, :] = 1/self.Q.shape[1]


    def update(self, Q, epsilon):
        self.Q = Q
        self.epsilon = epsilon
        self.probs = np.zeros(self.Q.shape)
        for state in range(self.Q.shape[0]):
            if np.sum(self.Q[state, :]) != 0:
                self.probs[state, :] = self.Q[state, :]/np.sum(self.Q[state, :])
            else:
                self.probs[state, :] = 1/self.Q.shape[1]

    def action_prob(self, state, action=None):
        return self.probs[state, action]

    def action(self, state):
        if np.random.random(1)[0] < self.epsilon:
            return np.random.randint(0, self.Q.shape[-1])
        else:
            if np.sum(np.where(self.probs[state, :] == self.probs[state, :].max(), 1, 0)) == 1:
                return np.argmax(self.probs[state, :])
            else:
                return np.random.choice(np.flatnonzero(self.probs[state, :] == self.probs[state, :].max()))
