from base64 import encode
import numpy as np
from .tilecoding import TileCoder
import math

class BaseExploration(object):
  # Base class for agent exploration strategies.
  def __init__(self, exploration_steps, epsilon):
      self.exploration_steps = exploration_steps

  def select_action(self, q_values):
    raise NotImplementedError("To be implemented")


class EpsilonGreedy(BaseExploration):
  '''
  Implementation of epsilon greedy exploration strategy
  '''
  def __init__(self, exploration_steps, epsilon):
    super().__init__(exploration_steps, epsilon)
    self.epsilon = epsilon['start']

  def select_action(self, q_values, step_count):
    if np.random.rand() < self.epsilon or step_count <= self.exploration_steps:
      action = np.random.randint(0, len(q_values))
    else:
      if np.sum(np.where(q_values == q_values.max(), 1, 0)) == 1:
        action = np.argmax(q_values)
      # If we don't, randomly choose one of the maximum values
      else:
        action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
    return action


class LinearEpsilonGreedy(BaseExploration):
  '''
  Implementation of linear decay epsilon greedy exploration strategy
  '''
  def __init__(self, exploration_steps, epsilon):
    super().__init__(exploration_steps, epsilon)
    self.inc = (epsilon['end'] - epsilon['start']) / epsilon['steps']
    self.start = epsilon['start']
    self.end = epsilon['end']
    if epsilon['end'] > epsilon['start']:
      self.bound = min
    else:
      self.bound = max

  def select_action(self, q_values, step_count):
    self.epsilon = self.bound(self.start + step_count * self.inc, self.end)
    if np.random.rand() < self.epsilon or step_count <= self.exploration_steps:
      action = np.random.randint(0, len(q_values))
    else:
      if np.sum(np.where(q_values == q_values.max(), 1, 0)) == 1:
        action = np.argmax(q_values)
      # If we don't, randomly choose one of the maximum values
      else:
        action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
    return action


class ExponentialEpsilonIntervalGreedy(BaseExploration):
  '''
  Implementation of exponential decay epsilon greedy exploration strategy:
    epsilon = bound(epsilon_end, epsilon_start * (decay ** step//interval))
  '''
  def __init__(self, exploration_steps, epsilon):
    super().__init__(exploration_steps, epsilon)
    self.decay = epsilon['decay']
    self.start = epsilon['start']
    self.interval = epsilon['steps']
    self.end = epsilon['end']
    if epsilon['end'] > epsilon['start']:
      self.bound = min
    else:
      self.bound = max

  def select_action(self, q_values, step_count):
    self.epsilon = self.bound(self.start * math.pow(self.decay, step_count//self.interval), self.end)
    if np.random.rand() < self.epsilon or step_count <= self.exploration_steps:
      action = np.random.randint(0, len(q_values))
    else:
      if np.sum(np.where(q_values == q_values.max(), 1, 0)) == 1:
        action = np.argmax(q_values)
      # If we don't, randomly choose one of the maximum values
      else:
        action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
    return action


class ExponentialEpsilonGreedy(BaseExploration):
  '''
  Implementation of exponential decay epsilon greedy exploration strategy:
    epsilon = bound(epsilon_end, epsilon_start * (decay ** step))
  '''
  def __init__(self, exploration_steps, epsilon):
    super().__init__(exploration_steps, epsilon)
    self.decay = epsilon['decay']
    self.start = epsilon['start']
    self.end = epsilon['end']
    if epsilon['end'] > epsilon['start']:
      self.bound = min
    else:
      self.bound = max

  def select_action(self, q_values, step_count):
    self.epsilon = self.bound(self.start * math.pow(self.decay, step_count), self.end)
    if np.random.rand() < self.epsilon or step_count <= self.exploration_steps:
      action = np.random.randint(0, len(q_values))
    else:
      if np.sum(np.where(q_values == q_values.max(), 1, 0)) == 1:
        action = np.argmax(q_values)
      # If we don't, randomly choose one of the maximum values
      else:
        action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
    return action


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


class GreedyPolicy(Policy):
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


class EGPolicy(Policy):
    def __init__(self, Q, epsilon, exploration_steps):
        """Epsilon Greedy Policy"""
        self.Q = Q
        if epsilon['strat'] == 'const':
            self.epsilon = EpsilonGreedy(exploration_steps, epsilon)
        elif epsilon['strat'] == 'linear':
            self.epsilon = LinearEpsilonGreedy(exploration_steps, epsilon)
        elif epsilon['strat'] == 'exp':
            self.epsilon = ExponentialEpsilonGreedy(exploration_steps, epsilon)
        elif epsilon['strat'] == 'exp_interval':
            self.epsilon = ExponentialEpsilonIntervalGreedy(exploration_steps, epsilon)
        else:
            assert False, 'Invalid epsilon strategy...'

        self.probs = np.zeros(self.Q.shape)
        for state in range(self.Q.shape[0]):
            self.probs[state, np.argmax(self.Q[state, :])] = 1

    def update(self, Q):
        self.Q = Q
        self.probs = np.zeros(self.Q.shape)
        for state in range(self.Q.shape[0]):
            self.probs[state, np.argmax(self.Q[state, :])] = 1

    def action_prob(self, state, action=None):
        return self.probs[state, action]

    def action(self, state, step_count):
        return self.epsilon.select_action(self.Q[state, :], step_count)


class QLearningAgent:
    """Q-Learning agent that can act on a continuous state space by discretizing it."""

    def __init__(self, T, Q, epsilon, exploration_steps):
        """Initialize variables, create grid for discretization."""
        # Environment info
        self.tq = T
        self.w = Q

        if epsilon['strat'] == 'const':
            self.epsilon = EpsilonGreedy(exploration_steps, epsilon)
        elif epsilon['strat'] == 'linear':
            self.epsilon = LinearEpsilonGreedy(exploration_steps, epsilon)
        elif epsilon['strat'] == 'exp':
            self.epsilon = ExponentialEpsilonGreedy(exploration_steps, epsilon)
        elif epsilon['strat'] == 'exp_interval':
            self.epsilon = ExponentialEpsilonIntervalGreedy(exploration_steps, epsilon)
        else:
            assert False, 'Invalid epsilon strategy...'

    def action(self, state, step_count):
        Q_s = np.sum(self.w[self.tq[state]], axis=0)
        # Pick the best action from Q table
        return self.epsilon.select_action(Q_s, step_count)


def update(Q, state, action, delta, T, n_tiles):
    w = Q
    for idx, act in enumerate(action):
        s = T[state[idx]]
        w[s, act] += delta[idx]/n_tiles
    return w


def S(state, Q, T):
    state = np.array([T[s] for s in state])
    Q_s = np.sum(Q[state], axis=1)
    return Q_s

<<<<<<< HEAD
    def action(self, state):
        if np.random.random(1)[0] < self.epsilon:
            return np.random.randint(0, self.Q.shape[-1])
        else:
            if np.sum(np.where(self.probs[state, :] == self.probs[state, :].max(), 1, 0)) == 1:
                return np.argmax(self.probs[state, :])
            else:
                return np.random.choice(np.flatnonzero(self.probs[state, :] == self.probs[state, :].max()))
=======

def SA(state, action, Q, T):
    state = np.array([T[s] for s in state])
    Q_ret = np.zeros((state.shape[0]))
    idx = np.arange(len(action))
    Q_ret[idx] = np.sum(Q[state], axis=1)[idx, action]
    return Q_ret
>>>>>>> d9bb02a9c3c9783d399f23a1d31eea0c781030d8
