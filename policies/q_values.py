from base64 import encode
import numpy as np

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


def create_tiling_grid(low, high, bins=(10, 10), offsets=(0.0, 0.0)):
    """Define a uniformly-spaced grid that can be used for tile-coding a space.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins or tiles along each corresponding dimension.
    offsets : tuple
        Split points for each dimension should be offset by these values.

    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    # TODO: Implement this
    grid = np.array([np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] + offsets[dim] for dim in range(len(bins))])
    # print("Tiling: [<low>, <high>] / <bins> + (<offset>) => <splits>")
    # for l, h, b, o, splits in zip(low, high, bins, offsets, grid):
        # print("    [{}, {}] / {} + ({}) => {}".format(l, h, b, o, splits))
    return grid


def create_tilings(low, high, tiling_specs):
    """Define multiple tilings using the provided specifications.

    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    tiling_specs : list of tuples
        A sequence of (bins, offsets) to be passed to create_tiling_grid().

    Returns
    -------
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    """
    # TODO: Implement this
    return np.array([create_tiling_grid(low, high, bins, offsets) for bins, offsets in tiling_specs])


def discretize(sample, grid):
    """Discretize a sample as per given grid.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.

    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    # TODO: Implement this
    return np.array([int(np.digitize(s, g)) for s, g in zip(sample, grid)]) # apply along each dimension


def tile_encode(sample, tilings, flatten=False):
    """Encode given sample using tile-coding.

    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    tilings : list
        A list of tilings (grids), each produced by create_tiling_grid().
    flatten : bool
        If true, flatten the resulting binary arrays into a single long vector.

    Returns
    -------
    encoded_sample : list or array_like
        A list of binary vectors, one for each tiling, or flattened into one.
    """
    # TODO: Implement this
    encoded_sample = np.array([discretize(sample, grid) for grid in tilings])
    return np.concatenate(encoded_sample) if flatten else encoded_sample


class QTable:
    """Simple Q-table."""

    def __init__(self, state_size, action_size, init='zero'):
        """Initialize Q-table.

        Parameters
        ----------
        state_size : tuple
            Number of discrete values along each dimension of state space.
        action_size : int
            Number of discrete actions in action space.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.shape = (self.state_size + (self.action_size,))

        # TODO: Create Q-table, initialize all Q-values to zero
        # Note: If state_size = (9, 9), action_size = 2, q_table.shape should be (9, 9, 2)
        if init == 'zero':
            self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
        elif init == 'rand':
            self.q_table = np.random.normal(0, 0.01, (self.state_size + (self.action_size,))).astype(dtype=np.float64)
        elif init == 'opt':
            self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
            self.q_table.fill(1.0)
        elif init == 'pes':
            self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
            self.q_table.fill(-1.0)


class TiledQTable:
    """Composite Q-table with an internal tile coding scheme."""

    def __init__(self, low, high, tiling_specs, action_size, init):
        """Create tilings and initialize internal Q-table(s).

        Parameters
        ----------
        low : array_like
            Lower bounds for each dimension of state space.
        high : array_like
            Upper bounds for each dimension of state space.
        tiling_specs : list of tuples
            A sequence of (bins, offsets) to be passed to create_tilings() along with low, high.
        action_size : int
            Number of discrete actions in action space.
        """
        self.tilings = create_tilings(low, high, tiling_specs)
        self.state_sizes = [tuple(len(splits)+1 for splits in tiling_grid) for tiling_grid in self.tilings]
        self.action_size = action_size
        self.q_tables = [QTable(state_size, self.action_size, init) for state_size in self.state_sizes]
        # print("TiledQTable(): no. of internal tables = ", len(self.q_tables))

        self.shape = tuple((len(self.q_tables),)) + self.q_tables[0].shape


    def __getitem__(self, idx):
        """Get Q-value for given <state, action> pair.

        Parameters
        ----------
        state : array_like
            Vector representing the state in the original continuous space.
        action : int
            Index of desired action.

        Returns
        -------
        value : float
            Q-value of given <state, action> pair, averaged from all internal Q-tables.
        """
        state, action = idx

        # If action is a single value
        if isinstance(action, int) and len(state.shape) == 1:
            encoded_state = tile_encode(state, self.tilings)
            # Retrieve q-value for each tiling, and return their average
            value = 0.0
            for idx, q_table in zip(encoded_state, self.q_tables):
                    value += q_table.q_table[tuple(idx) + tuple([action])]
            value /= len(self.q_tables)
            return np.array(value)

        # Array operation
        else:
            encoded_states = []
            for batch in range(state.shape[0]):
                encoded_states.append(tile_encode(state[batch], self.tilings))

            values = [] * state.shape[0]
            action = action if not isinstance(action, slice) else [0]*state.shape[0]
            for encoded_state, act in zip(encoded_states, action):
                if not isinstance(action, list):
                    value = 0.0
                    for idx, q_table in zip(encoded_state, self.q_tables):
                        # print('not slice: ', tuple(idx) + tuple([act]), q_table.q_table[tuple(idx) + tuple([act])])
                        value += q_table.q_table[tuple(idx) + tuple([act])]
                    value /= len(self.q_tables)
                    value = [value]
                else:
                    value = np.zeros((1, self.action_size))
                    for idx, q_table in zip(encoded_state, self.q_tables):
                        # print('slice: ', q_table.q_table[tuple(idx)])
                        value += q_table.q_table[tuple(idx)]
                    value /= len(self.q_tables)
                values.append(value)
            return np.asarray(np.concatenate(values, axis=0))


    def update(self, state, action, delta):
        """Soft-update Q-value for given <state, action> pair to value.

        Instead of overwriting Q(state, action) with value, perform soft-update:
            Q(state, action) = alpha * value + (1.0 - alpha) * Q(state, action)

        Parameters
        ----------
        state : array_like
            Vector representing the state in the original continuous space.
        action : int
            Index of desired action.
        value : float
            Desired Q-value for <state, action> pair.
        alpha : float
            Update factor to perform soft-update, in [0.0, 1.0] range.
        """

        encoded_states = []
        for batch in range(state.shape[0]):
            encoded_states.append(tile_encode(state[batch], self.tilings))

        encoded_states = np.array(encoded_states)

        for encoded_state, act, d in zip(encoded_states, action, delta):
            # Update q-value for each tiling by update factor alpha
            for idx, q_table in zip(encoded_state, self.q_tables):
                value_ = q_table.q_table[tuple(idx) + tuple([act])]
                q_table.q_table[tuple(idx) + tuple([act])] = d + value_


class QLearningAgent:
    """Q-Learning agent that can act on a continuous state space by discretizing it."""

    def __init__(self, tq, epsilon, exploration_steps):
        """Initialize variables, create grid for discretization."""
        # Environment info
        self.tq = tq
        self.action_size = tq.action_size # 1-dimensional discrete action space

        if epsilon['strat'] == 'const':
            self.epsilon = EpsilonGreedy(exploration_steps, epsilon)
        elif epsilon['strat'] == 'linear':
            self.epsilon = LinearEpsilonGreedy(exploration_steps, epsilon)
        elif epsilon['strat'] == 'exp':
            self.epsilon = ExponentialEpsilonGreedy(exploration_steps, epsilon)
        else:
            assert False, 'Invalid epsilon strategy...'

    def update(self, state, action, delta):
        self.tq.update(state, action, delta)

    def action(self, state, step_count):
        Q_s = np.array([self.tq[state, action] for action in range(self.action_size)])
        # Pick the best action from Q table
        return self.epsilon.select_action(Q_s, step_count)


class QLearningAgent_Maxmin:
    """Q-Learning agent that can act on a continuous state space by discretizing it."""

    def __init__(self, tq, epsilon, exploration_steps):
        """Initialize variables, create grid for discretization."""
        # Environment info
        self.tq = tq        # list of function approximators (TC)
        self.action_size = tq[0].action_size # 1-dimensional discrete action space

        if epsilon['strat'] == 'const':
            self.epsilon = EpsilonGreedy(exploration_steps, epsilon)
        elif epsilon['strat'] == 'linear':
            self.epsilon = LinearEpsilonGreedy(exploration_steps, epsilon)
        elif epsilon['strat'] == 'exp':
            self.epsilon = ExponentialEpsilonGreedy(exploration_steps, epsilon)
        else:
            assert False, 'Invalid epsilon strategy...'

    def update(self, MMQ, active_estimator, delta):
        self.tq.update(state, action, delta)

    def action(self, state, step_count):
        Q_s = np.array([self.tq[state, action] for action in range(self.action_size)])
        # Pick the best action from Q table
        return self.epsilon.select_action(Q_s, step_count)

    def q_min(self, state, action):
        if action is None:
            # return array
            pass

        else:
            # return value
            pass