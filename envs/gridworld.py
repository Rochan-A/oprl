from typing import List
import numpy as np
import copy
import gym

import gym_minigrid
from gym_minigrid.wrappers import *

from enum import IntEnum
from .mdp import EnvSpec


# Map of object type to integers
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,          # can't pass through
    "rand_r": 3,        # Stochastic reward
    "one_r": 4,         # Single time reward **** Not Implemented ****
    "rand_t": 5,        # Stochastic transition
    "hack": 6,          # Hack state, unlimited reward
    "bank": 7,          # **** Not Implemented ****
    "goal": 8,          # Episode terminates
    "lava": 9,          # Episode terminates
    "agent": 10,
}

RAND_R_MU = -0.1
RAND_R_STD = 1

# Map of rewards landing on specific position
REWARDS = {
    0: 0,
    1: 0,
    2: 0,
    3: RAND_R_MU,
    4: 1,   # **** Not Implemented ****
    5: 0,
    6: 1,
    7: 0,   # **** Not Implemented ****
    8: 1,
    9: -1,
    10: 0,
    'timeout': -1
}


# Terminal states
TERMINAL = [OBJECT_TO_IDX['lava'], OBJECT_TO_IDX['goal']]


class DistanceBonus(gym.core.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to get closer to the goal.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        agent = env.agent_pos
        goal = env.goal_pos[0]

        bonus = math.sqrt(
            (agent[0] - goal[0])**2 + (agent[1] - goal[1])**2
            )
        reward -= bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ActionBonus(gym.core.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class StateBonus(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = (tuple(env.agent_pos))

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class DelayedReward(gym.core.Wrapper):
    """Env Wrapper to delay the reward"""

    def __init__(self, env, delay):
        super().__init__(env)
        self.env = env
        self.delay_amt = self.env.max_steps+1 if delay == 'inf' else delay

    def step(self, action):
        """Returns Transition probability vector of len nS"""

        obs, reward, done, info = self.env.step(action)

        # buffer reward
        if done == False:
            self.rew.append(reward)
            self.time.append(0)
            self.time = [t+1 for t in self.time]
            indices = [i for i, x in enumerate(self.time) if x == self.delay_amt]
            reward = 0
            for idx in indices:
                reward += self.rew.pop(idx)
                _ = self.time.pop(idx)

        # Clear out reward buffer
        elif done == True:
            reward += sum(self.rew)

        return obs, reward, done, info

    def reset(self, **kwargs):
        """Returns reward vector of len nS"""
        self.rew, self.time = [], []
        return self.env.reset(**kwargs)


class NoisyReward(gym.core.Wrapper):
    """
    Wrapper which adds gaussian noise to the reward at each step.
    """

    def __init__(self, env, rng, std=1):
        super().__init__(env)
        self.std = std
        self.rng = rng

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward += np.random.normal(0, self.std)

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class MapsEnvModel(gym.core.Wrapper):
    """Env Wrapper to query model ie. p(s, a, s')"""

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def TD(self, state, action):
        """Returns Transition probability vector of len nS"""

        # convert state index to position on grid
        pos = self.S[state]

        # prob. vector
        res = np.zeros((self.nS))

        # skip if current state is terminal, or agent can't transition out from
        # there i.e., lava or goal
        if (
            self.state[pos[0], pos[1]] != OBJECT_TO_IDX["lava"]
            and self.state[pos[0], pos[1]] != OBJECT_TO_IDX["goal"]
            and self.state[pos[0], pos[1]] != OBJECT_TO_IDX["wall"]
        ):

            # Move left
            if action == self.actions.left:
                y, x = pos[0], pos[1] - 1

            # Move right
            elif action == self.actions.right:
                y, x = pos[0], pos[1] + 1

            # Move up
            elif action == self.actions.up:
                y, x = pos[0] - 1, pos[1]

            # Move down
            elif action == self.actions.down:
                y, x = pos[0] + 1, pos[1]

            else:
                assert False, "unknown action"

            # If random transition
            if self.state[pos[0], pos[1]] == OBJECT_TO_IDX["rand_t"]:

                # find all locations we can transition to
                valid_pos = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if self.state[pos[0] + dy, pos[1] + dx] != OBJECT_TO_IDX["wall"]:
                            valid_pos.append([pos[0] + dy, pos[1] + dx])

                if [y, x] in valid_pos:
                    res[np.where((self.S == [y, x]).all(axis=1))[0]] = 1/len(valid_pos)
                else:
                    res[np.where((self.S == [y, x]).all(axis=1))[0]] = 0

            # If we transitioned to a wall, cant go there, will 100% stay here
            elif self.state[y, x] == OBJECT_TO_IDX["wall"]:
                res[np.where((self.S == [y, x]).all(axis=1))[0]] = 0
                res[state] = 1

            else:
                res[np.where((self.S == [y, x]).all(axis=1))[0]] = 1

        return res

    def R(self, state, action):
        """Returns reward vector of len nS"""

        # convert state index to position on grid
        pos = self.S[state]

        # reward vector
        res = np.zeros((self.nS))

        # skip if current state is terminal, or agent can't take action from
        # there
        if (
            self.state[pos[0], pos[1]] != OBJECT_TO_IDX["lava"]
            and self.state[pos[0], pos[1]] != OBJECT_TO_IDX["goal"]
            and self.state[pos[0], pos[1]] != OBJECT_TO_IDX["wall"]
        ):

            # Move left
            if action == self.actions.left:
                y, x = pos[0], pos[1] - 1

            # Move right
            elif action == self.actions.right:
                y, x = pos[0], pos[1] + 1

            # Move up
            elif action == self.actions.up:
                y, x = pos[0] - 1, pos[1]

            # Move down
            elif action == self.actions.down:
                y, x = pos[0] + 1, pos[1]

            else:
                assert False, "unknown action"

            # if transitioned from stoch reward state
            if self.state[pos[0], pos[1]] == OBJECT_TO_IDX["rand_r"]:

                # check if the transition needs to be rewarded
                if self.env.transition[pos[0], pos[1]] == action+1 or \
                    self.env.transition[pos[0], pos[1]] == 5:
                    try:
                        pos = np.where((self.env.stoch_r_pos == pos).all(axis=1))[0][0]
                        if self.env.stoch_r_state[pos] == 0:
                            res[np.where((self.S == [y, x]).all(axis=1))[0]] = \
                                REWARDS[self.state[pos[0], pos[1]]]
                    except:
                                res[np.where((self.S == [y, x]).all(axis=1))[0]] = 0

            else:
                if isinstance(REWARDS[self.state[y, x]], int):
                    res[np.where((self.S == [y, x]).all(axis=1))[0]] = \
                            REWARDS[self.state[y, x]]

        return res


class Maps(gym.Env):
    """Splified gridworld environment, builds on top of gym-gridworld."""

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Move left, right, up, down
        left = 0
        right = 1

        up = 2
        down = 3


    def __init__(self, env, rng, seed=0):
        super().__init__()

        self.env = gym.make(env)
        self.seed = seed
        self.rng = rng
        self.env.seed = seed

        # Action enumeration for this environment
        self.actions = Maps.Actions
        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))
        self.nA = len(self.actions)


        self.step_count = 0
        self.init_flag = 0  # first time setting up env
        self.max_steps = 100
        self.map = None     # loaded env map
        self.transition = None # Transition maps

    def _tokenizer(self, fname):
        with open(fname) as f:
            chunk = []
            for line in f:
                if 'START'in line:
                    continue
                if 'END' in line:
                    yield chunk
                    chunk = []
                    continue
                chunk.append(line)
        return chunk


    def load_map(self, path):
        """Load world from file located at path"""
        self.map, self.transition = [np.loadtxt(A) for A in self._tokenizer(path)]
        self.map = np.array(self.map, dtype=np.int64)
        self.transition = np.array(self.transition, dtype=np.int64)

    def step(self, action):
        """Given action (and current state), transition to next state"""

        prev_pos = copy.deepcopy(self.agent_pos)

        # stochastic transition
        if self.state[self.agent_pos[0], self.agent_pos[1]] == OBJECT_TO_IDX['rand_t']:
            valid_dx_dy = self.stoch_visit[str(self.agent_pos)]
            idx = self.rng.integers(0, len(valid_dx_dy))
            self.agent_pos[0] += valid_dx_dy[idx][0]
            self.agent_pos[1] += valid_dx_dy[idx][1]

        # Check if action is valid from current state, otherwise don't transition
        elif self.transition[self.agent_pos[0], self.agent_pos[1]] == action+1 or \
            self.transition[self.agent_pos[0], self.agent_pos[1]] == 5:

            # Move left
            if action == self.actions.left:
                y, x = self.agent_pos[0], self.agent_pos[1] - 1

            # Move right
            elif action == self.actions.right:
                y, x = self.agent_pos[0], self.agent_pos[1] + 1

            # Move up
            elif action == self.actions.up:
                y, x = self.agent_pos[0] - 1, self.agent_pos[1]

            # Move down
            elif action == self.actions.down:
                y, x = self.agent_pos[0] + 1, self.agent_pos[1]

            # look ahead if there is a wall
            if self.state[y, x] != OBJECT_TO_IDX["wall"]:
                self.agent_pos = np.array([y, x], dtype=np.int64)

        reward = 0

        # if a new state was visited
        if prev_pos[0] != self.agent_pos[0] or \
            prev_pos[1] != self.agent_pos[1]:

            # Transition was from a rand_r state
            if self.state[prev_pos[0], prev_pos[1]] == OBJECT_TO_IDX['rand_r']:
                try:
                    pos = np.where((self.stoch_r_pos == prev_pos).all(axis=1))[0]
                    if self.stoch_r_state[pos] == 0:
                        reward = self.rng.normal(RAND_R_MU, RAND_R_STD)
                        self.stoch_r_state[pos] += 1
                except:
                    reward = 0

            else:
                if isinstance(REWARDS[self.state[self.agent_pos[0], self.agent_pos[1]]], int):
                    reward = REWARDS[self.state[self.agent_pos[0], self.agent_pos[1]]]

        done = True \
            if self.state[self.agent_pos[0], self.agent_pos[1]] in TERMINAL \
                else False

        if self.step_count == self.max_steps:
            done = True
            reward = REWARDS['timeout']

        self.step_count += 1

        obs = np.where((self.S == self.agent_pos).all(axis=1))[0][0]
        return obs, reward, done, {}


    def reset(self, **kwargs):
        """Create the converted environment"""

        # Use file map
        self.state = np.array(copy.deepcopy(self.map))

        # Get all possible locations that can be occupied
        self.S = np.stack(np.where(self.state != OBJECT_TO_IDX["wall"])).T
        self.nS = self.S.shape[0]

        # Agent, goal and lava positions
        self.agent_pos = np.concatenate(np.where(self.state == OBJECT_TO_IDX["agent"]))
        self.goal_pos = np.stack(np.where(self.state == OBJECT_TO_IDX["goal"]), axis=-1)

        # rename agent state to empty state
        self.state[self.agent_pos[0], self.agent_pos[1]] = OBJECT_TO_IDX["empty"]

        # Keep track of single visit reward states. 0 - unvisited, 1 = visited
        # self.single_visit_pos = np.stack(np.where(self.state == OBJECT_TO_IDX["one_r"]))
        # self.single_visit_state = np.zeros((len(self.single_visit_pos)))

        # Keep track of stoch reward state. 0 - unvisited, 1 = visited
        self.stoch_r_pos = np.stack(np.where(self.state == OBJECT_TO_IDX["rand_r"]), axis=-1)
        self.stoch_r_state = np.zeros((len(self.stoch_r_pos)))

        # Keep track of positions we can transition to from stochastic states
        stoch_states = np.stack(np.where(self.state == OBJECT_TO_IDX["rand_t"]), axis=-1)
        self.stoch_visit = {str(i):[] for i in stoch_states}
        for state in stoch_states:
            valid_dx_dy = []
            if len(state) > 0:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if self.state[state[0] + dy, state[1] + dx] != OBJECT_TO_IDX["wall"]:
                            valid_dx_dy.append([dy, dx])
            self.stoch_visit[str(state)] = valid_dx_dy

        # get state of agent
        idx = np.where((self.S == self.agent_pos).all(axis=1))[0][0]

        self.step_count = 0

        # array of final states
        self.final_state = []
        for goal_pos in self.goal_pos:
            self.final_state.append(np.where((self.S == goal_pos).all(axis=1))[0][0])
        self.final_state = np.asarray(self.final_state)

        return idx


    def render(self, mode="human", **kwargs):
        return self.state
