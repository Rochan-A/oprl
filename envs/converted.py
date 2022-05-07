
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
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}

# Map of rewards landing on specific position
REWARDS = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 1,
    7: 0,
    8: 1,
    9: -1,
    10: 0,
    'timeout': -1
}


# Terminal states
TERMINAL = [OBJECT_TO_IDX['lava'], OBJECT_TO_IDX['goal']]


class EnvModel(gym.core.Wrapper):
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

        # skip if current state is terminal, or agent can't transition from there
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

            res[np.where((self.S == [y, x]).all(axis=1))[0]] = 1 \
                if self.state[y, x] != OBJECT_TO_IDX["wall"] else 0

        return res


    def R(self, state, action):
        """Returns reward vector of len nS"""

        # convert state index to position on grid
        pos = self.S[state]

        # reward vector
        res = np.zeros((self.nS))

        # skip if current state is terminal, or agent can't be there
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

            if pos[0] != y or pos[1] != x:
                res[np.where((self.S == [y, x]).all(axis=1))[0]] = \
                    REWARDS[self.state[y, x]]

        return res


class Converted(gym.Env):
    """Convert GridWorld environments to work with numpy arrays
    Simplify environment, removes orientation of agent.

    Only lava and four rooms work
    """

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move up, move down
        left = 0
        right = 1

        up = 2
        down = 3

        # Other actions not needed

        # # Pick up an object
        # pickup = 4
        # # Drop an object
        # drop = 5
        # # Toggle/activate an object
        # toggle = 6

        # # Done completing task
        # done = 7


    def __init__(self, env, seed=0):
        super().__init__()

        self.env = gym.make(env)
        self.seed = seed
        self.env.seed(seed)

        # Action enumeration for this environment
        self.actions = Converted.Actions
        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))
        self.nA = len(self.actions)
        self.step_count = 0
        self.f = 0
        self.max_steps = 100
        self.map = None

    def load_map(self, path):
        """Load world from file located at path"""
        self.map = np.loadtxt(path)

    def step(self, action):

        # Move left
        if action == self.actions.left:
            y, x = self.agent_pos[0], self.agent_pos[1] - 1
            # look ahead if its a wall
            if self.state[y, x] != OBJECT_TO_IDX["wall"]:
                self.agent_pos[1] -= 1

        # Move right
        elif action == self.actions.right:
            y, x = self.agent_pos[0], self.agent_pos[1] + 1
            if self.state[y, x] != OBJECT_TO_IDX["wall"]:
                self.agent_pos[1] += 1

        # Move up
        elif action == self.actions.up:
            y, x = self.agent_pos[0] - 1, self.agent_pos[1]
            if self.state[y, x] != OBJECT_TO_IDX["wall"]:
                self.agent_pos[0] -= 1

        # Move down
        elif action == self.actions.down:
            y, x = self.agent_pos[0] + 1, self.agent_pos[1]
            if self.state[y, x] != OBJECT_TO_IDX["wall"]:
                self.agent_pos[0] += 1

        else:
            assert False, "unknown action"

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
        if self.map is None:
            # Use gridworld
            self.env.seed(self.seed)
            if self.f == 0:
                _ = self.env.reset()
                self.f = 1

            self.agent_pos = self.env.agent_pos
            self.state = copy.deepcopy(self.env.grid.encode()[:, :, 0])
            self.state[self.agent_pos[0], self.agent_pos[1]] = OBJECT_TO_IDX["agent"]

            # Fix the orientation so that we can save as an image
            self.state = np.rot90(np.flipud(self.state), 3)
        else:
            # Use file map
            self.state = np.array(copy.deepcopy(self.map))

        self.env.seed(self.seed)

        # Get all possible locations that can be occupied
        self.S = np.stack(np.where(self.state != OBJECT_TO_IDX["wall"])).T
        self.nS = self.S.shape[0]

        self.agent_pos = np.concatenate(np.where(self.state == OBJECT_TO_IDX["agent"]))
        self.goal_pos = np.concatenate(np.where(self.state == OBJECT_TO_IDX["goal"]))
        self.lava_pos = np.concatenate(np.where(self.state == OBJECT_TO_IDX["lava"]))

        self.state[self.agent_pos[0], self.agent_pos[1]] = OBJECT_TO_IDX["empty"]

        # get state of agent
        idx = np.where((self.S == self.agent_pos).all(axis=1))[0][0]

        self.step_count = 0
        self.final_state = np.where((self.S == [self.goal_pos]).all(axis=1))[0][0]

        return idx

    def render(self, mode="human", **kwargs):
        return self.state