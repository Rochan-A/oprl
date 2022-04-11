import numpy as np
import gym

import gym_minigrid
from gym_minigrid.wrappers import *

from enum import IntEnum
from .dynamics import EnvSpec


# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen'        : 0,
    'empty'         : 1,
    'wall'          : 2,
    'floor'         : 3,
    'door'          : 4,
    'key'           : 5,
    'ball'          : 6,
    'box'           : 7,
    'goal'          : 8,
    'lava'          : 9,
    'agent'         : 10,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]


class EnvModel(gym.core.Wrapper):
    """Env Wrapper to query model ie. p(s, a, s')
    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def TD(self, state, action):
        """Returns Transition probability vector of len nS"""

        # convert state index to position on grid
        pos = self.S[state]

        # prob. vector
        res = np.zeros((self.nS))

        # if current state is terminal, or agent can't be there
        if self.state[pos[0], pos[1]] != OBJECT_TO_IDX['lava'] and \
            self.state[pos[0], pos[1]] != OBJECT_TO_IDX['goal'] and \
            self.state[pos[0], pos[1]] != OBJECT_TO_IDX['wall']:

            # Move left
            if action == self.actions.left:
                y, x = pos[0], pos[1]-1
                res[np.where((self.S == [y, x]).all(axis=1))[0]] = 1 \
                    if self.state[y, x] != OBJECT_TO_IDX['wall'] else 0

            # Move right
            elif action == self.actions.right:
                y, x = pos[0], pos[1]+1
                res[np.where((self.S == [y, x]).all(axis=1))[0]] = 1 \
                    if self.state[y, x] != OBJECT_TO_IDX['wall'] else 0

            # Move up
            elif action == self.actions.up:
                y, x = pos[0]-1, pos[1]
                res[np.where((self.S == [y, x]).all(axis=1))[0]] = 1 \
                    if self.state[y, x] != OBJECT_TO_IDX['wall'] else 0

            # Move down
            elif action == self.actions.down:
                y, x = pos[0]+1, pos[1]
                res[np.where((self.S == [y, x]).all(axis=1))[0]] = 1 \
                    if self.state[y, x] != OBJECT_TO_IDX['wall'] else 0

            else:
                assert False, "unknown action"

        return res


    def R(self, state, action):
        """Returns reward vector of len nS"""

        # convert state index to position on grid
        pos = self.S[state]

        # reward vector
        res = np.zeros((self.nS))

        # if current state is terminal, or agent can't be there
        if self.state[pos[0], pos[1]] != OBJECT_TO_IDX['lava'] and \
            self.state[pos[0], pos[1]] != OBJECT_TO_IDX['goal'] and \
            self.state[pos[0], pos[1]] != OBJECT_TO_IDX['wall']:

            # Move left
            if action == self.actions.left:
                y, x = pos[0], pos[1]-1
                if self.state[y, x] != OBJECT_TO_IDX['wall']:
                    res[np.where((self.S == [y, x]).all(axis=1))[0]] = 1 \
                        if y == self.goal_pos[0] and x == self.goal_pos[1] else 0

            # Move right
            elif action == self.actions.right:
                y, x = pos[0], pos[1]+1
                if self.state[y, x] != OBJECT_TO_IDX['wall']:
                    res[np.where((self.S == [y, x]).all(axis=1))[0]] = 1 \
                        if y == self.goal_pos[0] and x == self.goal_pos[1] else 0

            # Move up
            elif action == self.actions.up:
                y, x = pos[0]-1, pos[1]
                if self.state[y, x] != OBJECT_TO_IDX['wall']:
                    res[np.where((self.S == [y, x]).all(axis=1))[0]] = 1 \
                        if y == self.goal_pos[0] and x == self.goal_pos[1] else 0

            # Move down
            elif action == self.actions.down:
                y, x = pos[0]+1, pos[1]
                if self.state[y, x] != OBJECT_TO_IDX['wall']:
                    res[np.where((self.S == [y, x]).all(axis=1))[0]] = 1 \
                        if y == self.goal_pos[0] and x == self.goal_pos[1] else 0
            else:
                assert False, "unknown action"

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
        self.env.seed = seed

        # Action enumeration for this environment
        self.actions = Converted.Actions
        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))
        self.nA = len(self.actions)
        self.step_count = 0
        self.max_steps = 100
        _ = self.reset()


    def step(self, action):
        reward = 0
        done = False

        # Move left
        if action == self.actions.left:
            y, x = self.agent_pos[0], self.agent_pos[1]-1
            # look ahead if its a wall
            if self.state[y, x] != OBJECT_TO_IDX['wall']:
                self.state[self.agent_pos[0], self.agent_pos[1]] = OBJECT_TO_IDX['empty']
                self.agent_pos[1] -= 1
                if self.state[self.agent_pos[0], self.agent_pos[1]] == OBJECT_TO_IDX['lava']:
                    done = True
                else:
                    self.state[self.agent_pos[0], self.agent_pos[1]] = OBJECT_TO_IDX['agent']

        # Move right
        elif action == self.actions.right:
            y, x = self.agent_pos[0], self.agent_pos[1]+1
            if self.state[y, x] != OBJECT_TO_IDX['wall']:
                self.state[self.agent_pos[0], self.agent_pos[1]] = OBJECT_TO_IDX['empty']
                self.agent_pos[1] += 1
                if self.state[self.agent_pos[0], self.agent_pos[1]] == OBJECT_TO_IDX['lava']:
                    done = True
                else:
                    self.state[self.agent_pos[0], self.agent_pos[1]] = OBJECT_TO_IDX['agent']

        # Move up
        elif action == self.actions.up:
            y, x = self.agent_pos[0]-1, self.agent_pos[1]
            if self.state[y, x] != OBJECT_TO_IDX['wall']:
                self.state[self.agent_pos[0], self.agent_pos[1]] = OBJECT_TO_IDX['empty']
                self.agent_pos[0] -= 1
                if self.state[self.agent_pos[0], self.agent_pos[1]] == OBJECT_TO_IDX['lava']:
                    done = True
                else:
                    self.state[self.agent_pos[0], self.agent_pos[1]] = OBJECT_TO_IDX['agent']

        # Move down
        elif action == self.actions.down:
            y, x = self.agent_pos[0]+1, self.agent_pos[1]
            if self.state[y, x] != OBJECT_TO_IDX['wall']:
                self.state[self.agent_pos[0], self.agent_pos[1]] = OBJECT_TO_IDX['empty']
                self.agent_pos[0] += 1
                if self.state[self.agent_pos[0], self.agent_pos[1]] == OBJECT_TO_IDX['lava']:
                    done = True
                else:
                    self.state[self.agent_pos[0], self.agent_pos[1]] = OBJECT_TO_IDX['agent']

        else:
            assert False, "unknown action"

        self.step_count +=1

        if self.agent_pos[0] == self.goal_pos[0] and self.agent_pos[1] == self.goal_pos[1]:
            reward = 1
            done = True
        else:
            if self.step_count == self.max_steps:
                done = True

        obs = np.where((self.S == self.agent_pos).all(axis=1))[0][0]
        return obs, reward, done, {}


    def reset(self, **kwargs):
        """Create the converted environment"""
        if self.step_count == 0:
            _ = self.env.reset(**kwargs)
        self.state = self.env.grid.encode()[:, :, 0]

        self.agent_pos = np.array(self.env.agent_pos)
        self.state[self.agent_pos[0], self.agent_pos[1]] = OBJECT_TO_IDX['agent']

        # Fix the orientation so that we can save as an image
        self.state = np.rot90(np.flipud(self.state), 3)

        # Get all possible locations that can be occupied
        self.S = np.stack(np.where(self.state != OBJECT_TO_IDX['wall']), axis=-1)
        self.nS = self.S.shape[0]

        self.spec = EnvSpec(self.nS, self.nA, 1)

        # get state of agent
        idx = np.where((self.S == self.agent_pos).all(axis=1))[0][0]

        self.goal_pos = np.concatenate(np.where(self.state == OBJECT_TO_IDX['goal']))
        self.lava_pos = np.concatenate(np.where(self.state == OBJECT_TO_IDX['lava']))

        self.step_count = 0
        self.final_state = np.where((self.S == self.goal_pos).all(axis=1))[0][0]

        return idx


    def render(self, mode="human", **kwargs):
        return self.state
