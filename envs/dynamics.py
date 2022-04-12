from typing import Tuple
import numpy as np


class EnvSpec(object):
    def __init__(self, nS, nA, gamma):
        self._nS = nS
        self._nA = nA
        self._gamma = gamma

    @property
    def nS(self) -> int:
        """# possible states"""
        return self._nS

    @property
    def nA(self) -> int:
        """# possible actions"""
        return self._nA

    @property
    def gamma(self) -> float:
        """discounting factor of the environment"""
        return self._gamma


class Env(object):
    def __init__(self, env_spec, rng):
        self._env_spec = env_spec
        self.rng = rng

    @property
    def spec(self) -> EnvSpec:
        return self._env_spec

    def reset(self) -> int:
        """
        reset the environment. It should be called when you want to generate a new episode
        return:
            initial state
        """
        raise NotImplementedError()

    def step(self, action: int) -> Tuple[int, int, bool]:
        """
        proceed one step.
        return:
            next state, reward, done (whether it reached to a terminal state)
        """
        raise NotImplementedError()


class EnvWithModel(Env):
    @property
    def TD(self) -> np.array:
        """
        Transition Dynamics
        return: a numpy array shape of [nS,nA,nS]
            TD[s,a,s'] := the probability it will resulted in s' when it execute action a given state s
        """
        raise NotImplementedError()

    @property
    def R(self) -> np.array:
        """
        Reward function
        return: a numpy array shape of [nS,nA,nS]
            R[s,a,s'] := reward the agent will get it experiences (s,a,s') transition.
        """
        raise NotImplementedError()


class OneStateMDP(Env):  # MDP introduced at Fig 5.4 in Sutton Book
    def __init__(self, rng):
        env_spec = EnvSpec(2, 2, 1.0)

        super().__init__(env_spec, rng)
        self.final_state = 1
        self.trans_mat, self.r_mat = self._build_trans_mat()

    def _build_trans_mat(self):
        trans_mat = np.zeros((2, 2, 2))

        trans_mat[0, 0, 0] = 0.9
        trans_mat[0, 0, 1] = 0.1
        trans_mat[0, 1, 0] = 0.0
        trans_mat[0, 1, 1] = 1.0
        trans_mat[1, :, 1] = 1.0

        r_mat = np.zeros((2, 2, 2))
        r_mat[0, 0, 1] = 1.0

        return trans_mat, r_mat

    def reset(self):
        self._state = 0
        return self._state

    def step(self, action):
        assert action in list(range(self.spec.nA)), "Invalid Action"
        assert self._state != self.final_state, "Episode has ended!"

        prev_state = self._state
        self._state = self.rng.choice(
            self.spec.nS, p=self.trans_mat[self._state, action]
        )
        r = self.r_mat[prev_state, action, self._state]

        if self._state == self.final_state:
            return self._state, r, True
        else:
            return self._state, r, False


class OneStateMDPWithModel(OneStateMDP, EnvWithModel):
    @property
    def TD(self) -> np.array:
        return self.trans_mat

    @property
    def R(self) -> np.array:
        return self.r_mat


class FiveStateMDP(Env):
    def __init__(self, rng):
        env_spec = EnvSpec(5, 5, 1.0)

        super().__init__(env_spec, rng)
        self.final_state = 4
        self.trans_mat, self.r_mat = self._build_trans_mat()

    def _build_trans_mat(self):
        trans_mat = np.zeros((5, 5, 5))
        trans_mat[0, :, 0] = 1.0
        trans_mat[1, :, 1] = 1.0
        trans_mat[2, :, 2] = 1.0
        trans_mat[3, :, 3] = 1.0

        trans_mat[0, 0, 0] = 0.9
        trans_mat[0, 0, 1] = 0.1
        trans_mat[1, 3, 1] = 0.1
        trans_mat[1, 3, 2] = 0.9
        trans_mat[1, 1, 1] = 0
        trans_mat[1, 1, 3] = 0.1
        trans_mat[1, 1, 2] = 0.9
        trans_mat[1, 2, 2] = 0.8
        trans_mat[1, 2, 1] = 0.2
        trans_mat[2, 4, 1] = 0.2
        trans_mat[2, 4, 3] = 0.8
        trans_mat[2, 4, 2] = 0
        trans_mat[2, 1, 2] = 0.2
        trans_mat[2, 1, 3] = 0.8
        trans_mat[2, 3, 3] = 0.5
        trans_mat[2, 3, 2] = 0.5
        trans_mat[3, 4, 4] = 0.2
        trans_mat[3, 4, 3] = 0.8
        trans_mat[3, 2, 1] = 0.2
        trans_mat[3, 2, 3] = 0.8
        trans_mat[4, :, 4] = 1

        r_mat = np.zeros((5, 5, 5))
        r_mat[3, 4, 4] = 1.0

        return trans_mat, r_mat

    def reset(self):
        self._state = 0
        return self._state

    def step(self, action):
        assert action in list(range(self.spec.nA)), "Invalid Action"
        assert self._state != self.final_state, "Episode has ended!"

        prev_state = self._state
        self._state = self.rng.random.choice(
            self.spec.nS, p=self.trans_mat[self._state, action]
        )
        r = self.r_mat[prev_state, action, self._state]

        if self._state == self.final_state:
            return self._state, r, True
        else:
            return self._state, r, False


class FiveStateMDPWithModel(FiveStateMDP, EnvWithModel):
    @property
    def TD(self) -> np.array:
        return self.trans_mat

    @property
    def R(self) -> np.array:
        return self.r_mat
