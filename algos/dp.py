from typing import Tuple
import numpy as np
import gym


def value_iteration(
    env: gym.Env,
    initV: np.ndarray,
    theta: float,
    gamma: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    inp:
        env: environment with model information, i.e. w/ transition dynamics
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
        gamma: discount factor
    return:
        value: optimal value function; numpy array shape of [nS]
        q: q values, numpy array shape of [nS, nA]
    """
    #####################
    # Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################

    nS = initV.shape[0]
    assert nS == env.TD.shape[0], "States in initV does not equal TD shape!"
    # (state, action, next_state)
    nA = env.TD.shape[1]
    while True:
        delta = 0
        for s_i in range(nS):
            v = initV[s_i]
            s = [0] * env.TD[s_i].shape[0]
            for a_i in range(env.TD[s_i].shape[0]):
                s[a_i] = np.sum(
                    env.TD[s_i, a_i, :] * (env.R[s_i, a_i, :] + gamma * initV[:])
                )
            initV[s_i] = max(s)
            delta = max(delta, abs(v - initV[s_i]))
        if delta < theta:
            break

    V = initV

    # Compute Q values for policy
    Q = np.zeros((nS, nA))
    for s_i in range(nS):
        for a_i in range(nA):
            Q[s_i, a_i] = np.sum(
                [
                    env.TD[s_i, a_i, s_p] * (env.R[s_i, a_i, s_p] + gamma * initV[s_p])
                    for s_p in range(nS)
                ]
            )

    return V, Q


def value_iteration_gridworld(
    env: gym.Env,
    initV: np.ndarray,
    theta: float,
    gamma: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Uses function calls to TD and R instead of reading from pre-computed
    array.

    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
        gamma: discount factor
    return:
        value: optimal value function; numpy array shape of [nS]
        q: q values, numpy array shape of [nS, nA]
    """

    #####################
    # Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################

    nS, nA = env.nS, env.nA

    while True:
        delta = 0
        for s_i in range(nS):
            v = initV[s_i]
            s = [0] * env.nA
            for a_i in range(env.nA):
                s[a_i] = np.sum(env.TD(s_i, a_i) * (env.R(s_i, a_i) + gamma * initV[:]))
            initV[s_i] = max(s)
            delta = max(delta, abs(v - initV[s_i]))
        if delta < theta:
            break

    V = initV

    # Compute Q values for policy
    Q = np.zeros((nS, nA))
    for s_i in range(nS):
        for a_i in range(nA):
            Q[s_i, a_i] = np.sum(
                [
                    env.TD(s_i, a_i)[s_p] * (env.R(s_i, a_i)[s_p] + gamma * initV[s_p])
                    for s_p in range(nS)
                ]
            )

    return V, Q
