from typing import Tuple
import numpy as np
from tqdm import tqdm

import argparse
import yaml
from easydict import EasyDict

import os
import os.path as osp
from os import listdir
from os.path import isfile, join

from envs.dynamics import OneStateMDP, OneStateMDPWithModel
from envs.dynamics import FiveStateMDP, FiveStateMDPWithModel

from policies.q_values import Policy, EGPolicy, GreedyPolicy


def value_iteration(env, initV:np.array, theta:float, rng) -> Tuple[np.array,Policy]:
    """
    inp:
        env: environment with model information, i.e. you know transition dynamics and reward function
        initV: initial V(s); numpy array shape of [nS,]
        theta: exit criteria
        rng: rng
    return:
        value: optimal value function; numpy array shape of [nS]
        policy: optimal deterministic policy; instance of Policy class
        q: q values, numpy array shape of [nS, nA]
    """

    #####################
    # Value Iteration Algorithm (Hint: Sutton Book p.83)
    #####################

    nS = initV.shape[0]
    assert nS == env.TD.shape[0], 'States in initV does not equal TD shape!'
    # (state, action, next_state)
    nA = env.TD.shape[1]
    while True:
        delta = 0
        for s_i in range(nS):
            v = initV[s_i]
            s = [0]*env.TD[s_i].shape[0]
            for a_i in range(env.TD[s_i].shape[0]):
                s[a_i] = np.sum(env.TD[s_i, a_i, :]*(env.R[s_i, a_i, :] + initV[:]))
            initV[s_i] = max(s)
            delta = max(delta, abs(v - initV[s_i]))
        if delta < theta:
            break

    V = initV

    # Compute Q values for policy
    Q = np.zeros((nS, nA))
    for s_i in range(nS):
        for a_i in range(nA):
            Q[s_i, a_i] = np.sum([
                    env.TD[s_i, a_i, s_p]*(env.R[s_i, a_i, s_p] + initV[s_p]) for s_p in range(nS)
                    ])

    pi = GreedyPolicy(Q, rng)

    return V, pi, Q


def Q_learning(env, n:int, alpha:float, gamma:float, epsilon:float, Q:np.array, rng) -> Tuple[np.array,Policy]:
    """
    input:
        env: environment
        n: how many steps?
        alpha: learning rate
        gamma: discount factor
        epslion: epsilon-greedy exploration
        Q: initial Q function
        rng:
    ret:
        Q: $q_star$ function; numpy array shape of [nS,nA]
    """

    #####################
    # Q Learning (Hint: Sutton Book p. 131)
    #####################

    terminal = env.final_state
    Q[terminal, :] = 0

    pi = EGPolicy(Q, epsilon, rng)

    for i in tqdm(range(n)):
        s = env.reset()
        done = False
        while not done:
            a = pi.action(s)
            s1, r, done = env.step(a)

            Q[s,a] += alpha*(r + (gamma*np.argmax(Q[s1, :]) - Q[s, a]))
            s = s1
            pi.update(Q)

    return Q


def DoubleQ(env, n:int, alpha:float, gamma:float, epsilon:float, Q1:np.array, Q2:np.array, rng) -> Tuple[np.array,Policy]:
    """
    input:
        env: environment
        n: how many steps?
        alpha: learning rate
        gamma: discount factor
        epslion: epsilon-greedy exploration
        Q1: initial Q1 function
        Q2: initial Q2 function
        rng: rng
    ret:
        Q1: Q1 function; numpy array shape of [nS,nA]
        Q1: Q2 function; numpy array shape of [nS,nA]
    """

    #####################
    # Double Q Learning (Hint: Sutton Book p. 135-136)
    #####################


    assert Q1.all() == Q2.all(), 'Q values need to be different, otherwise there is no point in running Double Q learning...'

    terminal = env.final_state
    Q1[terminal, :] = 0
    Q2[terminal, :] = 0

    pi = EGPolicy(Q1, epsilon, rng)

    for i in tqdm(range(n)):
        s = env.reset()
        done = False
        while not done:
            a = pi.action(s)
            s1, r, done = env.step(a)

            Q1[s,a] += alpha*(r + gamma*Q2[s1, np.argmax(Q1[s1, :])] - Q1[s, a])
            Q2[s,a] += alpha*(r + gamma*Q1[s1, np.argmax(Q2[s1, :])] - Q2[s, a])

            s = s1
            pi.update(Q1)

    return Q1, Q2


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        help="Path to Config file",
        required=True
    )
    parser.add_argument(
        "--seed",
        help="set numpy & torch seed",
        type=int,
        default=0
    )

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    if config.model == 'one_state':
        env = OneStateMDP(rng)
        env_with_model = OneStateMDPWithModel(rng)
    elif config.model == 'five_state':
        env = FiveStateMDP(rng)
        env_with_model = FiveStateMDPWithModel(rng)

    # Value Iteration
    V = rng.normal(size=(env.spec.nS))
    print('Value Iteration (DP)')
    print('Initial V values\nV: {}'.format(V))
    V_star, pi_star, q_star = value_iteration(env_with_model,V,config.value_iteration_theta,rng)
    print('True Value Values\n{}\nTrue Q Values\n{}'.format(V_star, q_star))
    print()

    # Q-learning
    print('Q Learning')
    Q = rng.normal(size=(env.spec.nS,env.spec.nA))
    print('Initial Q values\n{}'.format(Q))
    Q = Q_learning(env, config.q_learning.steps, config.q_learning.alpha, config.q_learning.gamma, config.q_learning.epsilon, Q, rng)
    print('Estimated Q Values\n{}'.format(Q))
    print()

    # Double Q-learning
    print('Double Q Learning')
    Q1 = rng.normal(size=(env.spec.nS,env.spec.nA))
    Q2 = rng.normal(size=(env.spec.nS,env.spec.nA))
    print('Initial Q values\nQ1: {}\nQ2: {}'.format(Q1, Q2))
    Q1, Q2 = DoubleQ(env, config.dq_learning.steps, config.dq_learning.alpha, config.dq_learning.gamma, config.dq_learning.epsilon, Q1, Q2, rng)
    print('Estimated Q1 Values\n{}\nEstimated Q2 Values\n{}'.format(Q1, Q2))
    print()