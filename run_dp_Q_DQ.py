from typing import Tuple
import numpy as np
from tqdm import tqdm

import argparse
import yaml
import matplotlib.pyplot as plt
from easydict import EasyDict

import os
import os.path as osp
from os import listdir
from os.path import isfile, join

from envs.dynamics import OneStateMDP, OneStateMDPWithModel
from envs.dynamics import FiveStateMDP, FiveStateMDPWithModel

from policies.q_values import Policy, EGPolicy, GreedyPolicy

from algos import *

def plot_overestimation(q_star, Q, DQ1, DQ2):
    lbs = []
    for i in range(len(Q)):
        for j in range(len(Q[0])):
            lbs.append('S' + str(i+1) + 'A' + str(j+1))
    q_star_flatten = q_star.flatten()
    Q_flatten = Q.flatten()
    DQ1_flatten = DQ1.flatten()
    DQ2_flatten = DQ2.flatten()

    bar_width = 0.2

    plt.figure()
    x_axis = np.arange(len(q_star_flatten))
    plt.bar(x_axis - 1.5*bar_width, q_star_flatten, width=bar_width, label='Q-star')
    plt.bar(x_axis - 0.5*bar_width, Q_flatten, width=bar_width, label='Q learning')
    plt.bar(x_axis + 0.5*bar_width, DQ1_flatten, width=bar_width, label='Double Q table 1')
    plt.bar(x_axis + 1.5*bar_width, DQ2_flatten, width=bar_width, label='Double Q table 2')
    plt.xticks(x_axis, lbs)
    plt.legend()
    plt.show()


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
    DQ1 = rng.normal(size=(env.spec.nS,env.spec.nA))
    DQ2 = rng.normal(size=(env.spec.nS,env.spec.nA))
    print('Initial Q values\nQ1: {}\nQ2: {}'.format(DQ1, DQ2))
    DQ1, DQ2 = DoubleQ(env, config.dq_learning.steps, config.dq_learning.alpha, config.dq_learning.gamma, config.dq_learning.epsilon, DQ1, DQ2, rng)
    print('Estimated Q1 Values\n{}\nEstimated Q2 Values\n{}'.format(DQ1, DQ2))
    print()

    plot_overestimation(q_star, Q, DQ1, DQ2)
