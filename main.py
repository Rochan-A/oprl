from typing import Tuple
import numpy as np

import argparse
import yaml
import matplotlib.pyplot as plt
from easydict import EasyDict

import os
import os.path as osp
from os import listdir
from os.path import isfile, join

from envs import *
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
    plt.savefig('over.png')
    plt.show()

def get_overestimation_time_method(q_star, vl, steps, DQ=False):
    over = np.zeros(steps)
    sum = 0

    for i in range(steps):
        sum_episode = 0
        sum_episode_qstar = 0
        for j in range( len(vl[i][0]) ):
            if DQ:
                qval = (vl[i][2][j] + vl[i][3][j]) / 2.0
            else:
                qval = vl[i][2][j]
            sum_episode += qval
            sum_episode_qstar += q_star[ vl[i][0][j], vl[i][1][j] ]
        avg_episode = sum_episode / len(vl[i][0])
        avg_episode_qstar = sum_episode_qstar / len(vl[i][0])

        over[i] = avg_episode - avg_episode_qstar

    return over


def plot_overestimation_time(q_star, vlQ, vlDQ, vlPQ, config):
    over_ql = get_overestimation_time_method(q_star, vlQ, config.q_learning.steps)
    over_dql = get_overestimation_time_method(q_star, vlDQ, config.dq_learning.steps, True)
    over_pq = get_overestimation_time_method(q_star, vlPQ, config.pessimistic_q_learning.steps)

    plt.figure()
    plt.plot(np.arange(config.q_learning.steps), over_ql, label="Q learning")
    plt.plot(np.arange(config.dq_learning.steps), over_dql, label="DQ learning")
    plt.plot(np.arange(config.pessimistic_q_learning.steps), over_pq, label="PQ learning")
    plt.legend()
    plt.savefig('over_time.png')
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
    elif config.model == 'gridworld':
        env = Converted(config.env, args.seed)

        env_with_model = Converted(config.env, args.seed)
        env_with_model = EnvModel(env_with_model)

    # Value Iteration
    V = np.zeros(shape=(env.spec.nS))
    print('Value Iteration (DP)')
    print('Initial V values\nV: {}'.format(V))
    V_star, pi_star, q_star = value_iteration_gridworld(env_with_model,V,config.value_iteration_theta,config.gamma,rng)
    print('True Value Values\n{}\nTrue Q Values\n{}'.format(V_star, q_star))
    print()


    # Save true state values and env state as an image
    plt.imsave('env.png', env_with_model.state)
    grid = np.zeros(env_with_model.state.shape)
    for idx, val in enumerate(V_star):
        y, x = env_with_model.S[idx]
        grid[y, x] = val

    fig, ax = plt.subplots()
    ax.matshow(grid, cmap=plt.cm.Blues)
    for i in range(env_with_model.state.shape[0]):
        for j in range(env_with_model.state.shape[1]):
            c = grid[j,i]
            ax.text(i, j, str(c)[:4], va='center', ha='center')
    plt.savefig('true_value.png')


    # Q-learning
    print('Q Learning')
    Q = np.zeros(shape=(env.spec.nS,env.spec.nA))
    print('Initial Q values\n{}'.format(Q))
    Q, vlQ = Q_learning(env, config.q_learning.steps, config.q_learning.alpha, config.gamma, config.q_learning.epsilon, Q, rng)
    print('Estimated Q Values\n{}'.format(Q))
    print()

    # Double Q-learning
    print('Double Q Learning')
    DQ1 = np.zeros(shape=(env.spec.nS,env.spec.nA))
    DQ2 = np.zeros(shape=(env.spec.nS,env.spec.nA))
    print('Initial Q values\nQ1: {}\nQ2: {}'.format(DQ1, DQ2))
    DQ1, DQ2, vlDQ = DoubleQ(env, config.dq_learning.steps, config.dq_learning.alpha, config.gamma, config.dq_learning.epsilon, DQ1, DQ2, rng)
    print('Estimated Q1 Values\n{}\nEstimated Q2 Values\n{}'.format(DQ1, DQ2))
    print()

    print('Pessimistic Q Learning')
    PQ = np.zeros(shape=(env.spec.nS,env.spec.nA))
    print('Initial Q values\nQ: {}'.format(PQ))
    PQ, vlPQ = PessimisticQ(env, config.pessimistic_q_learning.steps, config.pessimistic_q_learning.alpha, config.gamma, config.pessimistic_q_learning.epsilon, config.pessimistic_q_learning.pessimism_coeff, PQ, rng)
    print('Estimated PQ Values\n{}\n'.format(PQ))
    print()

    plot_overestimation_time(q_star, vlQ, vlDQ, vlPQ, config)
    plot_overestimation(q_star, Q, DQ1, DQ2)
