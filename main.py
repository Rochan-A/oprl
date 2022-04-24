from typing import Tuple
import numpy as np
import ast

import argparse
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from easydict import EasyDict
import copy
from tqdm import tqdm
import pathlib

from envs import *
from algos import *
from policies import GreedyPolicy


def make_dirs(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def plot_overestimation(q_star, Q, DQ1, DQ2, PQ):
    lbs = []
    for i in range(len(Q)):
        for j in range(len(Q[0])):
            lbs.append("S" + str(i + 1) + "A" + str(j + 1))
    q_star_flatten = q_star.flatten()
    Q_flatten = Q.flatten()
    DQ1_flatten = DQ1.flatten()
    DQ2_flatten = DQ2.flatten()
    PQ_flatten = PQ.flatten()

    bar_width = 0.2

    plt.figure(figsize=(18, 5))
    x_axis = np.arange(len(q_star_flatten))
    plt.bar(x_axis - 1.5 * bar_width, q_star_flatten, width=bar_width, label="Q-star")
    plt.bar(x_axis - 0.5 * bar_width, Q_flatten, width=bar_width, label="Q learning")
    plt.bar(
        x_axis + 0.5 * bar_width, DQ1_flatten, width=bar_width, label="Double Q table 1"
    )
    plt.bar(
        x_axis + 1 * bar_width, DQ2_flatten, width=bar_width, label="Double Q table 2"
    )
    plt.bar(x_axis + 2 * bar_width, PQ_flatten, width=bar_width, label="PQ learning")
    plt.xticks(x_axis, lbs)
    plt.legend()
    plt.savefig("over.pdf")
    plt.show()


def get_overestimation_time_method(q_star, vl, steps, DQ=False):
    over = np.zeros(steps)
    sum = 0

    for i in range(steps):
        sum_episode = 0
        sum_episode_qstar = 0
        for j in range(len(vl[i][0])):
            if DQ:
                qval = (vl[i][2][j] + vl[i][3][j]) / 2.0
            else:
                qval = vl[i][2][j]
            sum_episode += qval
            sum_episode_qstar += q_star[vl[i][0][j], vl[i][1][j]]
        avg_episode = sum_episode / len(vl[i][0])
        avg_episode_qstar = sum_episode_qstar / len(vl[i][0])

        over[i] = avg_episode - avg_episode_qstar

    return over


def plot_overestimation_time(q_star, vlQ, vlDQ, vlPQ, config):
    over_ql = get_overestimation_time_method(q_star, vlQ, config.q_learning.steps)
    over_dql = get_overestimation_time_method(
        q_star, vlDQ, config.dq_learning.steps, True
    )
    over_pq = get_overestimation_time_method(
        q_star, vlPQ, config.pessimistic_q_learning.steps
    )

    plt.figure()
    plt.plot(np.arange(config.q_learning.steps), over_ql, label="Q learning")
    plt.plot(np.arange(config.dq_learning.steps), over_dql, label="DQ learning")
    plt.plot(
        np.arange(config.pessimistic_q_learning.steps), over_pq, label="PQ learning"
    )
    plt.legend()
    plt.savefig("over_time.pdf")
    plt.show()


def save_value_as_image(env, V, filename, int_=False):
    grid = np.zeros((env.state.shape[0], env.state.shape[1]))
    for idx, val in enumerate(V):
        y, x = env.S[idx]
        grid[y, x] = val

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    ax.matshow(grid, cmap=plt.cm.Blues)
    for i in range(env.state.shape[1]):
        for j in range(env.state.shape[0]):
            if not int_:
                c = '{:.2f}'.format(grid[j,i])
            else:
                c = '{:0.0f}'.format(grid[j,i])
            ax.text(i, j, c, va='center', ha='center', fontsize=10)
    plt.savefig('{}.png'.format(filename))


def q_to_v(Q, V_star):
    """Compute state values from state-action values"""
    pi = GreedyPolicy(Q)
    V = np.zeros(shape=(V_star.shape[-1]), dtype=np.float64)
    for s in range(V_star.shape[-1]):
        if V_star[s] != 0.0:
            sum = 0
            for a in range(Q.shape[-1]):
                sum += pi.action_prob(s, a) * Q[s, a]
            V[s] = sum
    return V


def set_initial_values(config, env):
    """Set the initial values for Q and V."""

    if config.init.Q == 'zero':
        Q = np.zeros((env.nS, env.nA), dtype=np.float64)
    elif config.init.Q == 'rand':
        Q = np.random.random((env.nS, env.nA)).astype(dtype=np.float64)
    elif config.init.Q == 'opt':
        Q = np.zeros((env.nS, env.nA), dtype=np.float64)
        Q.fill(1.0)
    elif config.init.Q == 'pes':
        Q = np.zeros((env.nS, env.nA), dtype=np.float64)
        Q.fill(-1.0)
    else:
        assert False, 'unknown Q value initialization'

    if config.init.V == 'zero':
        V = np.zeros((env.nS,), dtype=np.float64)
    elif config.init.V == 'rand':
        V = np.random.random((env.nS,)).astype(dtype=np.float64)
    elif config.init.V == 'opt':
        V = np.zeros((env.nS,), dtype=np.float64)
        V.fill(1.0)
    elif config.init.V == 'pes':
        V = np.zeros((env.nS,), dtype=np.float64)
        V.fill(-1.0)
    else:
        assert False, 'unknown V value initialization'
    return V, Q


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_mean_cum_rewards(data, labels):
    """Compute mean and std of env data."""
    means, stds = [], []
    for val in data:
        # reward
        means.append(val[:, :, 0].mean(0, dtype=np.float64))
        stds.append(val[:, :, 0].std(0, ddof=1, dtype=np.float64))

    fig, ax = plt.subplots(figsize=(20, 6), dpi=200)
    clrs = sns.color_palette("husl", len(data))
    with sns.axes_style("darkgrid"):
        for i in range(len(data)):
            ax.plot(np.arange(data[i].shape[-2]), means[i], label=labels[i], c=clrs[i])
            ax.fill_between(
                np.arange(data[i].shape[-2]),
                means[i]-stds[i],
                means[i]+stds[i],
                alpha=0.3,
                facecolor=clrs[i]
            )
        ax.legend()
        ax.set_title("Mean Cummulative Reward")
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
    plt.savefig('reward_plot.png')


def plot_Q_values(env, q_star, data, labels):
    # data: [VQ_Logger, DQ1_Logger, DQ2_Logger, PQ_Logger]
    means, stds = [], []
    for val in data:
        means.append(val.mean(0, dtype=np.float64) )
        stds.append(val.std(0, ddof=1, dtype=np.float64) )

    # trace Q values we are interested in
    pi_star = GreedyPolicy(q_star)
    optimal_traj = []
    done = False
    s = env.reset()
    while not done:
        a = pi_star.action(s)
        optimal_traj.append([s, a])
        s, _, done, _ = env.step(a)

    # fig, ax = plt.subplots(len(optimal_traj), 1)
    clrs = sns.color_palette("husl", len(data) + 1)
    make_dirs('state-action')
    with sns.axes_style("darkgrid"):
        for s in range(env.nS):
            for a in range(4):
                if not (env.TD(s, a) == 0).all():
                    plt.figure(figsize=(20, 6), dpi=200)
                    plt.plot(
                        np.arange(data[0].shape[1]),
                        [q_star[s, a]]*data[1].shape[1],
                        '--',
                        alpha=0.7,
                        label=labels[0],
                        c=clrs[0]
                    )
                    for i in range(len(data)):
                        plt.plot(
                            np.arange(data[i].shape[1]),
                            means[i][:, s, a],
                            label=labels[i+1],
                            c=clrs[i+1]
                        )
                        plt.fill_between(
                            np.arange(data[i].shape[1]),
                            means[i][:, s, a]-stds[i][:, s, a],
                            means[i][:, s, a]+stds[i][:, s, a],
                            alpha=0.3,
                            facecolor=clrs[i+1]
                        )
                    plt.legend()
                    plt.title('state-{}-action-{}-values'.format(s, a))
                    plt.ylabel("Q Value")
                    plt.xlabel('Episode')
                    plt.savefig('state-action/s-{}-a-{}.png'.format(s, a))


def plot_V_values(env, star_values, data, labels):
    # star_values: [V_star, q_star]
    # data: [VQ_Logger, DQ1_Logger, DQ2_Logger, PQ_Logger]

    V_star, q_star = star_values
    # Plot true values
    save_value_as_image(env, V_star, labels[0])
    means, stds = [], []
    for idx, val in enumerate(data):
        values = np.zeros((val.shape[0], V_star.shape[0])) # (# of exps, nS)
        for i, Q in enumerate(val[:, -1, :]): # consider final Q values only
            values[i, :] = q_to_v(Q, V_star)
        means.append(values.mean(0))
        stds.append(values.std(0))
        save_value_as_image(env, values.mean(0), labels[idx+1])

    # trace states we are interested in
    pi_star = GreedyPolicy(q_star)
    optimal_traj = []
    done = False
    s = env.reset()
    while not done:
        a = pi_star.action(s)
        optimal_traj.append(s)
        s, _, done, _ = env.step(a)

    bar_width = 0.2
    fig, ax = plt.subplots(figsize=(20, 6), dpi=200)
    with sns.axes_style("darkgrid"):
        # optimal values
        ax.bar(
            np.arange(V_star.shape[0]) - 1.5*bar_width,
            V_star, width=bar_width, label=labels[0]
        )
        ax.bar(
            np.arange(V_star.shape[0]) - 0.7*bar_width,
            means[0], yerr=stds[0], ecolor='black', capsize=2,
            align='center', width=bar_width, label=labels[1]
        )
        ax.bar(np.arange(V_star.shape[0]), means[1], yerr=stds[1],
            ecolor='black', capsize=2, align='center',
            width=bar_width, label=labels[2]
        )
        ax.bar(np.arange(V_star.shape[0]) + 0.7*bar_width, means[2],
            yerr=stds[2], ecolor='black', capsize=2, align='center',
            width=bar_width, label=labels[3]
        )
        ax.bar(np.arange(V_star.shape[0]) + 1.5*bar_width, means[3],
            yerr=stds[3], ecolor='black', capsize=2, align='center',
            width=bar_width, label=labels[4]
        )

        ax.legend()
        ax.set_title("V Value")
        ax.set_xlabel('State #')
    plt.show()
    plt.savefig('state_values.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="Path to Config file", required=True)
    parser.add_argument("--seed", help="set numpy & env seed", type=int, default=0)

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    if config.model == "one_state":
        env = OneStateMDP(rng)
        env_with_model = OneStateMDPWithModel(rng)
    elif config.model == "five_state":
        env = FiveStateMDP(rng)
        env_with_model = FiveStateMDPWithModel(rng)
    elif config.model == "gridworld" or config.model == 'file':
        env = Converted(config.env, args.seed)
        env = EnvModel(env)
        if ast.literal_eval(config.env_mods.delay) is not None:
            env = DelayedReward(env, config.env_mods.delay)
        if config.model == 'file':
            env.load_map(config.map_path)
        _ = env.reset()
        plt.imsave('env.png', env.state)

    V, Q = set_initial_values(config, env)

    # Value Iteration
    print("Value Iteration (DP)")
    if '_state' in config.model:
        V_star, q_star = value_iteration(
            env_with_model, V, config.value_iteration_theta, config.gamma
        )
    else:
        V_star, q_star = value_iteration_gridworld(
            env, V, config.value_iteration_theta, config.gamma
        )


    # Vanilla Q-learning
    print("Vanilla Q Learning")
    VQ_Logger = np.empty((config.exp.repeat, config.q_learning.steps, env.nS, env.nA))
    VQ_EnvLogger = np.empty((config.exp.repeat, config.q_learning.steps, 2))
    for rep in tqdm(range(config.exp.repeat)):
        _, VQ_Logger[rep, ::], VQ_EnvLogger[rep, ::] = Q_learning(
            env,
            config.q_learning.steps,
            config.q_learning.alpha,
            config.gamma,
            config.q_learning.epsilon,
            config.q_learning.decay,
            config.q_learning.interval,
            copy.deepcopy(Q),
        )


    # Double Q-learning
    print("Double Q Learning")
    DQ1_Logger = np.empty((config.exp.repeat, config.q_learning.steps, env.nS, env.nA))
    DQ2_Logger = np.empty((config.exp.repeat, config.q_learning.steps, env.nS, env.nA))
    DQ_EnvLogger = np.empty((config.exp.repeat, config.q_learning.steps, 2))
    for rep in tqdm(range(config.exp.repeat)):
        _, _, DQ1_Logger[rep, ::], DQ2_Logger[rep, ::], DQ_EnvLogger[rep, ::] = DoubleQ(
            env,
            config.dq_learning.steps,
            config.dq_learning.alpha,
            config.gamma,
            config.dq_learning.epsilon,
            config.dq_learning.decay,
            config.dq_learning.interval,
            copy.deepcopy(Q),
            copy.deepcopy(Q),
        )


    print("Pessimistic Q Learning")
    PQ_Logger = np.empty((config.exp.repeat, config.q_learning.steps, env.nS, env.nA))
    PQ_EnvLogger = np.empty((config.exp.repeat, config.q_learning.steps, 2))
    for rep in tqdm(range(config.exp.repeat)):
        _, PQ_Logger[rep, ::], PQ_EnvLogger[rep, ::] = PessimisticQ(
            env,
            config.pessimistic_q_learning.steps,
            config.pessimistic_q_learning.alpha,
            config.gamma,
            config.pessimistic_q_learning.epsilon,
            config.pessimistic_q_learning.decay,
            config.pessimistic_q_learning.interval,
            config.pessimistic_q_learning.pessimism_coeff,
            copy.deepcopy(Q),
        )

    print("Maxmin Q learning")
    MMQ_Logger = np.empty(( config.exp.repeat, config.q_learning.steps, config.mmq_learning.estimators, env.nS, env.nA ))
    MMQ_EnvLogger = np.empty(( config.exp.repeat, config.q_learning.steps, 2 ))
    print(MMQ_Logger.shape)
    for rep in tqdm(range(config.exp.repeat)):
        print(rep)
        _, MMQ_Logger[rep, :, :, :, :], MMQ_EnvLogger[rep, ::] = MaxminQ(
            env,
            config,
            copy.deepcopy(Q),
        )

    # Plot mean cummulative reward
    plot_mean_cum_rewards(
        [VQ_EnvLogger, DQ_EnvLogger, PQ_EnvLogger, MMQ_EnvLogger],
        ['Vanilla Q', 'Double Q', 'Pessimistic Q', 'Minimax Q']
        )

    # plot Q and V
    plot_Q_values(
        env,
        q_star,
        [VQ_Logger, DQ1_Logger, DQ2_Logger, PQ_Logger],
        ['Q Optimal', 'Vanilla Q', 'Double Q1', 'Double Q1', 'Pessimistic Q']
    )
    plot_V_values(
        env,
        [V_star, q_star],
        [VQ_Logger, DQ1_Logger, DQ2_Logger, PQ_Logger],
        ['Q Optimal', 'Vanilla Q', 'Double Q1', 'Double Q1', 'Pessimistic Q']
    )

    save_value_as_image(env, np.arange(env.nS), 'state_indices', int_=True)
