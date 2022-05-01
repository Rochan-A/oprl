import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns

import bz2
import pickle
import _pickle as cPickle
import pathlib

from policies import GreedyPolicy
from moviepy.editor import ImageSequenceClip


def make_dirs(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def compress_pickle(fname, data):
    with bz2.BZ2File(fname, 'wb') as f:
        cPickle.dump(data, f)


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


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


def save_matrix_as_image(env, V, filename, int_=False, cmap=plt.cm.Blues):
    grid = np.zeros((env.state.shape[0], env.state.shape[1]))
    for idx, val in enumerate(V):
        y, x = env.S[idx]
        grid[y, x] = val

    fig, ax = plt.subplots(dpi=200)
    ax.matshow(grid, cmap=cmap)
    for i in range(env.state.shape[1]):
        for j in range(env.state.shape[0]):
            if not int_:
                c = '{:.2f}'.format(grid[j,i])
            else:
                c = '{:0.0f}'.format(grid[j,i])
            ax.text(i, j, c, va='center', ha='center', fontsize=6)

    plt.tight_layout()
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight')


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


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def plot_mean_cum_rewards(data, labels, env_name, do_smooth=False):
    """Compute mean and std of env data."""
    means, stds = [], []
    for val in data:
        # reward
        means.append(val[:, :, 0].mean(0, dtype=np.float64))
        stds.append(val[:, :, 0].std(0, ddof=1, dtype=np.float64))

    if do_smooth:
        means = smooth(means, 0.9)

    fig, ax = plt.subplots(dpi=200)
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
    plt.tight_layout()
    plt.savefig(join(env_name, 'reward_plot.png'), bbox_inches='tight')


def plot_Q_values(env, q_star, data, labels, env_name):
    # data: [VQ_Logger, DQ1_Logger, DQ2_Logger, PQ_Logger]
    means, stds = [], []
    for val in data:
        means.append(val.mean(0, dtype=np.float64))
        stds.append(val.std(0, ddof=1, dtype=np.float64))

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
    make_dirs(join(env_name, 'state-action'))
    with sns.axes_style("darkgrid"):
        for s in range(env.nS):
            for a in range(4):
                if not (env.TD(s, a) == 0).all():
                    plt.figure(dpi=200)
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
                    plt.tight_layout()
                    plt.savefig(join(env_name, 'state-action', 's-{}-a-{}.png'.format(s, a)), bbox_inches='tight')


def plot_V_values(env, star_values, data, labels, env_name):
    # star_values: [V_star, q_star]
    # data: [VQ_Logger, DQ1_Logger, DQ2_Logger, PQ_Logger]

    V_star, q_star = star_values
    # Plot true values
    save_matrix_as_image(env, V_star, join(env_name, '{}.png'.format(labels[0])))
    means, stds = [], []
    for idx, val in enumerate(data):
        values = np.zeros((val.shape[0], V_star.shape[0])) # (# of exps, nS)
        for i, Q in enumerate(val[:, -1, :]): # consider final Q values only
            values[i, :] = q_to_v(Q, V_star)
        means.append(values.mean(0))
        stds.append(values.std(0))
        save_matrix_as_image(env, values.mean(0), join(env_name, labels[idx+1]+'.png'))

        # Diff between true value and calculated
        a = (values.mean(0) - V_star)/V_star
        grid = np.zeros((env.state.shape[0], env.state.shape[1]))
        grid[:] = np.nan
        for i, val in enumerate(a):
            if val != 0:
                y, x = env.S[i]
                grid[y, x] = val

        fig, ax = plt.subplots(dpi=200)
        ax.matshow(grid, cmap=plt.cm.inferno)
        for i in range(grid.shape[1]):
            for j in range(grid.shape[0]):
                c = '{:0.2f}'.format(grid[j,i])
                ax.text(i, j, c, va='center', ha='center', fontsize=6, backgroundcolor='white')
        plt.tight_layout()
        plt.savefig(join(env_name, '{}_diff.png'.format(labels[idx+1])), bbox_inches='tight')

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
    fig, ax = plt.subplots(figsize=(20, 5), dpi=200)
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
        ax.bar(np.arange(V_star.shape[0]) + 2*bar_width, means[4],
            yerr=stds[4], ecolor='black', capsize=2, align='center',
            width=bar_width, label=labels[5]
        )

        ax.legend()
        ax.set_title("V Value")
        ax.set_xlabel('State #')
    plt.show()
    plt.tight_layout()
    plt.savefig(join(env_name, 'state_values.png'), bbox_inches='tight')


def save_experiments(env_loggers=[], loggers=[], visits=[], env_labels=[], labels=[], env_name='.'):
    """Save experiment data"""
    for idx, env_logger in enumerate(env_loggers):
        compress_pickle(join(env_name, env_labels[idx] + '.pbz2'), env_logger)

    for idx, logger in enumerate(loggers):
        compress_pickle(join(env_name, labels[idx] + '.pbz2'), logger)

    for idx, visit in enumerate(visits):
        compress_pickle(join(env_name, env_labels[idx] + '.pbz2'), visit)


def plot_heatmap(visits, labels, env_name):
    for idx, visit in enumerate(visits):
        clip = ImageSequenceClip(list(visit), fps=10)
        clip.write_gif(join(env_name, '{}.gif'.format(labels[idx])), fps=10)

        fig, ax = plt.subplots(dpi=200)

        ax.matshow(visit[-2, :, :], cmap=plt.cm.inferno)
        for i in range(visit[-2, :, :].shape[1]):
            for j in range(visit[-2, :, :].shape[0]):
                c = '{:0.0f}'.format(visit[-2, j,i, 0])
                ax.text(i, j, c, va='center', ha='center', fontsize=6, backgroundcolor='white')

        plt.tight_layout()
        plt.savefig(join(env_name, '{}_visit_freq.png'.format(labels[idx])), bbox_inches='tight')