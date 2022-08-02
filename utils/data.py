from cProfile import label
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style('darkgrid')

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', titlesize=BIGGER_SIZE)

import bz2
import pickle
import _pickle as cPickle
import pathlib

from policies import GreedyPolicy
# from moviepy.editor import ImageSequenceClip


def make_dirs(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def compress_pickle(fname, data):
    with bz2.BZ2File(fname, 'wb') as f:
        cPickle.dump(data, f)


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


def save_experiments(env_loggers={}, loggers={}, visits={}, exp_name='test'):
    """Save experiment data"""
    make_dirs(join(exp_name, 'store'))
    for key in env_loggers:
        compress_pickle(join(exp_name, 'store', key + '_env.pbz2'), env_loggers[key])

    for key in loggers:
        compress_pickle(join(exp_name, 'store', key + '_qvals.pbz2'), loggers[key])

    for key in visits:
        compress_pickle(join(exp_name, 'store', key + '_visits.pbz2'), visits[key])


def state_values_to_grid(V, env):
    """Convert vector of state values to matrix of env shape."""
    grid = np.zeros((env.state.shape[0], env.state.shape[1]))
    grid[:] = np.nan
    for idx, val in enumerate(V):
        y, x = env.S[idx]
        grid[y, x] = val
    return grid


def save_matrix_as_image(grid, filename, int_=False, cmap=plt.cm.Blues, fmt='png', vmin=-5, vmax=5):
    """Given grid of values, plot and save."""
    fig, ax = plt.subplots(dpi=200)
    ax.matshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(grid.shape[1]):
        for j in range(grid.shape[0]):
            if not np.isnan(grid[j,i]):
                if not int_:
                    c = '{:.2f}'.format(grid[j,i])
                else:
                    c = '{:0.0f}'.format(grid[j,i])
                ax.text(i, j, c, va='center', ha='center', fontsize=6)

    plt.tight_layout()
    ax = plt.gca()

    # Major ticks
    ax.set_xticks(np.arange(0, grid.shape[1], 1))
    ax.set_yticks(np.arange(0, grid.shape[0], 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid.shape[0], 1), minor=True)
    ax.tick_params(colors='w', which='both')

    empty_string_labels = ['']*grid.shape[1]
    ax.set_xticklabels(empty_string_labels)
    empty_string_labels = ['']*grid.shape[0]
    ax.set_yticklabels(empty_string_labels)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    ax.grid(b=False, which='major')

    plt.savefig(filename, format=fmt, bbox_inches='tight')


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


def smooth_exp(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def plot_mean_cum_rewards(data, exp_name, do_smooth=False, std_factor=0.5, fmt='png'):
    """Compute mean and std of env data."""
    labels = list(data.keys())
    means, stds = {}, {}
    for i, key in enumerate(data):
        val = data[key][:, :4000, :]
        # reward
        if do_smooth:
            means[key] = smooth_exp(val[:, :, 0].mean(0, dtype=np.float64), 0.995)
        else:
            means[key] = val[:, :, 0].mean(0, dtype=np.float64)
        stds[key] = val[:, :, 0].std(0, ddof=1, dtype=np.float64)

    fig, ax = plt.subplots(dpi=200)
    for i, key in enumerate(data):
            #data[key].shape[-2]),
        ax.plot(np.arange(
            4000),
            means[key],
            label=labels[i],
        )
        ax.fill_between(
            # np.arange(data[key].shape[-2]),
            np.arange(4000),
            means[key]-(stds[key]*std_factor),
            means[key]+(stds[key]*std_factor),
            alpha=0.1,
        )
        ax.set_xlabel('Steps')
        ax.set_ylabel('Reward')
        plt.tight_layout()
        plt.legend()
        plt.savefig(join(exp_name, 'reward_plot.{}'.format(fmt)), format=fmt, bbox_inches='tight')


def plot_Q_values(env, q_star, data, exp_name, fmt='png'):
    # data: [VQ_Logger, DQ1_Logger, DQ2_Logger, PQ_Logger]
    labels = list(data.keys())
    means, stds = [], []
    for val in data.values():
        means.append(val.mean(0, dtype=np.float64))
        stds.append(val.std(0, ddof=1, dtype=np.float64))

    clrs = sns.color_palette("husl", len(data) + 1)
    make_dirs(join(exp_name, 'state-action'))
    with sns.axes_style("darkgrid"):
        for s in range(env.nS):
            for a in range(4):
                if not (env.TD(s, a) == 0).all():
                    plt.figure(dpi=200)
                    plt.plot(
                        np.arange(data[labels[0]].shape[1]),
                        [q_star[s, a]]*data[labels[0]].shape[1],
                        '--',
                        alpha=0.7,
                        label='Q Optimal',
                        c=clrs[0]
                    )
                    for i, key in enumerate(data):
                        plt.plot(
                            np.arange(data[key].shape[1]),
                            means[i][:, s, a],
                            label=key,
                            c=clrs[i+1]
                        )
                        plt.fill_between(
                            np.arange(data[key].shape[1]),
                            means[i][:, s, a]-stds[i][:, s, a],
                            means[i][:, s, a]+stds[i][:, s, a],
                            alpha=0.3,
                            facecolor=clrs[i+1]
                        )
                    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
                    plt.title('state-{}-action-{}-values'.format(s, a))
                    plt.ylabel("Q Value")
                    plt.xlabel('Episode')
                    plt.tight_layout()
                    plt.savefig(
                        join(
                            exp_name,
                            'state-action',
                            's-{}-a-{}.{}'.format(s, a, fmt)
                        ),
                        format=fmt,
                        bbox_inches='tight'
                    )


def plot_V_values(env, star_values, data, exp_name, fmt='png'):
    # star_values: [V_star, q_star]
    # data: [VQ_Logger, DQ1_Logger, DQ2_Logger, PQ_Logger]
    labels = list(data.keys())
    V_star, q_star = star_values

    if V_star is not None:
        # Plot true values
        grid = state_values_to_grid(V_star, env)
        save_matrix_as_image(grid, join(exp_name, 'Optimal Q.{}'.format(fmt)), vmin=-1.0, vmax=1.0)

    means, stds = [], []
    for idx, val in enumerate(data.values()):
        values = np.zeros((val.shape[0], V_star.shape[0])) # (# of exps, nS)
        for i, Q in enumerate(val[:, -1, :]): # consider final Q values only
            values[i, :] = q_to_v(Q, V_star)
        means.append(values.mean(0))
        stds.append(values.std(0))
        grid = state_values_to_grid(values.mean(0), env)
        save_matrix_as_image(grid, join(exp_name, labels[idx]+'.{}'.format(fmt)), vmin=-1.0, vmax=1.0)

        # Diff between true value and calculated
        a = (values.mean(0) - V_star)/V_star

        grid = state_values_to_grid(a, env)
        save_matrix_as_image(
            grid,
            join(exp_name, '{}_diff.{}'.format(labels[idx], fmt)),
            cmap=plt.cm.coolwarm,
            fmt=fmt)

    # bar_width = 0.2
    # fig, ax = plt.subplots(figsize=(20, 5), dpi=200)
    # with sns.axes_style("darkgrid"):
    #     # optimal values
    #     ax.bar(
    #         np.arange(V_star.shape[0]) - 1.5*bar_width,
    #         V_star, width=bar_width, label=labels[0]
    #     )
    #     ax.bar(
    #         np.arange(V_star.shape[0]) - 0.7*bar_width,
    #         means[0], yerr=stds[0], ecolor='black', capsize=2,
    #         align='center', width=bar_width, label=labels[1]
    #     )
    #     ax.bar(np.arange(V_star.shape[0]), means[1], yerr=stds[1],
    #         ecolor='black', capsize=2, align='center',
    #         width=bar_width, label=labels[2]
    #     )
    #     ax.bar(np.arange(V_star.shape[0]) + 0.7*bar_width, means[2],
    #         yerr=stds[2], ecolor='black', capsize=2, align='center',
    #         width=bar_width, label=labels[3]
    #     )
    #     ax.bar(np.arange(V_star.shape[0]) + 1.5*bar_width, means[3],
    #         yerr=stds[3], ecolor='black', capsize=2, align='center',
    #         width=bar_width, label=labels[4]
    #     )
    #     ax.bar(np.arange(V_star.shape[0]) + 2*bar_width, means[4],
    #         yerr=stds[4], ecolor='black', capsize=2, align='center',
    #         width=bar_width, label=labels[5]
    #     )

    #     ax.legend()
    #     ax.set_title("V Value")
    #     ax.set_xlabel('State #')
    #     plt.tight_layout()
    #     plt.savefig(join(exp_name, 'state_values.png'), bbox_inches='tight')


def plot_visits(visits, s_a, exp_name, fmt='png'):
    make_dirs(join(exp_name, 'visits'))

    labels = list(visits.keys())
    for _, key in enumerate(visits.keys()):
        visit = visits[key]
        visit_mean = visit.mean(0, dtype=np.float64)
        visit_std = visit.std(0, ddof=1, dtype=np.float64)
        # clip = ImageSequenceClip(list(visit), fps=fps)
        # clip.write_gif(join(exp_name, 'visits', '{}.gif'.format(labels[idx])), fps=fps)

        labels = ['left', 'right', 'up', 'down']
        for idx, state in enumerate(s_a):
            fig, ax = plt.subplots(dpi=200)
            sum_ = np.sum(visit, axis=0)
            for act in range(4):
                y = smooth(visit_mean[:, state, act], 10)
                ax.plot(np.arange(0, visit.shape[1]), y, label=labels[act])
                ax.fill_between(
                    np.arange(0, visit.shape[1]),
                    y-(visit_std[:, state, act]*0.5),
                    y+(visit_std[:, state, act]*0.5),
                    alpha=0.1
                )

            ax.set_title("Action Freq @ State {}".format(state))
            ax.set_xlabel('Episode')
            ax.set_ylabel('Mean Count')
            plt.tight_layout()
            plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
            plt.savefig(join(exp_name, 'visits', '{}-s-{}.{}'.format(key, state, fmt)), format=fmt, bbox_inches='tight')


def plot_bandit(bandit, exp_name, fmt='png'):
    make_dirs(join(exp_name, 'bandit'))

    # First, plot bandit q values
    for key in bandit['q']:
        labels = list(bandit['q'].keys())
        q_vals_mean = bandit['q'][key].mean(0, dtype=np.float64)
        q_vals_std = bandit['q'][key].std(0, ddof=1, dtype=np.float64)

        fig, ax = plt.subplots(dpi=200)
        for i in range(q_vals_mean.shape[-2]):
            y = smooth(q_vals_mean[:, i, 0], 10)
            ax.plot(np.arange(0, q_vals_mean.shape[0]), y, label='n={}'.format(i+1))
            ax.fill_between(
                np.arange(0, q_vals_mean.shape[0]),
                y-(q_vals_std[:, i, 0]*0.5),
                y+(q_vals_std[:, i, 0]*0.5),
                alpha=0.1
            )

        # ax.set_title("Bandit Q Value ({})".format(key))
        ax.set_xlabel('Episode')
        ax.set_ylabel('Q Value')
        plt.tight_layout()
        plt.legend()#bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        plt.savefig(join(exp_name, 'bandit', '{}-q_val.{}'.format(key, fmt)), format=fmt, bbox_inches='tight')

    # Plot estimator count selection
    for key in bandit['est']:
        labels = list(bandit['est'].keys())
        est_selected = bandit['est'][key][0, ::]

        fig, ax = plt.subplots(dpi=200)
        ax.scatter(np.arange(0, len(est_selected)), est_selected, marker='|', s=200)
        # ax.set_title("Bandit Selected Estimators ({})".format(key))
        ax.set_xlabel('Episode') # **** change to 'Steps' for Mountain Car ****
        ax.set_ylabel('Estimator')
        plt.tight_layout()
        # plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        plt.savefig(join(exp_name, 'bandit', '{}-est.{}'.format(key, fmt)), format=fmt, bbox_inches='tight')
