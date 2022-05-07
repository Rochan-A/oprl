import numpy as np
import ast
from os.path import join
import json

import argparse
import yaml
import matplotlib.pyplot as plt
from easydict import EasyDict
import copy
from tqdm import tqdm

from envs import *
from algos.dp import *
from algos.q import *
from utils.data import *


def set_initial_values(config, env):
    """Set the initial values for Q and V."""

    if config.init.Q == 'zero':
        Q = np.zeros((env.nS, env.nA), dtype=np.float64)
    elif config.init.Q == 'rand':
        Q = np.random.normal(0, 0.01, (env.nS, env.nA)).astype(dtype=np.float64)
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
        V = np.random.normal(0, 0.01, (env.nS,)).astype(dtype=np.float64)
    elif config.init.V == 'opt':
        V = np.zeros((env.nS,), dtype=np.float64)
        V.fill(1.0)
    elif config.init.V == 'pes':
        V = np.zeros((env.nS,), dtype=np.float64)
        V.fill(-1.0)
    else:
        assert False, 'unknown V value initialization'
    return V, Q


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="Path to Config file", required=True)
    parser.add_argument("--seed", help="set numpy & env seed", type=int, default=0)
    parser.add_argument("-e", "--exp_name", help="Experiment name", type=str, required=True)

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    rng = np.random.default_rng(config.seed)

    make_dirs(join(args.exp_name, 'store'))
    with open(join(args.exp_name, 'config.yaml'), 'w') as f: 
        json.dump(config, f, indent=2)

    if config.model == "one_state":
        env = OneStateMDP(rng)
        env_with_model = OneStateMDPWithModel(rng)
    elif config.model == "five_state":
        env = FiveStateMDP(rng)
        env_with_model = FiveStateMDPWithModel(rng)
    elif "file" in config.model or "gridworld" in config.model:

        # Gridworld or old env
        if config.model == "gridworld" or config.model == 'file':
            env = Converted(config.env, config.seed)
            env = EnvModel(env)
            _ = env.reset()
            plt.imsave(join(args.exp_name, 'env.png'), env.state)

            grid = state_values_to_grid(np.arange(0, env.S.shape[0], 1), env)
            save_matrix_as_image(
                grid,
                join(args.exp_name, 'state_indices.png'),
                int_=True
            )

        # New env
        elif config.model == 'filev2':
            env = Maps(config.env, config.seed)
            env = MapsEnvModel(env)

        # Apply reward modifiers
        if config.env_mods.state_bonus:
            env = StateBonus(env)
        if config.env_mods.action_bonus:
            env = ActionBonus(env)
        if config.env_mods.distance_bonus:
            env = DistanceBonus(env)
        if ast.literal_eval(config.env_mods.delay) is not None:
            env = DelayedReward(env, config.env_mods.delay)

        if 'file' in config.model:
            env.load_map(config.map_path)
            _ = env.reset()
            plt.imsave(join(args.exp_name, 'env.png'), env.map)
            grid = state_values_to_grid(np.arange(env.nS), env)
            save_matrix_as_image(
                grid,
                join(args.exp_name, 'state_indices.png'),
                int_=True
            )

    else:
        assert False, 'Invalid env model args...'
        quit()

    # Save experiment logs
    env_loggers = {}    # Log rewards, (config.exp.repeat, config.exp.steps)
    loggers = {}        # Log Q values, (config.exp.repeat, config.exp.steps, env.nS, env.nA)
    visits = {}         # Log State-action visits, (config.exp.repeat, config.exp.steps, env.nS, env.nA)

    V, Q = set_initial_values(config, env)

    # Value Iteration
    if 'val_iter' in config.perform:
        print("Value Iteration (DP)")
        if '_state' in config.model:
            V_star, q_star = value_iteration(
                env_with_model, V, config.value_iteration_theta, config.gamma
            )
        else:
            V_star, q_star = value_iteration_gridworld(
                env, V, config.value_iteration_theta, config.gamma
            )
        compress_pickle(join(args.exp_name, 'store', 'Optimal.pbz2'), {'v_star': V_star, 'Q_star': q_star})


    # Vanilla Q-learning
    if 'q' in config.perform:
        print("Vanilla Q Learning")
        VQ_Logger = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA))
        VQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        VQ_Visits = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA), dtype=np.int64)
        for rep in tqdm(range(config.exp.repeat)):
            _, VQ_Logger[rep, ::], VQ_EnvLogger[rep, ::], VQ_Visits[rep, ::] = Q_learning(
                env,
                config,
                copy.deepcopy(Q),
            )
        env_loggers['Vanilla Q'] = VQ_EnvLogger
        loggers['Vanilla Q'] = VQ_Logger
        visits['Vanilla Q'] = VQ_Visits


    # Double Q-learning
    if 'dq' in config.perform:
        print("Double Q Learning")
        DQ1_Logger = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA))
        DQ2_Logger = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA))
        DQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        DQ_Visits = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA), dtype=np.int64)
        for rep in tqdm(range(config.exp.repeat)):
            _, _, DQ1_Logger[rep, ::], DQ2_Logger[rep, ::], DQ_EnvLogger[rep, ::], DQ_Visits[rep, ::] = DoubleQ(
                env,
                config,
                copy.deepcopy(Q)
            )
        env_loggers['Double Q'] = DQ_EnvLogger
        loggers['Double Q1'] = DQ1_Logger
        loggers['Double Q2'] = DQ2_Logger
        visits['Double Q'] = DQ_Visits


    # Pessimistic Q-learning
    if 'pq' in config.perform:
        print("Pessimistic Q Learning")
        PQ_Logger = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA))
        PQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        PQ_Visits = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA), dtype=np.int64)
        for rep in tqdm(range(config.exp.repeat)):
            _, PQ_Logger[rep, ::], PQ_EnvLogger[rep, ::], PQ_Visits[rep, ::] = PessimisticQ(
                env,
                config,
                copy.deepcopy(Q),
            )
        env_loggers['Pessimistic Q'] = PQ_EnvLogger
        loggers['Pessimistic Q'] = PQ_Logger
        visits['Pessimistic Q'] = PQ_Visits


    # Perform Maxmin for 'n' different estimator counts
    if 'mmq' in config.perform:
        for estimators in config.mmq_learning.estimator_pools:
            print("Maxmin Q learning, n = {}".format(estimators))
            MMQ_Logger = np.empty(( config.exp.repeat, config.exp.steps, env.nS, env.nA))
            MMQ_EnvLogger = np.empty(( config.exp.repeat, config.exp.steps, 2))
            MMQ_Visits = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA), dtype=np.int64)
            for rep in tqdm(range(config.exp.repeat)):
                _, MMQ_Logger[rep, ::], MMQ_EnvLogger[rep, ::], MMQ_Visits[rep, ::] = MaxminQ(
                    env,
                    config,
                    estimators,
                    copy.deepcopy(Q)
                )
            env_loggers['Maxmin Q n_{}'.format(estimators)] = MMQ_EnvLogger
            loggers['Maxmin Q n_{}'.format(estimators)] = MMQ_Logger
            visits['Maxmin Q n_{}'.format(estimators)] = MMQ_Visits


    # Mean Variance Q-learning
    if 'meanvar' in config.perform:
        print("Mean-Var Q Learning")
        MVQ_Logger = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA))
        MVQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        MVQ_Visits = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA), dtype=np.int64)
        for rep in tqdm(range(config.exp.repeat)):
            _, MVQ_Logger[rep, ::], MVQ_EnvLogger[rep, ::], MVQ_Visits[rep, ::] = MeanVarianceQ(
                env,
                config,
                copy.deepcopy(Q),
            )
        env_loggers['Mean-Var Q'] = MVQ_EnvLogger
        loggers['Mean-Var Q'] = MVQ_Logger
        visits['Mean-Var Q'] = MVQ_Visits


    # Maxmin Bandit Q learning
    if 'mmbq' in config.perform:
        print("Maxmin Bandit Q learning")
        MMBQ_Logger = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA))
        MMBQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        MMBQ_Visits = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA), dtype=np.int64)
        MMBQ_Bandit = np.empty((config.exp.repeat, config.exp.steps, config.mmbq_learning.max_estimators, 2))
        MMBQ_Estimator = np.empty((config.exp.repeat, config.exp.steps), dtype=np.int64)
        for rep in tqdm(range(config.exp.repeat)):
            _, MMBQ_Logger[rep, :, :, :], MMBQ_EnvLogger[rep, ::], MMBQ_Visits[rep, ::], MMBQ_Bandit[rep, ::], MMBQ_Estimator[rep] = MaxminBanditQ(
                env,
                config,
                copy.deepcopy(Q),
            )
        env_loggers['Maxmin Bandit Q'] = MMBQ_EnvLogger
        loggers['Maxmin Bandit Q'] = MMBQ_Logger
        visits['Maxmin Bandit Q'] = MMBQ_Visits
        compress_pickle(join(args.exp_name, 'store', 'Maxmin Bandit Q_bandit.pbz2'), MMBQ_Bandit)
        compress_pickle(join(args.exp_name, 'store', 'Maxmin Bandit Q_estimator.pbz2'), MMBQ_Estimator)


    # Maxmin Bandit Q learning v2
    if 'mmbq_v2' in config.perform:
        print("Maxmin Bandit Q learning V2")
        MMBQ_Logger = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA))
        MMBQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        MMBQ_Visits = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA), dtype=np.int64)
        MMBQ_Bandit = np.empty((config.exp.repeat, config.exp.steps, config.mmbq_learning.max_estimators, 2))
        MMBQ_Estimator = np.empty((config.exp.repeat, config.exp.steps), dtype=np.int64)
        for rep in tqdm(range(config.exp.repeat)):
            _, MMBQ_Logger[rep, :, :, :], MMBQ_EnvLogger[rep, ::], MMBQ_Visits[rep, ::], MMBQ_Bandit[rep, ::], MMBQ_Estimator[rep] = MaxminBanditQ_v2(
                env,
                config,
                copy.deepcopy(Q),
            )
        env_loggers['Maxmin Bandit Q v2'] = MMBQ_EnvLogger
        loggers['Maxmin Bandit Q v2'] = MMBQ_Logger
        visits['Maxmin Bandit Q v2'] = MMBQ_Visits
        compress_pickle(join(args.exp_name, 'store', 'Maxmin Bandit Q v2_bandit.pbz2'), MMBQ_Bandit)
        compress_pickle(join(args.exp_name, 'store', 'Maxmin Bandit Q v2_estimator.pbz2'), MMBQ_Estimator)


    # # Save experiment values
    save_experiments(
        env_loggers,
        loggers,
        visits,
        args.exp_name
    )

    # # Plot mean cummulative reward
    # plot_mean_cum_rewards(
    #     env_loggers,
    #     args.exp_name,
    #     do_smooth=True,
    #     std_factor=0.5
    #     )

    # # plot Q and V
    # plot_Q_values(
    #     env,
    #     q_star,
    #     loggers,
    #     args.exp_name
    # )

    # plot_V_values(
    #     env,
    #     [V_star, q_star],
    #     loggers,
    #     args.exp_name
    # )

    # # Plot visitation heat map
    # plot_heatmap(
    #     visits,
    #     args.exp_name
    # )
