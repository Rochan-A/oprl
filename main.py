import numpy as np
import ast
from os.path import join

import argparse
import yaml
import matplotlib.pyplot as plt
from easydict import EasyDict
import copy
from tqdm import tqdm

from envs import *
from algos import *
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
    rng = np.random.default_rng(args.seed)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    make_dirs(join(args.exp_name))

    if config.model == "one_state":
        env = OneStateMDP(rng)
        env_with_model = OneStateMDPWithModel(rng)
    elif config.model == "five_state":
        env = FiveStateMDP(rng)
        env_with_model = FiveStateMDPWithModel(rng)

    if "file" in config.model or "gridworld" in config.model:

        if config.model == "gridworld" or config.model == 'file':
            env = Converted(config.env, args.seed)
            env = EnvModel(env)

        elif config.model == 'filev2':
            env = Maps(config.env, args.seed)
            env = MapsEnvModel(env)

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
            save_matrix_as_image(env, np.arange(env.nS), join(args.exp_name, 'state_indices.png'), int_=True)
        _ = env.reset()

    # Save experiment logs
    env_loggers = {}
    loggers = {}
    visits = {}

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
        _, VQ_Logger[rep, ::], VQ_EnvLogger[rep, ::], VQ_Visits = Q_learning(
            env,
            config.q_learning.steps,
            config.q_learning.alpha,
            config.gamma,
            config.q_learning.epsilon,
            config.q_learning.decay,
            config.q_learning.interval,
            copy.deepcopy(Q),
        )
    env_loggers['Vanilla Q'] = VQ_EnvLogger
    loggers['Vanilla Q'] = VQ_Logger
    visits['Vanilla Q'] = VQ_Visits


    # Double Q-learning
    print("Double Q Learning")
    DQ1_Logger = np.empty((config.exp.repeat, config.q_learning.steps, env.nS, env.nA))
    DQ2_Logger = np.empty((config.exp.repeat, config.q_learning.steps, env.nS, env.nA))
    DQ_EnvLogger = np.empty((config.exp.repeat, config.q_learning.steps, 2))
    for rep in tqdm(range(config.exp.repeat)):
        _, _, DQ1_Logger[rep, ::], DQ2_Logger[rep, ::], DQ_EnvLogger[rep, ::], DQ_Visits = DoubleQ(
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
    env_loggers['Double Q'] = DQ_EnvLogger
    loggers['Double Q1'] = DQ1_Logger
    loggers['Double Q2'] = DQ2_Logger
    visits['Double Q'] = DQ_Visits


    print("Pessimistic Q Learning")
    PQ_Logger = np.empty((config.exp.repeat, config.q_learning.steps, env.nS, env.nA))
    PQ_EnvLogger = np.empty((config.exp.repeat, config.q_learning.steps, 2))
    for rep in tqdm(range(config.exp.repeat)):
        _, PQ_Logger[rep, ::], PQ_EnvLogger[rep, ::], PQ_Visits = PessimisticQ(
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
    env_loggers['Pessimistic Q'] = PQ_EnvLogger
    loggers['Pessimistic Q'] = PQ_Logger
    visits['Pessimistic Q'] = PQ_Visits


    # Perform Maxmin for 'n' different estimator counts
    for estimators in config.mmq_learning.estimator_pools:
        print("Maxmin Q learning, n = {}".format(estimators))
        MMQ_Logger = np.empty(( config.exp.repeat, config.mmq_learning.steps, estimators, env.nS, env.nA))
        MMQ_EnvLogger = np.empty(( config.exp.repeat, config.mmq_learning.steps, 2))
        for rep in tqdm(range(config.exp.repeat)):
            _, MMQ_Logger[rep, :, :, :, :], MMQ_EnvLogger[rep, ::], MMQ_Visits = MaxminQ(
                env,
                config,
                estimators,
                copy.deepcopy(Q)
            )
        env_loggers['Maxmin Q n_{}'.format(estimators)] = MMQ_EnvLogger
        loggers['Maxmin Q n_{}'.format(estimators)] = MMQ_Logger[:, :, 0, ::]
        visits['Maxmin Q n_{}'.format(estimators)] = MMQ_Visits


    print("Mean-Var Q Learning")
    MVQ_Logger = np.empty((config.exp.repeat, config.meanvar_q_learning.steps, env.nS, env.nA))
    MVQ_EnvLogger = np.empty((config.exp.repeat, config.meanvar_q_learning.steps, 2))
    for rep in tqdm(range(config.exp.repeat)):
        _, MVQ_Logger[rep, ::], MVQ_EnvLogger[rep, ::], MVQ_Visits = MeanVarianceQ(
            env,
            config.meanvar_q_learning.steps,
            config.meanvar_q_learning.alpha,
            config.gamma,
            config.meanvar_q_learning.epsilon,
            config.meanvar_q_learning.decay,
            config.meanvar_q_learning.interval,
            config.meanvar_q_learning.coeff,
            copy.deepcopy(Q),
        )
    env_loggers['Mean-Var Q'] = MVQ_EnvLogger
    loggers['Mean-Var Q'] = MVQ_Logger
    visits['Mean-Var Q'] = MVQ_Visits


    print("Maxmin Bandit Q learning")
    MMBQ_Logger = np.empty((config.exp.repeat, config.mmbq_learning.steps, config.mmbq_learning.max_estimators, env.nS, env.nA))
    MMBQ_EnvLogger = np.empty((config.exp.repeat, config.mmbq_learning.steps, 2))
    for rep in tqdm(range(config.exp.repeat)):
        _, MMBQ_Logger[rep, :, :, :, :], MMBQ_EnvLogger[rep, ::] = MaxminBanditQ(
            env,
            config,
            copy.deepcopy(Q),
        )
    env_loggers['Maxmin Bandit Q'] = MMBQ_EnvLogger
    loggers['Maxmin Bandit Q'] = MMBQ_Logger[:, :, 0, ::]


    # Save experiment values
    save_experiments(
        env_loggers,
        loggers,
        visits,
        args.exp_name
    )

    # Plot visitation heat map
    plot_heatmap(
        visits,
        args.exp_name
        )

    print("Maxmin Bandit Q learning")
    MMBQ_Logger = np.empty(( config.exp.repeat, config.mmbq_learning.steps, config.mmbq_learning.max_estimators, env.nS, env.nA ))
    MMBQ_EnvLogger = np.empty(( config.exp.repeat, config.mmbq_learning.steps, 2 ))
    for rep in tqdm(range(config.exp.repeat)):
        _, MMBQ_Logger[rep, :, :, :, :], MMBQ_EnvLogger[rep, ::] = MaxminBanditQ(
            env,
            config,
            copy.deepcopy(Q),
        )

    # Plot mean cummulative reward
    plot_mean_cum_rewards(
        env_loggers,
        args.exp_name,
        do_smooth=True
        )

    # plot Q and V
    plot_Q_values(
        env,
        q_star,
        loggers,
        args.exp_name
    )

    plot_V_values(
        env,
        [V_star, q_star],
        loggers,
        args.exp_name
    )