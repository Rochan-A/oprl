from asyncio.log import logger
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

from envs import (
    Converted,
    EnvModel,
    Maps,
    MapsEnvModel,
    StateBonus,
    ActionBonus,
    DistanceBonus,
    NoisyReward,
    DelayedReward
)
from algos.dp import value_iteration_gridworld
from algos.q import Q_learning, DoubleQ, MaxminQ, MaxminBanditQ, MaxminBanditQ_v2
from utils.data import *
from multiprocessing import Pool


def set_initial_values(config, env, rng):
    """Set the initial values for Q and V."""

    if config.init.Q == 'zero':
        Q = np.zeros((env.nS, env.nA), dtype=np.float64)
    elif config.init.Q == 'rand':
        Q = rng.normal(0, 0.01, (env.nS, env.nA)).astype(dtype=np.float64)
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
        V = rng.normal(0, 0.01, (env.nS,)).astype(dtype=np.float64)
    elif config.init.V == 'opt':
        V = np.zeros((env.nS,), dtype=np.float64)
        V.fill(1.0)
    elif config.init.V == 'pes':
        V = np.zeros((env.nS,), dtype=np.float64)
        V.fill(-1.0)
    else:
        assert False, 'unknown V value initialization'
    return V, Q


def _wrapper(func, args):
    """Wrapper to call function and pass dict args"""
    return func(**args)


def q_learning(Q, config, env, rng, env_loggers, loggers, visits):
    """Q Learning runner"""
    for _, lr in enumerate(config.q_learning.alpha):
        VQ_Logger = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA))
        VQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        VQ_Visits = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA), dtype=np.int64)

        data = {
            'env': env,
            'config': config,
            'Q': copy.deepcopy(Q),
            'rng': rng,
            'alpha': lr
        }

        with Pool(processes=config.exp.repeat) as pool:
            results = [pool.apply_async(_wrapper, args=(Q_learning, data,)) for i in range(config.exp.repeat)]
            returns = [r.get() for r in tqdm(results)]

        for rep in range(config.exp.repeat):
            VQ_Logger[rep, ::] = returns[rep]['QLog']
            VQ_EnvLogger[rep, ::] = returns[rep]['EnvLog']
            VQ_Visits[rep, ::] = returns[rep]['VisitLog']

        env_loggers['Vanilla Q (lr={})'.format(lr)] = VQ_EnvLogger
        loggers['Vanilla Q (lr={})'.format(lr)] = VQ_Logger
        visits['Vanilla Q (lr={})'.format(lr)] = VQ_Visits

    return env_loggers, loggers, visits


def dq_learning(Q, config, env, rng, env_loggers, loggers, visits):
    """DQ Learning runner"""
    for _, lr in enumerate(config.dq_learning.alpha):
        DQ1_Logger = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA))
        DQ2_Logger = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA))
        DQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        DQ_Visits = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA), dtype=np.int64)

        data = {
            'env': env,
            'config': config,
            'Q': copy.deepcopy(Q),
            'rng': rng,
            'alpha': lr
        }

        with Pool(processes=config.exp.repeat) as pool:
            results = [pool.apply_async(_wrapper, args=(DoubleQ, data,)) for i in range(config.exp.repeat)]
            returns = [r.get() for r in tqdm(results)]

        for rep in range(config.exp.repeat):
            DQ1_Logger[rep, ::] = returns[rep]['QLog']
            DQ_EnvLogger[rep, ::] = returns[rep]['EnvLog']
            DQ_Visits[rep, ::] = returns[rep]['VisitLog']

        env_loggers['Double Q (lr={})'.format(lr)] = DQ_EnvLogger
        loggers['Double Q1 (lr={})'.format(lr)] = DQ1_Logger
        loggers['Double Q2 (lr={})'.format(lr)] = DQ2_Logger
        visits['Double Q (lr={})'.format(lr)] = DQ_Visits

    return env_loggers, loggers, visits


def mmq_learning(Q, config, env, rng, estimators, env_loggers, loggers, visits):
    """MMQ runner"""

    for _, lr in enumerate(config.mmq_learning.alpha):
        MMQ_Logger = np.empty(( config.exp.repeat, config.exp.steps, env.nS, env.nA))
        MMQ_EnvLogger = np.empty(( config.exp.repeat, config.exp.steps, 2))
        MMQ_Visits = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA), dtype=np.int64)

        data = {
            'env': env,
            'config': config,
            'Q': copy.deepcopy(Q),
            'rng': rng,
            'alpha': lr,
            'estimators': estimators
        }

        with Pool(processes=config.exp.repeat) as pool:
            results = [pool.apply_async(_wrapper, args=(MaxminQ, data,)) for i in range(config.exp.repeat)]
            returns = [r.get() for r in tqdm(results)]

        for rep in range(config.exp.repeat):
            MMQ_Logger[rep, ::] = returns[rep]['QLog']
            MMQ_EnvLogger[rep, ::] = returns[rep]['EnvLog']
            MMQ_Visits[rep, ::] = returns[rep]['VisitLog']

        env_loggers['Maxmin Q n={} (lr={})'.format(estimators, lr)] = MMQ_EnvLogger
        loggers['Maxmin Q n={} (lr={})'.format(estimators, lr)] = MMQ_Logger
        visits['Maxmin Q n={} (lr={})'.format(estimators, lr)] = MMQ_Visits

    return env_loggers, loggers, visits


def mmbq_learning(Q, config, env, rng, env_loggers, loggers, visits, exp_name):
    """MMBQ runner"""
    for _, lr in enumerate(config.mmbq_learning.alpha):
        MMBQ_Logger = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA))
        MMBQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        MMBQ_Visits = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA), dtype=np.int64)
        MMBQ_Bandit = np.empty((config.exp.repeat, config.exp.steps, config.mmbq_learning.max_estimators, 2))
        MMBQ_Estimator = np.empty((config.exp.repeat, config.exp.steps), dtype=np.int64)

        data = {
            'env': env,
            'config': config,
            'Q': copy.deepcopy(Q),
            'rng': rng,
            'alpha': lr,
        }

        with Pool(processes=config.exp.repeat) as pool:
            results = [pool.apply_async(_wrapper, args=(MaxminBanditQ, data,)) for i in range(config.exp.repeat)]
            returns = [r.get() for r in tqdm(results)]

        for rep in range(config.exp.repeat):
            MMBQ_Logger[rep, ::] = returns[rep]['QLog']
            MMBQ_EnvLogger[rep, ::] = returns[rep]['EnvLog']
            MMBQ_Visits[rep, ::] = returns[rep]['VisitLog']
            MMBQ_Bandit[rep, ::] = returns[rep]['BanditLog']
            MMBQ_Estimator[rep, ::] = returns[rep]['EstimLog']

        env_loggers['Maxmin Bandit Q (lr={})'.format(lr)] = MMBQ_EnvLogger
        loggers['Maxmin Bandit Q (lr={})'.format(lr)] = MMBQ_Logger
        visits['Maxmin Bandit Q (lr={})'.format(lr)] = MMBQ_Visits

        compress_pickle(join(exp_name, 'store', 'Maxmin Bandit Q_bandit (lr={}).pbz2'.format(lr)), MMBQ_Bandit)
        compress_pickle(join(exp_name, 'store', 'Maxmin Bandit Q_estimator (lr={}).pbz2'.format(lr)), MMBQ_Estimator)

    return env_loggers, loggers, visits


def mmbq_v2_learning(Q, config, env, rng, env_loggers, loggers, visits, exp_name):
    """MMBQ V2 learner"""
    for _, lr in enumerate(config.mmbq_learning.alpha):
        MMBQ_Logger = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA))
        MMBQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        MMBQ_Visits = np.empty((config.exp.repeat, config.exp.steps, env.nS, env.nA), dtype=np.int64)
        MMBQ_Bandit = np.empty((config.exp.repeat, config.exp.steps, config.mmbq_learning.max_estimators, 2))
        MMBQ_Estimator = np.empty((config.exp.repeat, config.exp.steps), dtype=np.int64)

        data = {
            'env': env,
            'config': config,
            'Q': copy.deepcopy(Q),
            'rng': rng,
            'alpha': lr,
        }

        with Pool(processes=config.exp.repeat) as pool:
            results = [pool.apply_async(_wrapper, args=(MaxminBanditQ_v2, data,)) for i in range(config.exp.repeat)]
            returns = [r.get() for r in tqdm(results)]

        for rep in range(config.exp.repeat):
            MMBQ_Logger[rep, ::] = returns[rep]['QLog']
            MMBQ_EnvLogger[rep, ::] = returns[rep]['EnvLog']
            MMBQ_Visits[rep, ::] = returns[rep]['VisitLog']
            MMBQ_Bandit[rep, ::] = returns[rep]['BanditLog']
            MMBQ_Estimator[rep, ::] = returns[rep]['EstimLog']

        env_loggers['Maxmin Bandit Q v2 (lr={})'.format(lr)] = MMBQ_EnvLogger
        loggers['Maxmin Bandit Q v2 (lr={})'.format(lr)] = MMBQ_Logger
        visits['Maxmin Bandit Q v2 (lr={})'.format(lr)] = MMBQ_Visits

        compress_pickle(join(exp_name, 'store', 'Maxmin Bandit Q v2_bandit (lr={}).pbz2'.format(lr)), MMBQ_Bandit)
        compress_pickle(join(exp_name, 'store', 'Maxmin Bandit Q v2_estimator (lr={}).pbz2'.format(lr)), MMBQ_Estimator)

    return env_loggers, loggers, visits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to Config file", required=True)
    parser.add_argument("--seed", help="set numpy & env seed", type=int, default=42)
    parser.add_argument("-e", "--exp_name", help="Experiment name", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    rng = np.random.default_rng(config.seed)

    # Create output dir and save config state
    make_dirs(join(args.exp_name, 'store'))
    with open(join(args.exp_name, 'config.yaml'), 'w') as f: 
        json.dump(config, f, indent=2)

    if "file" in config.model or "gridworld" in config.model:

        # Gridworld or old env
        if config.model == "gridworld" or config.model == 'file':
            env = Converted(config.env, rng, config.seed)
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
            env = Maps(config.env, rng, config.seed)
            env = MapsEnvModel(env)

        # Apply reward modifiers
        if config.env_mods.state_bonus:
            env = StateBonus(env)
        if config.env_mods.action_bonus:
            env = ActionBonus(env)
        if config.env_mods.action_bonus:
            env = NoisyReward(env, rng)
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

    # Save experiment logs
    env_loggers = {}    # Log rewards, (config.exp.repeat, config.exp.steps)
    loggers = {}        # Log Q values, (config.exp.repeat, config.exp.steps, env.nS, env.nA)
    visits = {}         # Log State-action visits, (config.exp.repeat, config.exp.steps, env.nS, env.nA)

    V, Q = set_initial_values(config, env, rng)

    # Value Iteration
    if 'val_iter' in config.perform:
        print("Value Iteration (DP)")
        V_star, q_star = value_iteration_gridworld(
            env, V, config.value_iteration_theta, config.gamma
        )
        compress_pickle(join(args.exp_name, 'store', 'Optimal.pbz2'), {'v_star': V_star, 'Q_star': q_star})

    # Vanilla Q-learning
    if 'q' in config.perform:
        print("Vanilla Q Learning")
        env_loggers, loggers, visits = q_learning(Q, config, env, rng, env_loggers, loggers, visits)

        # Save experiment values
        save_experiments(
            env_loggers,
            loggers,
            visits,
            args.exp_name
        )

    # Double Q-learning
    if 'dq' in config.perform:
        print("Double Q Learning")
        env_loggers, loggers, visits = dq_learning(Q, config, env, rng, env_loggers, loggers, visits)

        # Save experiment values
        save_experiments(
            env_loggers,
            loggers,
            visits,
            args.exp_name
        )


    # Perform Maxmin for 'n' different estimator counts
    if 'mmq' in config.perform:
        for estimators in config.mmq_learning.estimator_pools:
            print("Maxmin Q learning, n = {}".format(estimators))
            env_loggers, loggers, visits = mmq_learning(Q, config, env, rng, estimators, env_loggers, loggers, visits)

        # Save experiment values
        save_experiments(
            env_loggers,
            loggers,
            visits,
            args.exp_name
        )

    # Maxmin Bandit Q learning
    if 'mmbq' in config.perform:
        print("Maxmin Bandit Q learning")
        env_loggers, loggers, visits = mmbq_learning(Q, config, env, rng, env_loggers, loggers, visits, args.exp_name)

        # Save experiment values
        save_experiments(
            env_loggers,
            loggers,
            visits,
            args.exp_name
        )

    # Maxmin Bandit Q learning v2
    if 'mmbq_v2' in config.perform:
        print("Maxmin Bandit Q learning V2")
        env_loggers, loggers, visits = mmbq_v2_learning(Q, config, env, rng, env_loggers, loggers, visits, args.exp_name)

        # Save experiment values
        save_experiments(
            env_loggers,
            loggers,
            visits,
            args.exp_name
        )