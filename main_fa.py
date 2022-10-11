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
from algos.tc_q import *
from utils.data import *
from policies import *

from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

def set_initial_values(config, env, rng):
    """Set the initial values for Q and V."""

    # number of tile spanning each dimension
    tiles_per_dim = [config.tc.n_bins for _ in range(env.observation_space.shape[0])]
    # value limits of each dimension
    lims = [(env.observation_space.low[idx], env.observation_space.high[idx]) for idx in range(env.observation_space.shape[0])]
    # number of tilings
    tilings = config.tc.num_tiles

    T = TileCoder(tiles_per_dim, lims, tilings)

    if config.init.Q == 'zero':
        Q = np.zeros((T.n_tiles, env.action_space.n), dtype=np.float64)
    elif config.init.Q == 'rand':
        Q = rng.normal(0, 0.01, (T.n_tiles, env.action_space.n)).astype(dtype=np.float64)
    elif config.init.Q == 'opt':
        Q = np.zeros((T.n_tiles, env.action_space.n), dtype=np.float64)
        Q.fill(1.0)
    elif config.init.Q == 'pes':
        Q = np.zeros((T.n_tiles, env.action_space.n), dtype=np.float64)
        Q.fill(-1.0)
    else:
        assert False, 'unknown Q value initialization'

    return T, Q


def _wrapper(func, args):
    """Wrapper to call function and pass dict args"""
    return func(**args)


def q_learning(env, config, T, Q, rng, env_loggers, loggers):
    """Q Learning runner"""
    for _, lr in enumerate(config.q_learning.alpha):
        VQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))

        data = {
            'env': env,
            'config': config,
            'Q': copy.deepcopy(Q),
            'T': copy.deepcopy(T),
            'rng': rng,
            'alpha': lr
        }

        with Pool(processes=config.exp.repeat) as pool:
            results = [pool.apply_async(_wrapper, args=(Q_learning, data,)) for i in range(config.exp.repeat)]
            returns = [r.get() for r in tqdm(results)]

        for rep in range(config.exp.repeat):
            VQ_EnvLogger[rep, ::] = returns[rep]['EnvLog']

        env_loggers['Vanilla Q (lr={})'.format(lr)] = VQ_EnvLogger

    return env_loggers, loggers


def dq_learning(env, config, T, Q, rng, env_loggers, loggers):
    """DQ Learning runner"""
    for _, lr in enumerate(config.dq_learning.alpha):
        DQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))

        data = {
            'env': env,
            'config': config,
            'Q': copy.deepcopy(Q),
            'T': copy.deepcopy(T),
            'rng': rng,
            'alpha': lr
        }

        with Pool(processes=config.exp.repeat) as pool:
            results = [pool.apply_async(_wrapper, args=(DoubleQ, data,)) for i in range(config.exp.repeat)]
            returns = [r.get() for r in tqdm(results)]

        for rep in range(config.exp.repeat):
            DQ_EnvLogger[rep, ::] = returns[rep]['EnvLog']

        env_loggers['Double Q (lr={})'.format(lr)] = DQ_EnvLogger

    return env_loggers, loggers


def mmq_learning(env, config, T, Q, rng, estimators, env_loggers, loggers):
    """MMQ runner"""

    for _, lr in enumerate(config.mmq_learning.alpha):
        MMQ_EnvLogger = np.empty(( config.exp.repeat, config.exp.steps, 2))

        data = {
            'env': env,
            'config': config,
            'Q': copy.deepcopy(Q),
            'T': copy.deepcopy(T),
            'estimators': estimators,
            'rng': rng,
            'alpha': lr
        }

        with Pool(processes=config.exp.repeat) as pool:
            results = [pool.apply_async(_wrapper, args=(MaxminQ, data,)) for i in range(config.exp.repeat)]
            returns = [r.get() for r in tqdm(results)]

        for rep in range(config.exp.repeat):
            MMQ_EnvLogger[rep, ::] = returns[rep]['EnvLog']

        env_loggers['Maxmin Q n_{} (lr={})'.format(estimators, lr)] = MMQ_EnvLogger

    return env_loggers, loggers


def mmbq_learning(env, config, T, Q, rng, env_loggers, loggers, exp_name):
    """MMBQ runner"""
    for _, lr in enumerate(config.mmbq_learning.alpha):
        MMBQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        MMBQ_Bandit = np.empty((config.exp.repeat, config.exp.steps, config.mmbq_learning.max_estimators, 2))
        MMBQ_Estimator = np.empty((config.exp.repeat, config.exp.steps), dtype=np.int64)

        data = {
            'env': env,
            'config': config,
            'Q': copy.deepcopy(Q),
            'T': copy.deepcopy(T),
            'rng': rng,
            'alpha': lr
        }

        with Pool(processes=config.exp.repeat) as pool:
            results = [pool.apply_async(_wrapper, args=(MaxminBanditQ, data,)) for i in range(config.exp.repeat)]
            returns = [r.get() for r in tqdm(results)]

        for rep in range(config.exp.repeat):
            MMBQ_EnvLogger[rep, ::] = returns[rep]['EnvLog']
            MMBQ_Bandit[rep, ::] = returns[rep]['BanditLog']
            MMBQ_Estimator[rep, ::] = returns[rep]['EstimLog']

        env_loggers['Maxmin Bandit Q (lr={})'.format(lr)] = MMBQ_EnvLogger

        compress_pickle(join(exp_name, 'store', 'Maxmin Bandit Q_bandit (lr={}).pbz2'.format(lr)), MMBQ_Bandit)
        compress_pickle(join(exp_name, 'store', 'Maxmin Bandit Q_estimator (lr={}).pbz2'.format(lr)), MMBQ_Estimator)

    return env_loggers, loggers


def mmbq_v2_learning(env, config, T, Q, rng, env_loggers, loggers, exp_name):
    """MMBQ V2 learner"""
    for _, lr in enumerate(config.mmbq_learning.alpha):
        MMBQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        MMBQ_Bandit = np.empty((config.exp.repeat, config.exp.steps, config.mmbq_learning.max_estimators, 2))
        MMBQ_Estimator = np.empty((config.exp.repeat, config.exp.steps), dtype=np.int64)

        data = {
            'env': env,
            'config': config,
            'Q': copy.deepcopy(Q),
            'T': copy.deepcopy(T),
            'rng': rng,
            'alpha': lr
        }

        with Pool(processes=config.exp.repeat) as pool:
            results = [pool.apply_async(_wrapper, args=(MaxminBanditQ_v2, data,)) for i in range(config.exp.repeat)]
            returns = [r.get() for r in tqdm(results)]

        for rep in range(config.exp.repeat):
            MMBQ_EnvLogger[rep, ::] = returns[rep]['EnvLog']
            MMBQ_Bandit[rep, ::] = returns[rep]['BanditLog']
            MMBQ_Estimator[rep, ::] = returns[rep]['EstimLog']

        env_loggers['Maxmin Bandit Q v2 (lr={})'.format(lr)] = MMBQ_EnvLogger

        compress_pickle(join(exp_name, 'store', 'Maxmin Bandit Q v2_bandit (lr={}).pbz2'.format(lr)), MMBQ_Bandit)
        compress_pickle(join(exp_name, 'store', 'Maxmin Bandit Q v2_estimator (lr={}).pbz2'.format(lr)), MMBQ_Estimator)

    return env_loggers, loggers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="Path to Config file", required=True)
    parser.add_argument("--seed", help="set numpy & env seed", type=int, default=0)
    parser.add_argument("-e", "--exp_name", help="Experiment name", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    rng = np.random.default_rng(args.seed)

    make_dirs(join(args.exp_name, 'store'))
    with open(join(args.exp_name, 'config.yaml'), 'w') as f:
        json.dump(config, f, indent=2)

    if config.model == "gym":
        env = gym.make(config.env)
        env = NoisyReward(env, rng, config.std)
        env.seed(args.seed)
    else:
        assert False, 'Invalid env model args...'

    # Save experiment logs
    env_loggers = {}    # Log rewards, (config.exp.repeat, config.exp.steps)
    loggers = {}        # Log Q values, (config.exp.repeat, config.exp.steps, TiledQTable.shape)


    T, Q = set_initial_values(config, env, rng)

    # Vanilla Q-learning
    if 'q' in config.perform:
        print("Vanilla Q Learning")
        env_loggers, loggers = q_learning(env, config, T, Q, rng, env_loggers, loggers)

        # Save experiment values
        save_experiments(
            env_loggers,
            {},
            {},
            args.exp_name
        )

    # Double Q-learning
    if 'dq' in config.perform:
        print("Double Q Learning")
        env_loggers, loggers = dq_learning(env, config, T, Q, rng, env_loggers, loggers)

        # Save experiment values
        save_experiments(
            env_loggers,
            {},
            {},
            args.exp_name
        )

    # Perform Maxmin for 'n' different estimator counts
    if 'mmq' in config.perform:
        for estimators in config.mmq_learning.estimator_pools:
            print("Maxmin Q learning, n = {}".format(estimators))
            env_loggers, loggers = mmq_learning(env, config, T, Q, rng, estimators, env_loggers, loggers)

            # Save experiment values
            save_experiments(
                env_loggers,
                {},
                {},
                args.exp_name
            )

    # Maxmin Bandit Q learning
    if 'mmbq' in config.perform:
        print("Maxmin Bandit Q learning")
        env_loggers, loggers = mmbq_learning(env, config, T, Q, rng, env_loggers, loggers, args.exp_name)

        # Save experiment values
        save_experiments(
            env_loggers,
            {},
            {},
            args.exp_name
        )

    # Maxmin Bandit Q learning v2
    if 'mmbq_v2' in config.perform:
        print("Maxmin Bandit Q learning V2")
        env_loggers, loggers = mmbq_v2_learning(env, config, T, Q, rng, env_loggers, loggers, args.exp_name)

        # Save experiment values
        save_experiments(
            env_loggers,
            {},
            {},
            args.exp_name
        )
