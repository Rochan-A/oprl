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


def set_initial_values(config, env):
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
        Q = np.random.normal(0, 0.01, (T.n_tiles, env.action_space.n)).astype(dtype=np.float64)
    elif config.init.Q == 'opt':
        Q = np.zeros((T.n_tiles, env.action_space.n), dtype=np.float64)
        Q.fill(1.0)
    elif config.init.Q == 'pes':
        Q = np.zeros((T.n_tiles, env.action_space.n), dtype=np.float64)
        Q.fill(-1.0)
    else:
        assert False, 'unknown Q value initialization'

    return T, Q


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

    make_dirs(join(args.exp_name, 'store'))
    with open(join(args.exp_name, 'config.yaml'), 'w') as f: 
        json.dump(config, f, indent=2)

    if config.model == "gym":
        env = gym.make(config.env)
        env = NoisyReward(env, config.std)
        env.seed(args.seed)
    else:
        assert False, 'Invalid env model args...'
        quit()

    # Save experiment logs
    env_loggers = {}    # Log rewards, (config.exp.repeat, config.exp.steps)
    loggers = {}        # Log Q values, (config.exp.repeat, config.exp.steps, TiledQTable.shape)


    T, Q = set_initial_values(config, env)

    # Vanilla Q-learning
    if 'q' in config.perform:
        print("Vanilla Q Learning")
        # VQ_Logger = np.empty((config.exp.repeat, config.exp.steps) + Q.shape)
        VQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        for rep in tqdm(range(config.exp.repeat)):
            _, VQ_EnvLogger[rep, ::] = Q_learning(
                env,
                config,
                copy.deepcopy(Q),
                copy.deepcopy(T)
            )
        env_loggers['Vanilla Q'] = VQ_EnvLogger
        # loggers['Vanilla Q'] = VQ_Logger


    # Double Q-learning
    if 'dq' in config.perform:
        print("Double Q Learning")
        # DQ1_Logger = np.empty((config.exp.repeat, config.exp.steps) + Q.shape)
        # DQ2_Logger = np.empty((config.exp.repeat, config.exp.steps) + Q.shape)
        DQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        for rep in tqdm(range(config.exp.repeat)):
            _, _, DQ_EnvLogger[rep, ::] = DoubleQ(
                env,
                config,
                copy.deepcopy(Q),
                copy.deepcopy(T)
            )
        env_loggers['Double Q'] = DQ_EnvLogger
        # loggers['Double Q1'] = DQ1_Logger
        # loggers['Double Q2'] = DQ2_Logger


    # Pessimistic Q-learning
    if 'pq' in config.perform:
        print("Pessimistic Q Learning")
        # PQ_Logger = np.empty((config.exp.repeat, config.exp.steps) + Q.shape)
        PQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        for rep in tqdm(range(config.exp.repeat)):
            _, PQ_EnvLogger[rep, ::] = PessimisticQ(
                env,
                config,
                copy.deepcopy(Q),
                copy.deepcopy(T)
            )
        env_loggers['Pessimistic Q'] = PQ_EnvLogger
        # loggers['Pessimistic Q'] = PQ_Logger


    # Perform Maxmin for 'n' different estimator counts
    if 'mmq' in config.perform:
        for estimators in config.mmq_learning.estimator_pools:
            print("Maxmin Q learning, n = {}".format(estimators))
            # MMQ_Logger = np.empty(( config.exp.repeat, config.exp.steps, Q.shape))
            MMQ_EnvLogger = np.empty(( config.exp.repeat, config.exp.steps, 2))
            for rep in tqdm(range(config.exp.repeat)):
                _, MMQ_EnvLogger[rep, ::] = MaxminQ(
                    env,
                    config,
                    estimators,
                    copy.deepcopy(Q),
                    copy.deepcopy(T)
                )
            env_loggers['Maxmin Q n_{}'.format(estimators)] = MMQ_EnvLogger
            # loggers['Maxmin Q n_{}'.format(estimators)] = MMQ_Logger


    # Mean Variance Q-learning
    if 'meanvar' in config.perform:
        print("Mean-Var Q Learning")
        # MVQ_Logger = np.empty((config.exp.repeat, config.exp.steps) + Q.shape)
        MVQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        for rep in tqdm(range(config.exp.repeat)):
            _, MVQ_EnvLogger[rep, ::] = MeanVarianceQ(
                env,
                config,
                copy.deepcopy(Q),
                copy.deepcopy(T)
            )
        env_loggers['Mean-Var Q'] = MVQ_EnvLogger
        # loggers['Mean-Var Q'] = MVQ_Logger


    # Maxmin Bandit Q learning
    if 'mmbq' in config.perform:
        print("Maxmin Bandit Q learning")
        # MMBQ_Logger = np.empty((config.exp.repeat, config.exp.steps) + Q.shape)
        MMBQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        MMBQ_Bandit = np.empty((config.exp.repeat, config.exp.steps, config.mmbq_learning.max_estimators, 2))
        MMBQ_Estimator = np.empty((config.exp.repeat, config.exp.steps), dtype=np.int64)
        for rep in tqdm(range(config.exp.repeat)):
            _, MMBQ_EnvLogger[rep, ::], MMBQ_Bandit[rep, ::], MMBQ_Estimator[rep] = MaxminBanditQ(
                env,
                config,
                copy.deepcopy(Q),
                copy.deepcopy(T)
            )
        env_loggers['Maxmin Bandit Q'] = MMBQ_EnvLogger
        # loggers['Maxmin Bandit Q'] = MMBQ_Logger
        compress_pickle(join(args.exp_name, 'store', 'Maxmin Bandit Q_bandit.pbz2'), MMBQ_Bandit)
        compress_pickle(join(args.exp_name, 'store', 'Maxmin Bandit Q_estimator.pbz2'), MMBQ_Estimator)


    # Maxmin Bandit Q learning v2
    if 'mmbq_v2' in config.perform:
        print("Maxmin Bandit Q learning V2")
        # MMBQ_Logger = np.empty((config.exp.repeat, config.exp.steps) + Q.shape)
        MMBQ_EnvLogger = np.empty((config.exp.repeat, config.exp.steps, 2))
        MMBQ_Bandit = np.empty((config.exp.repeat, config.exp.steps, config.mmbq_learning.max_estimators, 2))
        MMBQ_Estimator = np.empty((config.exp.repeat, config.exp.steps), dtype=np.int64)
        for rep in tqdm(range(config.exp.repeat)):
            _, MMBQ_EnvLogger[rep, ::], MMBQ_Bandit[rep, ::], MMBQ_Estimator[rep] = MaxminBanditQ_v2(
                env,
                config,
                copy.deepcopy(Q),
                copy.deepcopy(T)
            )
        env_loggers['Maxmin Bandit Q v2'] = MMBQ_EnvLogger
        # loggers['Maxmin Bandit Q v2'] = MMBQ_Logger
        compress_pickle(join(args.exp_name, 'store', 'Maxmin Bandit Q v2_bandit.pbz2'), MMBQ_Bandit)
        compress_pickle(join(args.exp_name, 'store', 'Maxmin Bandit Q v2_estimator.pbz2'), MMBQ_Estimator)


    # Save experiment values
    save_experiments(
        env_loggers,
        {},
        {},
        args.exp_name
    )
