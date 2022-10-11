from os.path import join

import argparse
import yaml
from easydict import EasyDict
import glob

from envs import Converted, EnvModel, Maps, MapsEnvModel
from utils.data import *

def load_experiments_and_env(path, algos, plots):
    """Save experiment data"""

    load_data = {}
    if 'val_iter' in algos:
        x = decompress_pickle(
            join(path, 'store', 'Optimal.pbz2')
        )
        load_data['v_star'] = x['v_star']
        load_data['Q_star'] = x['Q_star']
    else:
        load_data['v_star'], load_data['Q_star'] = None, None

    load_data['env_loggers'] = {}
    if 'cum_r' in plots:
        env_files = glob.glob(join(path, 'store', '*_env.pbz2'))
        for key in env_files:
            load_data['env_loggers'][' '.join(key.split('/')[-1].split('_')[:-1])] = \
                    decompress_pickle(key)

    load_data['q_loggers'] = {}
    if 'q_vals' in plots or 'v_vals' in plots:
        logger_files = glob.glob(join(path, 'store', '*_qvals.pbz2'))
        for key in logger_files:
            load_data['q_loggers'][' '.join(key.split('/')[-1].split('_')[:-1])] = \
                    decompress_pickle(key)

    load_data['visits'] = {}
    if 'heat_map' in plots or 'visits' in plots:
        visit_files = glob.glob(join(path, 'store', '*_visits.pbz2'))
        for key in visit_files:
            load_data['visits'][' '.join(key.split('/')[-1].split('_')[:-1])] = \
                    decompress_pickle(key)

    load_data['bandit'] = {'est': {}, 'q': {}}

    if 'mmbq' in algos and 'bandit' in plots:
        load_data['bandit']['q']['Maxmin Bandit Q'] = decompress_pickle(
            join(path, 'store', 'Maxmin Bandit Q_bandit (lr=0.01).pbz2')
        )
        load_data['bandit']['est']['Maxmin Bandit Q'] = decompress_pickle(
            join(path, 'store', 'Maxmin Bandit Q_estimator (lr=0.01).pbz2')
        )

    if 'mmbqv2' in algos and 'bandit' in plots:
        load_data['bandit']['q']['Maxmin Bandit Q v2'] = decompress_pickle(
            join(path, 'store', 'Maxmin Bandit Q v2_bandit (lr=0.01).pbz2')
        )
        load_data['bandit']['est']['Maxmin Bandit Q v2'] = decompress_pickle(
            join(path, 'store', 'Maxmin Bandit Q v2_estimator (lr=0.01).pbz2')
        )

    with open(join(path, 'config.yaml'), 'r') as f:
        exp_config = yaml.safe_load(f)
    exp_config = EasyDict(exp_config)

    env = None
    if "file" in exp_config.model or "gridworld" in exp_config.model:

        # Gridworld or old env
        if exp_config.model == "gridworld" \
            or exp_config.model == 'file':
            env = Converted(exp_config.env, exp_config.seed)
            env = EnvModel(env)
            _ = env.reset()

        # New env
        elif exp_config.model == 'filev2':
            env = Maps(exp_config.env, exp_config.seed)
            env = MapsEnvModel(env)

        if 'file' in exp_config.model:
            env.load_map(exp_config.map_path)
            _ = env.reset()

    else:
        print('Got unknown env, not plotting Q and V...')

    return load_data, env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", help="Path to plotter config file", required=True)
    parser.add_argument("-p", "--path", help="Path to experiment folder", type=str, required=True)

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    # Get all algos whose data we need to read
    algos = []
    for val in config.perform:
        algos.extend([i for i in config[val].algos])
    algos = set(algos)

    load_data, env = load_experiments_and_env(args.path, algos, config.perform)

    try:
        exp_name = join(args.path).split('/')[-2]
    except:
        print('Ensure path is with trailing backslash...')
        quit()

    if 'cum_r' in config.perform:
        # Plot mean cummulative reward
        plot_mean_cum_rewards(
            load_data['env_loggers'],
            exp_name,
            do_smooth=config.cum_r.smooth,
            std_factor=config.cum_r.std_factor,
            fmt=config.fmt
            )

    if 'q_vals' in config.perform:
        # plot Q and V
        plot_Q_values(
            env,
            load_data['Q_star'],
            load_data['q_loggers'],
            exp_name
        )

    if 'v_vals' in config.perform:
        plot_V_values(
            env,
            [load_data['v_star'], load_data['Q_star']],
            load_data['q_loggers'],
            exp_name
        )

    if 'visits' in config.perform:
        plot_visits(
            load_data['visits'],
            config.visits.s_a,
            exp_name
            )

    if 'bandit' in config.perform:
        plot_bandit(
            load_data['bandit'],
            exp_name
            )