#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

from envs.gridworld import *
import yaml
from easydict import EasyDict

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    # if args.seed != -1:
    #     env.seed(args.seed)

    obs = env.reset()

    # if hasattr(env, 'mission'):
    #     print('Mission: %s' % env.mission)
    #     window.set_caption(env.mission)

    # redraw(obs)

def step(action):
    obs, reward, done, info = env.step(action)
    print('step=%s, obs=%s, reward=%.2f, done=%s' % (env.step_count, env.S[obs], reward, done))

    if done:
        print('done!')
        reset()
    # else:
    #     redraw(obs)

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.up)
        return
    if event.key == 'down':
        step(env.actions.down)
        return


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Path to Config file", required=True)
parser.add_argument("--seed", help="set numpy & env seed", type=int, default=0)

args = parser.parse_args()
rng = np.random.default_rng(args.seed)

with open(args.config) as f:
    config = yaml.safe_load(f)
config = EasyDict(config)

env = Maps(config.env, args.seed)
env.load_map(config.map_path)

# env = DistanceBonus(env)
# env = StateBonus(env)
# env = ActionBonus(env)

_ = env.reset()

window = Window('gym_minigrid')
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
